import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock
from loguru import logger

from arxitex.db.error_utils import classify_processing_error
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.llms.usage_context import llm_usage_context
from arxitex.tools.citations_openalex import strip_arxiv_version
from arxitex.workflows.runner import ArxivPipelineComponents, AsyncWorkflowRunnerBase
from arxitex.workflows.utils import save_graph_data, transform_graph_to_search_format


class ProcessingWorkflow(AsyncWorkflowRunnerBase):
    """
    Processes papers from the DiscoveryIndex queue. For each paper, it performs
    a temporary download, generates a graph, saves the result, and cleans up.

    Optionally persists normalized outputs (artifacts/edges/definitions/term maps)
    into SQLite.
    """

    def __init__(
        self,
        components: ArxivPipelineComponents,
        infer_dependencies: bool,
        enrich_content: bool,
        max_concurrent_tasks: int,
        format_for_search=False,
        save_graph: bool = True,
        persist_db: bool = False,
        mode: str = "raw",
        dependency_mode: str = "auto",
        dependency_config: dict | None = None,
        min_citations: int | None = None,
        semantic_tags: bool = False,
        semantic_tag_model: str = "gpt-5-mini-2025-08-07",
        semantic_tag_concurrency: int = 4,
    ):
        super().__init__(components, max_concurrent_tasks)
        self.persist_db = persist_db
        self.mode = mode
        self.infer_dependencies = infer_dependencies
        self.enrich_content = enrich_content
        self.max_concurrent_tasks = max_concurrent_tasks
        self.format_for_search = format_for_search
        self.save_graph = save_graph
        self.dependency_mode = dependency_mode
        self.dependency_config = dependency_config or {}
        self.min_citations = min_citations
        self.semantic_tags = semantic_tags
        self.semantic_tag_model = semantic_tag_model
        self.semantic_tag_concurrency = semantic_tag_concurrency

        self.graphs_base_dir = os.path.join(self.components.output_dir, "graphs")
        self.search_indices_base_dir = os.path.join(
            self.components.output_dir, "search_indices"
        )

        # Only create output dirs that will actually be used.
        if self.save_graph:
            os.makedirs(self.graphs_base_dir, exist_ok=True)
        if self.format_for_search:
            os.makedirs(self.search_indices_base_dir, exist_ok=True)

    async def run(self, max_papers: int, target_arxiv_id: str | None = None):
        """Find and process papers from the discovery queue.

        Args:
            max_papers: Maximum number of papers to process in this run.
            target_arxiv_id: If provided, restrict processing to this specific
                arXiv ID (useful for reprocessing a single paper). When set,
                other discovered papers are ignored for this run.
        """
        logger.info("Starting 'processing' workflow...")

        # Certain failures are explicitly retryable and should be prioritized
        # in subsequent `process` runs so they don't starve behind high-citation
        # items (e.g. transient download blocks like arXiv reCAPTCHA).
        retry_priority_ids = self._get_retry_priority_arxiv_ids(
            reason_code="source_blocked_by_recaptcha"
        )

        if target_arxiv_id is not None:
            # When a specific target is requested, keep the existing behavior
            # (ignore citation-based ordering) and just look at the queue.
            all_discovered_papers = self.components.discovery_index.get_pending_papers()
        elif self.min_citations is not None:
            # Restrict to papers with citation_count >= min_citations and
            # order them by citation_count DESC (then base_id ASC).
            all_discovered_papers = self._get_citation_filtered_pending_papers()

            # Ensure transiently blocked papers (e.g. reCAPTCHA) are retried
            # even if they are currently below the citation threshold.
            if retry_priority_ids:
                all_pending = self.components.discovery_index.get_pending_papers()
                retry_papers = [
                    p for p in all_pending if p.get("arxiv_id") in retry_priority_ids
                ]

                # Deduplicate (preserve order): retries first, then citation-ordered.
                seen: set[str] = set()
                merged: list[dict] = []
                for p in retry_papers + all_discovered_papers:
                    aid = p.get("arxiv_id")
                    if not aid or aid in seen:
                        continue
                    seen.add(aid)
                    merged.append(p)
                all_discovered_papers = merged
        else:
            all_discovered_papers = self.components.discovery_index.get_pending_papers()

        # Stable partition: retryable reCAPTCHA-blocked papers first, without
        # otherwise changing ordering.
        if retry_priority_ids:
            prioritized = [
                p
                for p in all_discovered_papers
                if p.get("arxiv_id") in retry_priority_ids
            ]
            remaining = [
                p
                for p in all_discovered_papers
                if p.get("arxiv_id") not in retry_priority_ids
            ]
            all_discovered_papers = prioritized + remaining

        papers_to_process = []
        for paper_metadata in all_discovered_papers:
            if len(papers_to_process) >= max_papers:
                break

            arxiv_id = paper_metadata["arxiv_id"]

            # If a specific target is requested, skip all others.
            if target_arxiv_id is not None and arxiv_id != target_arxiv_id:
                continue

            if self.components.processing_index.is_successfully_processed(arxiv_id):
                logger.debug(
                    f"Skipping {arxiv_id}: already successfully processed. Removing from discovery queue."
                )
                self.components.discovery_index.remove_paper(arxiv_id)
                continue

            papers_to_process.append(paper_metadata)

        if not papers_to_process:
            if target_arxiv_id is not None:
                logger.info(
                    f"No pending papers found in discovery queue matching arxiv_id={target_arxiv_id}. Exiting."
                )
            else:
                logger.info(
                    "No new papers in the discovery queue are pending processing. Exiting."
                )
            return

        logger.info(
            f"Found {len(papers_to_process)} pending papers to process. Starting batch..."
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        tasks = [
            self._process_and_handle_paper(paper, semaphore)
            for paper in papers_to_process
        ]

        batch_results = await asyncio.gather(*tasks)
        self.results.extend(filter(None, batch_results))

        self._write_summary_report()
        logger.info("Processing workflow finished.")

    def _write_summary_report(self):
        """Categorize results and write a summary report for this processing run.

        Extends the base summary with an explicit 'skipped_in_this_run' bucket so
        that non-retryable cases like graph_empty are visible, rather than only
        being implied by the difference between 'attempted' and
        success/failure counts.
        """

        successful = [r for r in self.results if r and r.get("status") == "success"]
        failed = [r for r in self.results if r and r.get("status") == "failure"]
        skipped = [r for r in self.results if r and r.get("status") == "skipped"]

        summary = {
            "workflow_name": self.__class__.__name__,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "papers_attempted_in_this_run": len(self.results),
            "successful_in_this_run": successful,
            "failed_in_this_run": failed,
            "skipped_in_this_run": skipped,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"summary_{self.__class__.__name__.lower()}_{timestamp}.json"
        report_path = os.path.join(self.components.output_dir, report_filename)

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                f"Successfully wrote this run's summary report to {report_path}"
            )
        except Exception as e:
            logger.error(f"Could not write summary report: {e}")

    def _get_citation_filtered_pending_papers(self) -> list[dict]:
        """Return pending papers with citation_count >= min_citations.

        Papers are ordered by citation_count DESC, then base_id ASC. This
        method uses the normalized SQLite DB directly:

        - ``paper_citations`` for citation counts + global ordering
        - ``discovered_papers`` for metadata and best arxiv_id/version
        - ``processed_papers`` to skip already-successful papers

        This allows "VIP"-style processing (highly-cited first) while
        preserving stable resume behaviour across runs.
        """

        if self.min_citations is None:
            return self.components.discovery_index.get_pending_papers()

        db_path = self.components.db_path
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # 1) Map base_id -> best metadata from the discovery queue.
            cur = conn.execute("SELECT arxiv_id, metadata FROM discovered_papers")
            best_meta_by_base: dict[str, dict] = {}
            for row in cur:
                arxiv_id = row["arxiv_id"]
                base_id = strip_arxiv_version(arxiv_id)

                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    continue

                prev = best_meta_by_base.get(base_id)
                # Prefer the highest arxiv_id string for that base_id
                if prev is None or arxiv_id > prev.get("arxiv_id", ""):
                    best_meta_by_base[base_id] = meta

            # 2) Collect successfully processed arxiv_ids to skip on resume.
            processed_ok: set[str] = set()
            cur = conn.execute(
                "SELECT arxiv_id, status FROM processed_papers WHERE status LIKE 'success%'"
            )
            for row in cur:
                processed_ok.add(row["arxiv_id"])

            # 3) Walk citation records in global order, filtering by min_citations.
            cur = conn.execute(
                """
                SELECT paper_id AS base_id, citation_count
                FROM paper_citations
                WHERE citation_count >= ?
                ORDER BY citation_count DESC, paper_id ASC
                """,
                (self.min_citations,),
            )

            ordered: list[dict] = []
            for row in cur:
                base_id = row["base_id"]
                meta = best_meta_by_base.get(base_id)
                if not meta:
                    # Paper has citations but is not currently in the discovery queue.
                    continue

                arxiv_id = meta.get("arxiv_id")
                if arxiv_id in processed_ok:
                    # Already successfully processed in a prior run.
                    continue

                ordered.append(meta)

            logger.info(
                f"Citation-filtered pending papers: {len(ordered)} with citation_count >= {self.min_citations}"
            )

            return ordered
        finally:
            conn.close()

    def _get_retry_priority_arxiv_ids(self, reason_code: str) -> set[str]:
        """Return arxiv_ids that should be prioritized for retry.

        Currently used to ensure transient download failures (e.g. arXiv
        reCAPTCHA) get retried promptly on subsequent `process` runs.

        We store error details as JSON text; we avoid depending on SQLite's JSON
        extension by doing a simple substring match.
        """

        if not reason_code:
            return set()

        db_path = self.components.db_path
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            # "details" is stored as a JSON string. We match either the raw
            # code or a JSON-ish fragment; this is deliberately tolerant.
            like1 = f"%{reason_code}%"
            rows = conn.execute(
                """
                SELECT arxiv_id
                FROM processed_papers
                WHERE status = 'failure'
                  AND details LIKE ?
                """,
                (like1,),
            ).fetchall()
            return {row["arxiv_id"] for row in rows}
        except Exception:
            # Best-effort: if schema missing or DB unavailable, don't block.
            return set()
        finally:
            conn.close()

    async def _process_single_item(self, item: dict) -> dict:
        """
        Manages the lifecycle for one paper.

        Args:
            item (dict): A dictionary containing at least {'arxiv_id': '...'}.

        Returns:
            A dictionary with the status and results of the processing.
        """
        arxiv_id = item["arxiv_id"]
        paper_metadata = item
        category = paper_metadata.get("primary_category", "unknown").replace(".", "_")

        logger.info(
            f"[paper={arxiv_id}] Starting processing "
            f"(mode={self.mode}, dep_mode={self.dependency_mode})"
        )

        try:
            temp_base_dir = Path(self.components.output_dir) / "temp_processing"
            os.makedirs(temp_base_dir, exist_ok=True)

            # Mode drives whether we use LLM features.
            if self.mode == "raw":
                infer_dependencies = False
                enrich_content = False
            elif self.mode == "defs":
                infer_dependencies = False
                enrich_content = True
            elif self.mode == "full":
                infer_dependencies = True
                enrich_content = True
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            with llm_usage_context(paper_id=arxiv_id, mode=self.mode):
                results = await agenerate_artifact_graph(
                    arxiv_id=arxiv_id,
                    infer_dependencies=infer_dependencies,
                    enrich_content=enrich_content,
                    source_dir=temp_base_dir,
                    dependency_mode=self.dependency_mode,
                    dependency_config=self.dependency_config,
                )

            graph = results.get("graph")
            bank = results.get("bank")
            artifact_to_terms_map = results.get("artifact_to_terms_map")

            if not graph or not graph.nodes:
                raise ValueError("Graph generation resulted in an empty graph.")

            if self.semantic_tags:
                if not enrich_content:
                    logger.warning(
                        f"[paper={arxiv_id}] semantic_tags requested but enrich_content is disabled; skipping tags."
                    )
                else:
                    from arxitex.extractor.semantic_tagger import SemanticTagger

                    tagger = SemanticTagger(
                        model=self.semantic_tag_model,
                        concurrency=self.semantic_tag_concurrency,
                    )
                    await tagger.tag_nodes(graph.nodes)

            graph_data = graph.to_dict(arxiv_id=arxiv_id)

            graph_filepath = None
            if self.save_graph:
                category_graph_dir = os.path.join(self.graphs_base_dir, category)
                os.makedirs(category_graph_dir, exist_ok=True)
                graph_filepath = save_graph_data(
                    arxiv_id, category_graph_dir, graph_data
                )
                logger.info(f"Saved primary graph for {arxiv_id} to {graph_filepath}")
            else:
                logger.info(
                    f"Not saving graph JSON to disk for {arxiv_id} (persist_db={self.persist_db})."
                )

            if self.format_for_search:
                logger.info(f"Formatting artifacts from {arxiv_id} for search index...")
                artifact_to_terms_map = results.get("artifact_to_terms_map", {})

                searchable_artifacts = transform_graph_to_search_format(
                    graph_nodes=graph.nodes,
                    artifact_to_terms_map=artifact_to_terms_map,
                    paper_metadata=paper_metadata,
                )

                if searchable_artifacts:
                    search_index_path = os.path.join(
                        self.search_indices_base_dir, f"{category}.jsonl"
                    )
                    lock_path = os.path.join(
                        self.search_indices_base_dir, f"{category}.jsonl.lock"
                    )

                    with FileLock(lock_path):
                        with open(search_index_path, "a", encoding="utf-8") as f:
                            for artifact_doc in searchable_artifacts:
                                f.write(json.dumps(artifact_doc) + "\n")
                    logger.success(
                        f"Appended {len(searchable_artifacts)} artifacts from {arxiv_id} to search index."
                    )

            if self.persist_db:
                from arxitex.db.persistence import persist_extraction_result

                await persist_extraction_result(
                    db_path=self.components.db_path,
                    paper_metadata=paper_metadata,
                    graph=graph,
                    mode=self.mode,
                    bank=bank,
                    artifact_to_terms_map=artifact_to_terms_map,
                )

            self.components.processing_index.update_processed_papers_status(
                arxiv_id,
                status="success",
                output_path=str(graph_filepath) if graph_filepath else None,
                stats=graph_data.get("stats", {}),
            )
            self.components.discovery_index.remove_paper(arxiv_id)

            logger.success(
                f"SUCCESS: Fully processed {arxiv_id} and removed from discovery queue."
            )

            return {
                "status": "success",
                "arxiv_id": arxiv_id,
                "output_path": str(graph_filepath) if graph_filepath else None,
                "stats": graph_data.get("stats", {}),
            }

        except Exception as e:
            err = classify_processing_error(e)
            # Special-case: empty/invalid graphs are considered non-retryable.
            if err.code == "graph_empty":
                # Record as a skipped outcome in processed_papers for auditing.
                self.components.processing_index.update_processed_papers_status(
                    arxiv_id,
                    status="skipped_graph_empty",
                    **err.to_details_dict(),
                )

                # Remove from discovery queue and add to skipped index so we
                # never retry this paper in future runs.
                try:
                    self.components.discovery_index.remove_paper(arxiv_id)
                except Exception:
                    # Best-effort; if this fails we still mark it as skipped.
                    logger.warning(
                        f"Could not remove {arxiv_id} from discovery queue after graph_empty."
                    )

                try:
                    self.components.skipped_index.add(
                        arxiv_id,
                        "graph_empty_no_detectable_statements",
                    )
                except Exception:
                    logger.warning(
                        f"Could not record {arxiv_id} in skipped_papers after graph_empty."
                    )

                logger.warning(
                    f"SKIPPED (graph_empty): {arxiv_id} [{err.code} @ {err.stage}]: {err.message}. "
                    "Moved to skipped_papers and removed from discovery queue."
                )

                return {
                    "status": "skipped",
                    "arxiv_id": arxiv_id,
                    **err.to_details_dict(),
                }

            # Default: treat as retryable failure and leave in discovery queue.
            self.components.processing_index.update_processed_papers_status(
                arxiv_id,
                status="failure",
                **err.to_details_dict(),
            )
            logger.error(
                f"FAILURE processing {arxiv_id} [{err.code} @ {err.stage}]: {err.message}. "
                "It will remain in the discovery queue for a future retry.",
                exc_info=True,
            )
            raise e
