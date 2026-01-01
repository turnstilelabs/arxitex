import asyncio
import json
import os
import sqlite3
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

        if target_arxiv_id is not None:
            # When a specific target is requested, keep the existing behavior
            # (ignore citation-based ordering) and just look at the queue.
            all_discovered_papers = self.components.discovery_index.get_pending_papers()
        elif self.min_citations is not None:
            # Restrict to papers with citation_count >= min_citations and
            # order them by citation_count DESC (then base_id ASC).
            all_discovered_papers = self._get_citation_filtered_pending_papers()
        else:
            all_discovered_papers = self.components.discovery_index.get_pending_papers()

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
