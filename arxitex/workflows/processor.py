import asyncio
import json
import os
from pathlib import Path

from filelock import FileLock
from loguru import logger

from arxitex.db.error_utils import classify_processing_error
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.llms.usage_context import llm_usage_context
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
        persist_db: bool = False,
        mode: str = "raw",
        dependency_mode: str = "auto",
        dependency_config: dict | None = None,
    ):
        super().__init__(components, max_concurrent_tasks)
        self.persist_db = persist_db
        self.mode = mode
        self.infer_dependencies = infer_dependencies
        self.enrich_content = enrich_content
        self.max_concurrent_tasks = max_concurrent_tasks
        self.format_for_search = format_for_search
        self.dependency_mode = dependency_mode
        self.dependency_config = dependency_config or {}

        self.graphs_base_dir = os.path.join(self.components.output_dir, "graphs")
        self.search_indices_base_dir = os.path.join(
            self.components.output_dir, "search_indices"
        )

        os.makedirs(self.graphs_base_dir, exist_ok=True)
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

            category_graph_dir = os.path.join(self.graphs_base_dir, category)
            os.makedirs(category_graph_dir, exist_ok=True)
            graph_data = graph.to_dict(arxiv_id=arxiv_id)
            graph_filepath = save_graph_data(arxiv_id, category_graph_dir, graph_data)
            logger.info(f"Saved primary graph for {arxiv_id} to {graph_filepath}")

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
                output_path=str(graph_filepath),
                stats=graph_data.get("stats", {}),
            )
            self.components.discovery_index.remove_paper(arxiv_id)

            logger.success(
                f"SUCCESS: Fully processed {arxiv_id} and removed from discovery queue."
            )

            return {
                "status": "success",
                "arxiv_id": arxiv_id,
                "output_path": str(graph_filepath),
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
