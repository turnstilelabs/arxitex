import asyncio
import os
import json
from filelock import FileLock
from pathlib import Path
from loguru import logger
from arxitex.workflows.runner import ArxivPipelineComponents,AsyncWorkflowRunnerBase
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.workflows.utils import save_graph_data, transform_graph_to_search_format


class ProcessingWorkflow(AsyncWorkflowRunnerBase):
    """
    Processes papers from the DiscoveryIndex queue. For each paper, it performs
    a temporary download, generates a graph, saves the result, and cleans up.
    """
    def __init__(self, components: ArxivPipelineComponents, infer_dependencies: bool, 
        enrich_content: bool, max_concurrent_tasks: int, format_for_search=False):
        super().__init__(components, max_concurrent_tasks)
        self.infer_dependencies = infer_dependencies
        self.enrich_content = enrich_content
        self.max_concurrent_tasks = max_concurrent_tasks
        self.format_for_search = format_for_search
        
        self.graphs_base_dir = os.path.join(self.components.output_dir, "graphs")
        self.search_indices_base_dir = os.path.join(self.components.output_dir, "search_indices")

        os.makedirs(self.graphs_base_dir, exist_ok=True)
        os.makedirs(self.search_indices_base_dir, exist_ok=True)

    async def run(self, max_papers: int):
        """
        Finds and processes all papers in the discovery queue up to the max_papers limit.
        """
        logger.info("Starting 'processing' workflow...")
        all_discovered_papers = self.components.discovery_index.get_pending_papers()

        papers_to_process = []
        for paper_metadata in all_discovered_papers:
            if len(papers_to_process) >= max_papers:
                break
            
            arxiv_id = paper_metadata['arxiv_id']
            if not self.components.processing_index.is_paper_processed(arxiv_id):
                papers_to_process.append(paper_metadata)
        
        if not papers_to_process:
            logger.info("No new papers in the discovery queue are pending processing. Exiting.")
            return

        logger.info(f"Found {len(papers_to_process)} pending papers to process. Starting batch...")
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
        arxiv_id = item['arxiv_id']
        paper_metadata = item
        category = paper_metadata.get('primary_category', 'unknown').replace('.', '_')

        try:
            temp_base_dir = Path(self.components.output_dir) / "temp_processing"
            os.makedirs(temp_base_dir, exist_ok=True)
            
            results = await agenerate_artifact_graph(
                arxiv_id=arxiv_id,
                infer_dependencies=self.infer_dependencies,
                enrich_content=self.enrich_content,
                source_dir=temp_base_dir
            )
            
            graph = results.get("graph")
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
                    arxiv_id=arxiv_id,
                    graph_nodes=graph.nodes,
                    artifact_to_terms_map=artifact_to_terms_map,
                    paper_metadata=paper_metadata
                )
                
                if searchable_artifacts:
                    search_index_path = os.path.join(self.search_indices_base_dir, f"{category}.jsonl")
                    lock_path = os.path.join(self.search_indices_base_dir, f"{category}.jsonl.lock")

                    with FileLock(lock_path):
                        with open(search_index_path, "a", encoding="utf-8") as f:
                            for artifact_doc in searchable_artifacts:
                                f.write(json.dumps(artifact_doc) + "\n")
                    logger.success(f"Appended {len(searchable_artifacts)} artifacts from {arxiv_id} to search index.")
            
            self.components.processing_index.update_processed_papers_index(
                arxiv_id, 
                status='success', 
                output_path=str(graph_filepath),
                stats=graph_data.get("stats", {})
            )
            
            logger.success(f"SUCCESS: Fully processed {arxiv_id}.")
            
            return {
                "status": "success",
                "arxiv_id": arxiv_id,
                "output_path": str(graph_filepath),
                "stats": graph_data.get("stats", {})
            }
            
        except Exception as e:
            self.components.processing_index.update_processed_papers_index(
                arxiv_id, status='failure', reason=str(e)
            )
            raise e
        
        finally:
            self.components.discovery_index.remove_paper(arxiv_id)        
        