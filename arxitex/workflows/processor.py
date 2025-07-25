import asyncio
import os
import json
from pathlib import Path
from loguru import logger
from arxitex.workflows.runner import ArxivPipelineComponents,AsyncWorkflowRunnerBase
from arxitex.extractor.pipeline import agenerate_artifact_graph


class ProcessingWorkflow(AsyncWorkflowRunnerBase):
    """
    Processes papers from the DiscoveryIndex queue. For each paper, it performs
    a temporary download, generates a graph, saves the result, and cleans up.
    """
    def __init__(self, components: ArxivPipelineComponents, infer_dependencies: bool, 
        enrich_content: bool, max_concurrent_tasks: int):
        super().__init__(components, max_concurrent_tasks)
        self.infer_dependencies = infer_dependencies
        self.enrich_content = enrich_content
        self.graphs_output_dir = os.path.join(self.components.output_dir, "graphs")
        os.makedirs(self.graphs_output_dir, exist_ok=True)

    async def run(self, max_papers: int):
        """
        Finds and processes all papers in the discovery queue up to the max_papers limit.
        """
        logger.info("Starting 'processing' workflow...")
        pending_ids = self.components.discovery_index.get_pending_ids(limit=max_papers)
        
        if not pending_ids:
            logger.info("No papers found in the discovery queue to process. Exiting.")
            return

        logger.info(f"Found {len(pending_ids)} papers to process. Starting batch...")
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        tasks = [
            self._process_and_handle_paper({"arxiv_id": arxiv_id}, semaphore)
            for arxiv_id in pending_ids
        ]
        
        batch_results = await asyncio.gather(*tasks)
        self.results.extend(filter(None, batch_results))

        self._write_summary_report()
        logger.info("Processing workflow finished.")

    async def _process_single_item(self, item: dict) -> dict:
        """
        Manages the lifecycle for one paper. This is called by the parent's
        _process_and_handle_paper method.
        
        Args:
            item (dict): A dictionary containing at least {'arxiv_id': '...'}.
        
        Returns:
            A dictionary with the status and results of the processing.
        """
        arxiv_id = item['arxiv_id']
        
        if self.components.processing_index.is_paper_processed(arxiv_id):
            logger.warning(f"Skipping {arxiv_id}, already in final processing index. Removing from discovery queue.")
            self.components.discovery_index.remove_id(arxiv_id)
            return {"status": "skipped", "arxiv_id": arxiv_id}

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
            graph_data = graph.to_dict(arxiv_id=arxiv_id)
            graph_filepath = self._save_graph_data(arxiv_id, graph_data)

            self.components.processing_index.update_processed_papers_index(
                arxiv_id, status='success', output_path=str(graph_filepath), stats=graph_data.get("stats", {})
            )
            logger.info(f"SUCCESS: Processed {arxiv_id} and saved graph to {graph_filepath}")
            
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
            self.components.discovery_index.remove_id(arxiv_id)

    def _save_graph_data(self, arxiv_id: str, graph_data: dict) -> Path:
        """Saves the generated graph data to a persistent JSON file."""
        safe_paper_id = arxiv_id.replace('/', '_')
        graph_filename = f"{safe_paper_id}.json"
        graph_filepath = Path(self.graphs_output_dir) / graph_filename
        
        with open(graph_filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        return graph_filepath
    