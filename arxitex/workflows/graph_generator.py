import os
import json
from loguru import logger
from conjecture.workflows.runner import ArxivPipelineComponents,AsyncArxivWorkflowRunner
from conjecture.graph.builder import agenerate_artifact_graph

class AsyncGraphGeneratorWorkflow(AsyncArxivWorkflowRunner):
    
    def __init__(self, components: ArxivPipelineComponents, use_llm: bool, 
                 max_concurrent_tasks: int = 10, force: bool = False):
        super().__init__(components, max_concurrent_tasks, force)
        self.use_llm = use_llm
        self.graphs_output_dir = os.path.join(self.components.output_dir, "graphs")
        os.makedirs(self.graphs_output_dir, exist_ok=True)
        
    async def _process_single_paper(self, paper: dict) -> dict:
        paper_id = paper['arxiv_id']
        logger.info(f"Starting graph generation for {paper_id}...")
        
        try:
            graph_data = await agenerate_artifact_graph(paper_id, self.use_llm)
        except Exception as e:
            reason = f"Error during agenerate_artifact_graph: {e}"
            logger.error(f"CONTROLLED_FAILURE for {paper_id}: {reason}", exc_info=True)
            self.components.paper_index.update_processed_papers_index(paper_id, status='failure', reason=reason)
            return {"status": "failure", "arxiv_id": paper_id, "reason": reason}

        if not graph_data or not graph_data.get("nodes"):
            reason = "No artifacts/nodes found after processing."
            logger.warning(f"CONTROLLED_FAILURE for {paper_id}: {reason}")
            self.components.paper_index.update_processed_papers_index(paper_id, status='failure', reason=reason)
            return {"status": "failure", "arxiv_id": paper_id, "reason": reason}

        safe_paper_id = paper_id.replace('/', '_')
        graph_filename = f"{safe_paper_id}.json"
        graph_filepath = os.path.join(self.graphs_output_dir, graph_filename)
        
        with open(graph_filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)

        stats = graph_data.get("stats", {})
        self.components.paper_index.update_processed_papers_index(
            paper_id, status='success', output_path=graph_filepath, stats=stats
        )

        logger.info(f"SUCCESS: Saved graph for {paper_id}")
        return {"status": "success", "arxiv_id": paper_id, "output_path": graph_filepath, "stats": stats}