import asyncio
from pathlib import Path
from loguru import logger
import argparse
import os

os.environ["RICH_QUIET"] = "True"
os.environ["TQDM_DISABLE"] = "1"

from arxitex.workflows.runner import ArxivPipelineComponents
from arxitex.workflows.discover import DiscoveryWorkflow
from arxitex.workflows.processor import ProcessingWorkflow
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.workflows.utils import save_graph_data

async def process_single_paper(arxiv_id: str, args):
    """
    Handles a single paper by running the temporary download and processing logic.
    """
    logger.info(f"Starting end-to-end processing for single paper: {arxiv_id}")
    
    components = ArxivPipelineComponents(output_dir=args.output_dir)
    
    if not args.force and components.processing_index.is_successfully_processed(arxiv_id):
        logger.warning(f"Paper {arxiv_id} already successfully processed. Use --force to override.")
        return {"status": "skipped"}

    try:
        temp_base_dir = Path(components.output_dir) / "temp_processing"
        
        results = await agenerate_artifact_graph(
            arxiv_id=arxiv_id,
            enrich_content=args.enrich_content,
            infer_dependencies=args.infer_dependencies,
            source_dir=temp_base_dir
        )

        graph = results.get("graph")
        if not graph or not graph.nodes:
            raise ValueError("Graph generation resulted in an empty or invalid graph.")
        
        graph_data = graph.to_dict(arxiv_id=arxiv_id)
        graphs_output_dir = os.path.join(components.output_dir, "graphs")
        os.makedirs(graphs_output_dir, exist_ok=True)
        graph_filepath = save_graph_data(arxiv_id, graphs_output_dir, graph_data)

        components.processing_index.update_processed_papers_index(
            arxiv_id, status='success', output_path=str(graph_filepath), stats=graph_data.get("stats", {})
        )
        logger.info(f"SUCCESS: Processed {arxiv_id} and saved graph to {graph_filepath}")
        return {"status": "success"}

    except Exception as e:
        reason = f"End-to-end processing failed for {arxiv_id}: {e}"
        logger.error(reason, exc_info=True)
        components.processing_index.update_processed_papers_index(
            arxiv_id, status='failure', reason=str(e)
        )
        return {"status": "failure", "reason": reason}


async def main():
    """Parses command-line arguments and runs the selected workflow."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    default_output_dir = project_root / "pipeline_output"

    parser = argparse.ArgumentParser(
        description="ArxiTex: A pipeline for discovering and processing ArXiv papers.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default=str(default_output_dir),
        help=f"Directory for all outputs (default: {default_output_dir})"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    # --- 'single' command ---
    parser_single = subparsers.add_parser(
        "single", 
        help="Temporarily download and process a single paper by its ID."
    )
    parser_single.add_argument("arxiv_id", help="The arXiv ID to process (e.g., '2305.15334').")
    parser_single.add_argument(
        '--enrich-content', 
        action='store_true', 
        help="Use LLM to find and synthesize term definitions."
    )
    parser_single.add_argument(
        '--infer-dependencies', 
        action='store_true', 
        help="Use LLM to infer dependencies between artifacts."
    )
    parser_single.add_argument('--force', action='store_true', help="Force re-processing even if it's in the index.")

    # --- 'discover' command ---
    parser_discover = subparsers.add_parser(
        "discover", 
        help="Find new paper IDs from ArXiv and add them to the processing queue."
    )
    parser_discover.add_argument(
        '-q', '--query', 
        type=str, 
        required=True, 
        help="ArXiv API search query (e.g., 'cat:math.GR')."
    )
    parser_discover.add_argument(
        '-n', '--max-papers', 
        type=int, 
        default=1000, 
        help="Target number of papers to discover in this run."
    )
    parser_discover.add_argument(
        '-b', '--batch-size', 
        type=int, 
        default=100, 
        help="Number of papers to fetch from the API in each batch."
    )

    # --- 'process' command ---
    parser_process = subparsers.add_parser(
        "process", 
        help="Process papers from the queue (downloads temporarily to generate graphs)."
    )
    parser_process.add_argument(
        '-n', '--max-papers', 
        type=int, 
        default=50, 
        help="Maximum number of papers from the queue to process in this run."
    )
    parser_process.add_argument(
        '-w', '--workers', 
        type=int, 
        default=4, 
        help="Number of concurrent processing tasks (match to CPU cores)."
    )
    parser_process.add_argument(
        '--enrich-content', 
        action='store_true', 
        help="Use LLM to find and synthesize term definitions for papers in the batch."
    )
    parser_process.add_argument(
        '--infer-dependencies', 
        action='store_true', 
        help="Use LLM to infer dependencies between artifacts for papers in the batch."
    )
    parser_process.add_argument(
    '--format-for-search', 
    action='store_true', 
    help="Additionally, transform and append artifacts to a .jsonl file."
    ) 

    args = parser.parse_args()
        
    exit_code = 0
    if args.command == "single":
        result = await process_single_paper(args.arxiv_id, args)
        if result.get("status") == "failure":
            exit_code = 1
    
    elif args.command == "discover":
        components = ArxivPipelineComponents(output_dir=args.output_dir)
        workflow = DiscoveryWorkflow(components)
        await workflow.run(
            search_query=args.query,
            max_papers=args.max_papers,
            batch_size=args.batch_size
        )

    elif args.command == "process":
        components = ArxivPipelineComponents(output_dir=args.output_dir)
        workflow = ProcessingWorkflow(
            components=components,
            enrich_content=args.enrich_content,
            infer_dependencies=args.infer_dependencies,
            max_concurrent_tasks=args.workers,
            format_for_search=args.format_for_search
        )
        await workflow.run(max_papers=args.max_papers)

    logger.info(f"Command '{args.command}' has completed.")
    return exit_code

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)