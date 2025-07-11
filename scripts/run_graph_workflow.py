import asyncio
from loguru import logger
import argparse
import os
os.environ["RICH_QUIET"] = "True"
os.environ["TQDM_DISABLE"] = "1"
from arxitex.workflows.runner import ArxivPipelineComponents
from arxitex.workflows.graph_generator import AsyncGraphGeneratorWorkflow

async def main(args):
    """Initializes and runs the graph generation workflow."""
    logger.info("Initializing workflow components...")
    
    pipeline_components = ArxivPipelineComponents(output_dir=args.output_dir)

    graph_workflow = AsyncGraphGeneratorWorkflow(
        components=pipeline_components,
        use_llm=args.use_llm,
        max_concurrent_tasks=args.workers,
        force=args.force
    )

    await graph_workflow.run(
        search_query=args.query,
        max_papers=args.max_papers,
        batch_size=args.batch_size
    )
    
    logger.info("Workflow has completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the ArXiv Artifact Graph Generation Workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-q', '--query', 
        type=str, 
        default="cat:math.GR", 
        help="ArXiv API search query."
    )
    parser.add_argument(
        '-n', '--max-papers', 
        type=int, 
        default=1, 
        help="Target number of papers to successfully process."
    )
    parser.add_argument(
        '-b', '--batch-size', 
        type=int, 
        default=1, 
        help="Number of papers to fetch from the API in each batch."
    )
    parser.add_argument(
        '-w', '--workers', 
        type=int, 
        default=1, 
        help="Number of concurrent processing tasks."
    )
    parser.add_argument(
        '-o', '--output-dir', 
        type=str, 
        default="pipeline_output", 
        help="Directory to store outputs and the persistent index."
    )
    parser.add_argument(
        '--use-llm', 
        action='store_true', 
        help="Flag to use the LLM-based extractor."
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help="Force re-processing of papers already in the index."
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
