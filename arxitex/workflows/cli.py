import asyncio
from loguru import logger
import argparse
import os

os.environ["RICH_QUIET"] = "True"
os.environ["TQDM_DISABLE"] = "1"

from arxitex.workflows.runner import ArxivPipelineComponents
from arxitex.workflows.graph_generator import AsyncGraphGeneratorWorkflow

async def process_single_paper(arxiv_id: str, args):
    paper = {"arxiv_id": arxiv_id}
    
    logger.info("Initializing workflow components...")
    pipeline_components = ArxivPipelineComponents(output_dir=args.output_dir)
    
    graph_workflow = AsyncGraphGeneratorWorkflow(
        components=pipeline_components,
        use_llm=args.use_llm,
        max_concurrent_tasks=1,
        force=args.force
    )
    
    result = await graph_workflow.process_single_paper(paper)    
    return result

async def process_batch(args):
    """Process papers using search query."""
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

async def main(args):
    """Main entry point that handles both single papers and batch processing."""
    if args.arxiv_id:
        result = await process_single_paper(args.arxiv_id, args)
        if result["status"] == "failure":
            logger.error(f"Failed to process {args.arxiv_id}: {result['reason']}")
            return 1
        else:
            logger.info(f"Successfully processed {args.arxiv_id}")
            return 0
    else:
        await process_batch(args)
        logger.info("Workflow has completed.")
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ArXiv Artifact Graph Generation - Single paper or batch processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:  
  # Process single paper with LLM enhancement
  python arxitex.workflows.cli --arxiv-id 2305.15334 --use-llm
  
  # Batch processing with search query
  python arxitex.workflows.cli --query "cat:math.GR" --max-papers 10 --workers 3
"""
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--arxiv-id',
        type=str,
        help="Process a single arXiv paper by ID (e.g., '2103.14030', 'math.AG/0601001')"
    )
    group.add_argument(
        '-q', '--query', 
        type=str, 
        default="cat:math.GR",
        help="ArXiv API search query (for batch processing)"
    )
    parser.add_argument(
        '-n', '--max-papers', 
        type=int, 
        default=1, 
        help="Target number of papers to successfully process (batch mode)"
    )
    parser.add_argument(
        '-b', '--batch-size', 
        type=int, 
        default=1, 
        help="Number of papers to fetch from the API in each batch"
    )
    parser.add_argument(
        '-w', '--workers', 
        type=int, 
        default=1, 
        help="Number of concurrent processing tasks"
    )
    
    # Common arguments
    parser.add_argument(
        '-o', '--output-dir', 
        type=str, 
        default="pipeline_output", 
        help="Directory to store outputs and the persistent index"
    )
    parser.add_argument(
        '--use-llm', 
        action='store_true', 
        help="Use the LLM-based extractor (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help="Force re-processing of papers already in the index"
    )
    
    args = parser.parse_args()
    
    exit_code = asyncio.run(main(args))
    exit(exit_code)