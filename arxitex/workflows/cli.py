import argparse
import asyncio
import os
import sqlite3
from pathlib import Path

from loguru import logger

from arxitex.db.error_utils import classify_processing_error
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.workflows.discover import DiscoveryWorkflow
from arxitex.workflows.processor import ProcessingWorkflow
from arxitex.workflows.runner import ArxivPipelineComponents
from arxitex.workflows.utils import save_graph_data

os.environ["RICH_QUIET"] = "True"
os.environ["TQDM_DISABLE"] = "1"


async def process_single_paper(arxiv_id: str, args):
    """
    Handles a single paper by running the temporary download and processing logic.
    """
    logger.info(f"Starting end-to-end processing for single paper: {arxiv_id}")

    components = ArxivPipelineComponents(output_dir=args.output_dir)

    if not args.force and components.processing_index.is_successfully_processed(
        arxiv_id
    ):
        logger.warning(
            f"Paper {arxiv_id} already successfully processed. Use --force to override."
        )
        return {"status": "skipped"}

    try:
        temp_base_dir = Path(components.output_dir) / "temp_processing"

        results = await agenerate_artifact_graph(
            arxiv_id=arxiv_id,
            enrich_content=args.enrich_content,
            infer_dependencies=args.infer_dependencies,
            source_dir=temp_base_dir,
        )

        graph = results.get("graph")
        if not graph or not graph.nodes:
            raise ValueError("Graph generation resulted in an empty or invalid graph.")

        graph_data = graph.to_dict(arxiv_id=arxiv_id)
        graphs_output_dir = os.path.join(components.output_dir, "graphs")
        os.makedirs(graphs_output_dir, exist_ok=True)
        graph_filepath = save_graph_data(arxiv_id, graphs_output_dir, graph_data)

        components.processing_index.update_processed_papers_status(
            arxiv_id,
            status="success",
            output_path=str(graph_filepath),
            stats=graph_data.get("stats", {}),
        )
        logger.info(
            f"SUCCESS: Processed {arxiv_id} and saved graph to {graph_filepath}"
        )
        return {"status": "success"}

    except Exception as e:
        err = classify_processing_error(e)
        logger.error(
            f"End-to-end processing failed for {arxiv_id} "
            f"[{err.code} @ {err.stage}]: {err.message}",
            exc_info=True,
        )
        components.processing_index.update_processed_papers_status(
            arxiv_id,
            status="failure",
            **err.to_details_dict(),
        )
        return {
            "status": "failure",
            "arxiv_id": arxiv_id,
            **err.to_details_dict(),
        }


async def main():
    """Parses command-line arguments and runs the selected workflow."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    default_output_dir = project_root / "pipeline_output"

    parser = argparse.ArgumentParser(
        description="ArxiTex: A pipeline for discovering and processing ArXiv papers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Directory for all outputs (default: {default_output_dir})",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )
    # --- 'single' command ---
    parser_single = subparsers.add_parser(
        "single", help="Temporarily download and process a single paper by its ID."
    )
    parser_single.add_argument(
        "arxiv_id", help="The arXiv ID to process (e.g., '2305.15334')."
    )
    parser_single.add_argument(
        "--enrich-content",
        action="store_true",
        help="Use LLM to find and synthesize term definitions.",
    )
    parser_single.add_argument(
        "--infer-dependencies",
        action="store_true",
        help="Use LLM to infer dependencies between artifacts.",
    )
    parser_single.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if it's in the index.",
    )

    # --- 'discover' command ---
    parser_discover = subparsers.add_parser(
        "discover",
        help="Find new paper IDs from ArXiv and add them to the processing queue.",
    )
    parser_discover.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="ArXiv API search query (e.g., 'cat:math.GR').",
    )
    parser_discover.add_argument(
        "-n",
        "--max-papers",
        type=int,
        default=1000,
        help="Target number of papers to discover in this run.",
    )
    parser_discover.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help="Number of papers to fetch from the API in each batch.",
    )

    # --- 'process' command ---
    parser_process = subparsers.add_parser(
        "process",
        help="Process papers from the queue (downloads temporarily to generate graphs).",
    )
    parser_process.add_argument(
        "-n",
        "--max-papers",
        type=int,
        default=50,
        help="Maximum number of papers from the queue to process in this run.",
    )
    parser_process.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent processing tasks (match to CPU cores).",
    )
    parser_process.add_argument(
        "--enrich-content",
        action="store_true",
        help="Use LLM to find and synthesize term definitions for papers in the batch.",
    )
    parser_process.add_argument(
        "--infer-dependencies",
        action="store_true",
        help="Use LLM to infer dependencies between artifacts for papers in the batch.",
    )
    parser_process.add_argument(
        "--format-for-search",
        action="store_true",
        help="Additionally, transform and append artifacts to a .jsonl file.",
    )
    parser_process.add_argument(
        "--persist-db",
        action="store_true",
        help="Persist normalized artifacts/edges/definitions into SQLite (arxitex_indices.db).",
    )
    parser_process.add_argument(
        "--mode",
        type=str,
        choices=["raw", "defs", "full"],
        default="raw",
        help=(
            "Extraction mode: 'raw' (regex only, no LLM), "
            "'defs' (LLM definitions/terms, no dependency inference), "
            "'full' (defs + dependency inference)."
        ),
    )

    # --- 'reprocess-paper' command ---
    parser_reprocess = subparsers.add_parser(
        "reprocess-paper",
        help=(
            "Reset DB state and reprocess a single paper by ID "
            "(discover + process) with the chosen mode."
        ),
    )
    parser_reprocess.add_argument(
        "arxiv_id", help="The arXiv ID to reprocess (e.g., '2305.15334')."
    )
    parser_reprocess.add_argument(
        "--mode",
        type=str,
        choices=["raw", "defs", "full"],
        default="raw",
        help=(
            "Extraction mode: 'raw' (regex only, no LLM), "
            "'defs' (LLM definitions/terms, no dependency inference), "
            "'full' (defs + dependency inference)."
        ),
    )
    parser_reprocess.add_argument(
        "--enrich-content",
        action="store_true",
        help="Backwards-compat: if set, will upgrade mode to 'defs' unless already 'full'.",
    )
    parser_reprocess.add_argument(
        "--infer-dependencies",
        action="store_true",
        help="Backwards-compat: if set, will force mode to 'full'.",
    )
    parser_reprocess.add_argument(
        "--persist-db",
        action="store_true",
        help="Persist normalized artifacts/edges/definitions into SQLite.",
    )
    parser_reprocess.add_argument(
        "--format-for-search",
        action="store_true",
        help="Additionally append artifacts to a JSONL search index.",
    )
    parser_reprocess.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent processing tasks (default: 1).",
    )
    parser_reprocess.add_argument(
        "--reset-modes",
        nargs="*",
        default=None,
        help=(
            "Optional list of ingestion modes to reset in paper_ingestion_state "
            "(e.g. raw defs full). Default: reset all modes for that paper."
        ),
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
            batch_size=args.batch_size,
        )

    elif args.command == "process":
        components = ArxivPipelineComponents(output_dir=args.output_dir)

        # Backwards compatibility: if the user still uses the old flags,
        # auto-select an equivalent mode.
        mode = args.mode
        if args.infer_dependencies:
            mode = "full"
        elif args.enrich_content and args.mode == "raw":
            mode = "defs"

        workflow = ProcessingWorkflow(
            components=components,
            enrich_content=args.enrich_content,
            infer_dependencies=args.infer_dependencies,
            max_concurrent_tasks=args.workers,
            format_for_search=args.format_for_search,
            persist_db=args.persist_db,
            mode=mode,
        )
        await workflow.run(max_papers=args.max_papers)

    elif args.command == "reprocess-paper":
        components = ArxivPipelineComponents(output_dir=args.output_dir)
        db_path = components.db_path

        logger.info(f"Resetting processed state for {args.arxiv_id} in {db_path}...")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()

            # Clear processed_papers status
            cur.execute(
                "DELETE FROM processed_papers WHERE arxiv_id = ?",
                (args.arxiv_id,),
            )

            # Clear ingestion state for this paper (optionally per-mode)
            if args.reset_modes:
                placeholders = ",".join("?" * len(args.reset_modes))
                sql = (
                    "DELETE FROM paper_ingestion_state "
                    "WHERE paper_id = ? AND mode IN (" + placeholders + ")"
                )
                cur.execute(sql, (args.arxiv_id, *args.reset_modes))
            else:
                cur.execute(
                    "DELETE FROM paper_ingestion_state WHERE paper_id = ?",
                    (args.arxiv_id,),
                )

            conn.commit()
        finally:
            conn.close()

        # Re-discover this paper by ID and add it to the discovery queue
        search_query = f"id:{args.arxiv_id}"
        logger.info(f"Fetching metadata from ArXiv for {args.arxiv_id}...")

        response_text = components.arxiv_api.fetch_papers(
            search_query, start=0, batch_size=1
        )
        if not response_text:
            logger.error(f"No response from ArXiv for {search_query}.")
            return 1

        entries_in_batch, _, entries = components.arxiv_api.parse_response(
            response_text
        )
        if not entries:
            logger.error(f"Paper {args.arxiv_id} not found on ArXiv.")
            return 1

        paper = components.arxiv_api.entry_to_paper(entries[0])
        if not paper:
            logger.error(f"Failed to parse ArXiv entry for {args.arxiv_id}.")
            return 1

        added_count = components.discovery_index.add_papers([paper])
        if added_count == 0:
            logger.info(
                f"Paper {args.arxiv_id} was already in the discovery queue; proceeding to process."
            )
        else:
            logger.info(f"Added {args.arxiv_id} to discovery queue.")

        # Derive effective mode with backwards-compat flags like the 'process' command
        mode = args.mode
        if args.infer_dependencies:
            mode = "full"
        elif args.enrich_content and args.mode == "raw":
            mode = "defs"

        workflow = ProcessingWorkflow(
            components=components,
            enrich_content=args.enrich_content,
            infer_dependencies=args.infer_dependencies,
            max_concurrent_tasks=args.workers,
            format_for_search=args.format_for_search,
            persist_db=args.persist_db,
            mode=mode,
        )
        # When reprocessing, restrict the workflow to the specific paper ID so
        # we don't accidentally pick the first pending paper from the queue.
        await workflow.run(max_papers=1, target_arxiv_id=args.arxiv_id)

    logger.info(f"Command '{args.command}' has completed.")
    return exit_code


def cli_main():
    """Synchronous wrapper for setuptools console_scripts entry point."""
    return asyncio.run(main())


if __name__ == "__main__":
    exit_code = cli_main()
    exit(exit_code)
