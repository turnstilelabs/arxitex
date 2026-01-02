#!/usr/bin/env python3
"""
Main script to extract dependency graphs from arXiv papers.

This script orchestrates the entire workflow and supports two extraction methods:
1. 'regex-only': A fast but basic method using regular expressions.
2. 'hybrid-llm': A powerful method using regex for node extraction and an
   LLM for semantic dependency analysis (requires OPENAI_API_KEY).
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from loguru import logger

from arxitex.downloaders.async_downloader import AsyncSourceDownloader
from arxitex.extractor.graph_building.graph_enhancer import GraphEnhancer
from arxitex.extractor.models import ArxivExtractorError
from arxitex.extractor.visualization import graph_viz


def get_examples_dir() -> Path:
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parents[1]
    examples_dir = project_root / "data"
    examples_dir.mkdir(exist_ok=True, parents=True)
    return examples_dir


async def agenerate_artifact_graph(
    arxiv_id: str,
    infer_dependencies: bool,
    enrich_content: bool,
    dependency_mode: str = "pairwise",
    dependency_config: Optional[dict] = None,
    source_dir: Optional[Path] = None,
    on_base_graph: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_enriched_node: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_dependency_edge: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_status: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict:
    """
    Orchestrates the full pipeline to generate a dependency graph fromve an arXiv paper.

    This function now acts as a high-level controller, delegating the graph
    building task to the appropriate class.

    Args:
        arxiv_id: The arXiv identifier (e.g., '2103.14030').
        infer_dependencies: Flag to enable Pass 2 (LLM dependency inference).
        enrich_content: Flag to enable Pass 3 (LLM content enrichment).
        source_dir: The base directory for temporary processing folders.

    Returns:
        A dictionary containing the full graph data and statistics.
    """
    log_location = source_dir if source_dir else "system's default temp directory"
    logger.debug(f"[{arxiv_id}] Creating temporary directory inside: {log_location}")

    with tempfile.TemporaryDirectory(
        prefix=f"{arxiv_id.replace('/', '_')}_", dir=source_dir
    ) as temp_dir:
        temp_path = Path(temp_dir)

        async with AsyncSourceDownloader(cache_dir=temp_path) as downloader:
            project_dir = await downloader.download_and_extract_source(arxiv_id)

            if not project_dir:
                raise ArxivExtractorError(
                    f"Failed to retrieve LaTeX content for {arxiv_id}"
                )

            logger.info(f"[{arxiv_id}] Instantiating GraphEnhancer...")
            enhancer = GraphEnhancer()

            dep_cfg = None
            if dependency_config:
                from arxitex.extractor.dependency_inference.dependency_mode import (
                    DependencyInferenceConfig,
                )

                dep_cfg = DependencyInferenceConfig(**dependency_config)

            graph, bank, artifact_to_terms_map = await enhancer.build_graph(
                project_dir=project_dir,
                source_file=f"arxiv:{arxiv_id}",
                infer_dependencies=infer_dependencies,
                enrich_content=enrich_content,
                dependency_mode=dependency_mode,
                dependency_config=dep_cfg,
                on_base_graph=on_base_graph,
                on_enriched_node=on_enriched_node,
                on_dependency_edge=on_dependency_edge,
                on_status=on_status,
            )

            return {
                "graph": graph,
                "bank": bank,
                "artifact_to_terms_map": artifact_to_terms_map,
            }


async def run_async_pipeline(args):
    try:
        dependency_config = {
            "auto_max_nodes_global": args.dependency_auto_max_nodes,
            "auto_max_tokens_global": args.dependency_auto_max_tokens,
            "max_total_pairs": args.dependency_max_pairs,
            "global_include_proofs": True,
            "global_proof_char_budget": args.dependency_global_proof_char_budget,
        }

        results = await agenerate_artifact_graph(
            arxiv_id=args.arxiv_id,
            infer_dependencies=args.infer_deps,
            enrich_content=args.enrich_content,
            dependency_mode=args.dependency_mode,
            dependency_config=dependency_config,
        )

        graph = results.get("graph")
        bank = results.get("bank")

        if not graph or not graph.nodes:
            logger.warning("No artifacts were extracted.")
            sys.exit(0)

        extractor_mode = "regex-only"
        if args.infer_deps and args.enrich_content:
            extractor_mode = "full-hybrid (deps + content)"
        elif args.infer_deps:
            extractor_mode = "hybrid (deps-only)"
        elif args.enrich_content:
            extractor_mode = "hybrid (content-only)"

        logger.info(f"Extraction completed using mode: {extractor_mode}")
        graph_data_to_save = graph.to_dict(
            arxiv_id=args.arxiv_id, extractor_mode=extractor_mode
        )

        examples_dir = get_examples_dir()
        graph_dir = examples_dir / "graphs"
        graph_dir.mkdir(exist_ok=True)
        graph_output_path = graph_dir / f"{args.arxiv_id.replace('/', '_')}.json"

        json_indent = 2 if args.pretty else None
        json_output = json.dumps(
            graph_data_to_save, indent=json_indent, ensure_ascii=False
        )
        graph_output_path.write_text(json_output, encoding="utf-8")
        logger.success(f"Document graph saved to {graph_output_path}")

        if bank and args.save_bank:
            logger.info("Serializing definition bank for output...")
            bank_data_to_save = await bank.to_dict()

            if bank_data_to_save:
                bank_dir = examples_dir / "definition_banks"
                bank_dir.mkdir(exist_ok=True)
                bank_output_path = (
                    bank_dir / f"{args.arxiv_id.replace('/', '_')}_bank.json"
                )

                json_output = json.dumps(
                    bank_data_to_save, indent=json_indent, ensure_ascii=False
                )
                bank_output_path.write_text(json_output, encoding="utf-8")
                logger.success(f"Definition bank saved to {bank_output_path}")
            else:
                logger.info("Definition bank was empty, no file was saved.")

        if args.visualize:
            if args.viz_output:
                viz_path = args.viz_output
            else:
                viz_dir = examples_dir / "viz"
                viz_dir.mkdir(exist_ok=True)
                viz_path = (
                    viz_dir / f"arxiv_{args.arxiv_id.replace('/', '_')}_graph.html"
                )

            graph_viz.create_visualization_html(graph_data_to_save, viz_path)
            try:
                file_url = viz_path.resolve().as_uri()
                webbrowser.open(file_url)
                logger.info(f"Opening visualization in browser: {file_url}")
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")

    except (ArxivExtractorError, FileNotFoundError, ValueError) as e:
        logger.error(f"A processing error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Defines and handles the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract mathematical dependency graphs from arXiv papers.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Fast regex-only extraction, output to stdout
  python pipeline.py 2211.11689

  # Regex + enrich artifact content with definitions
  python pipeline.py 2211.11689 --enrich-content -o enriched.json

  # Regex + infer dependency links
  python pipeline.py 2211.11689 --infer-deps --visualize -p

  # Regex + infer dependency links + enrich artifact + output to a specific JSON file
  python pipeline.py 2211.11689 --all-enhancements -o my_graph.json --pretty
""",
    )
    parser.add_argument(
        "arxiv_id", help="arXiv identifier (e.g., '2103.14030', 'math.AG/0601001')"
    )
    parser.add_argument(
        "--infer-deps",
        action="store_true",
        help="Enable Pass 3: Use LLM to infer dependency links between artifacts. This automatically enables content enrichment for best results.",
    )

    parser.add_argument(
        "--dependency-mode",
        type=str,
        choices=["pairwise", "global", "hybrid", "auto"],
        default="auto",
        help="Dependency inference mode when --infer-deps is enabled.",
    )
    parser.add_argument(
        "--dependency-auto-max-nodes",
        type=int,
        default=30,
        help="Auto-mode: max artifacts to allow global/hybrid.",
    )
    parser.add_argument(
        "--dependency-auto-max-tokens",
        type=int,
        default=12000,
        help="Auto-mode: max estimated tokens to allow global/hybrid.",
    )
    parser.add_argument(
        "--dependency-max-pairs",
        type=int,
        default=100,
        help=(
            "Global cap on the number of dependency pairs verified with the LLM "
            "per paper (applies to both hybrid and pairwise modes)."
        ),
    )
    parser.add_argument(
        "--dependency-global-proof-char-budget",
        type=int,
        default=1200,
        help="Global/Hybrid proposer: truncate each proof to this many chars.",
    )
    parser.add_argument(
        "--enrich-content",
        action="store_true",
        help="Enable Pass 2: Use LLM to enrich artifact content with prerequisite definitions.",
    )
    parser.add_argument(
        "--all-enhancements",
        action="store_true",
        help="Convenience flag to enable both --infer-deps and --enrich-content.",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const=True,
        default=None,
        help="Output JSON file. If path is omitted, a default is used. If flag is omitted, prints to stdout.",
    )
    parser.add_argument(
        "--save-bank",
        action="store_true",
        help="Save the definition bank to a separate JSON file if available.",
    )
    parser.add_argument(
        "-viz",
        "--visualize",
        action="store_true",
        help="Create and open an interactive HTML visualization.",
    )
    parser.add_argument(
        "--viz-output", type=Path, help="Custom path for visualization HTML file."
    )
    parser.add_argument(
        "-p", "--pretty", action="store_true", help="Pretty-print the JSON output."
    )

    args = parser.parse_args()
    if args.all_enhancements:
        args.infer_deps = True
        args.enrich_content = True

    if (args.infer_deps or args.enrich_content) and not os.getenv("OPENAI_API_KEY"):
        parser.error(
            "The --infer-deps and --enrich-content flags require the OPENAI_API_KEY environment variable."
        )

    asyncio.run(run_async_pipeline(args))


if __name__ == "__main__":
    main()
