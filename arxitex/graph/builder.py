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
from typing import Dict
from loguru import logger

from conjecture.source_downloader import SourceDownloader
from conjecture.graph import graph_builder
from conjecture.graph import hybrid_graph_builder
from conjecture.graph import visualization
from conjecture.graph.utils import ArxivExtractorError


async def agenerate_artifact_graph(arxiv_id: str, use_llm: bool) -> Dict:
    """
    Orchestrates the full pipeline to generate a dependency graph from an arXiv paper.
    Args:
        arxiv_id: The arXiv identifier (e.g., '2103.14030').
        use_llm: Flag to determine whether to use the hybrid LLM-based extractor.
    Returns:
        A dictionary containing the full graph data and statistics.
    """
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        raise ArxivExtractorError("The --use-llm flag requires the OPENAI_API_KEY environment variable to be set.")
    
    temp_dir_name = f"arxiv_{arxiv_id.replace('/', '_')}_"
    with tempfile.TemporaryDirectory(prefix=temp_dir_name) as temp_dir:
        temp_path = Path(temp_dir)
        downloader = SourceDownloader(cache_dir=str(temp_path))
        latex_content, tex_files = downloader.download_and_read_latex(arxiv_id, str(temp_path))
        
        if use_llm:
            logger.info("Using Hybrid (Regex + LLM) graph builder.")
            graph = await hybrid_graph_builder.build_graph_with_hybrid_model(latex_content)
        else:
            logger.info("Using Regex-only graph builder.")
            graph = graph_builder.build_graph_from_latex(latex_content)
        
        # Convert DocumentGraph to dictionary format for output
        nodes = []
        for _, node in enumerate(graph.nodes, 1):
            nodes.append(node.to_dict())
        
        edges = []
        for edge in graph.edges:
            edge_dict = {
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.dependency_type.value if edge.dependency_type is not None else None,
            }
            if edge.dependency:
                edge_dict["dependency"] = edge.dependency
            edges.append(edge_dict)
        
        logger.info("Finalizing graph statistics...")
        stats = graph.get_statistics()
        logger.info(f"Graph statistics: {stats}")
        
        return {
            "arxiv_id": arxiv_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": stats["total_nodes"],
                "edge_count": stats["total_edges"],
                "files_processed": len(tex_files),
                "extractor_used": "hybrid-llm" if use_llm else "regex-only"
            }
        }

async def run_async_pipeline(args):
    """Asynchronously runs the artifact generation and handles output."""
    try:
        graph_data = await agenerate_artifact_graph(args.arxiv_id, args.use_llm)
        
        if not graph_data["nodes"]:
             logger.warning("No artifacts were extracted. The paper might be empty, unscannable, or the extraction failed.")
             sys.exit(0)

        if args.output is not None:
            json_indent = 2 if args.pretty else None
            json_output = json.dumps(graph_data, indent=json_indent, ensure_ascii=False)

            if args.output is True:
                output_path = Path(f"{args.arxiv_id.replace('/', '_')}.json")
            else:
                output_path = Path(args.output)

            output_path.write_text(json_output, encoding='utf-8')
            logger.info(f"JSON output written to {output_path}")
        
        if args.visualize:
            viz_path = args.viz_output or Path(f"arxiv_{args.arxiv_id.replace('/', '_')}_graph.html")
            visualization.create_visualization_html(graph_data, viz_path)
            
            try:
                file_url = viz_path.resolve().as_uri()
                webbrowser.open(file_url)
                logger.info(f"Opening visualization in browser: {file_url}")
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                logger.info(f"Please open this file manually: {viz_path.resolve()}")
            
    except (ArxivExtractorError, FileNotFoundError, ValueError) as e:
        logger.error(f"A processing error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    """Defines and handles the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract mathematical dependency graphs from arXiv papers.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Fast regex-only extraction
  python -m arxiv_graph_extractor.main 2103.14030 --visualize

  # Powerful Hybrid (Regex+LLM) extraction (requires OPENAI_API_KEY)
  python -m arxiv_graph_extractor.main 2305.15334 --use-llm --visualize -p

  # Output to JSON
  python -m arxiv_graph_extractor.main 2211.11689 --use-llm -o graph.json --pretty
"""
    )
    parser.add_argument(
        "arxiv_id", help="arXiv identifier (e.g., '2103.14030', 'math.AG/0601001')"
    )
    parser.add_argument(
        "--use-llm", action="store_true", help="Use the Hybrid (Regex+LLM) extractor. Requires OPENAI_API_KEY."
    )
    parser.add_argument(
        "-o", "--output", nargs="?", const=True, type=str, help="Output JSON file path (default: print to stdout)."
    )
    parser.add_argument(
        "-viz", "--visualize", action="store_true", help="Create and open an interactive HTML visualization."
    )
    parser.add_argument(
        "--viz-output", type=Path, help="Custom path for visualization HTML file (default: arxiv_ID_graph.html)."
    )
    parser.add_argument(
        "-p", "--pretty", action="store_true", help="Pretty-print the JSON output."
    )
    
    args = parser.parse_args()
    asyncio.run(run_async_pipeline(args))

if __name__ == "__main__":
    main()