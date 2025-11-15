#!/usr/bin/env python3
"""
CLI to generate an interactive HTML visualization from a saved graph JSON.

Usage examples:
  # Minimal: infer output path from arxiv_id or file stem and open in browser
  arxitex-viz data/graphs/2211.11689.json

  # Custom output and don't open browser
  arxitex-viz data/graphs/2211.11689.json --output data/viz/my_custom.html --no-open

You can also run as a module:
  python -m arxitex.extractor.visualization.cli data/graphs/2211.11689.json --open
"""
import argparse
import json
import webbrowser
from pathlib import Path
from typing import Dict

from loguru import logger

from . import graph_viz


def _default_viz_output(graph_json_path: Path, graph_data: Dict) -> Path:
    # Project root = repo root (arxitex/extractor/visualization/cli.py -> up 3)
    project_root = Path(__file__).resolve().parents[3]
    viz_dir = project_root / "data" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    arxiv_id = graph_data.get("arxiv_id")
    if arxiv_id:
        name = f"arxiv_{arxiv_id.replace('/', '_')}_graph.html"
    else:
        name = f"{graph_json_path.stem}_graph.html"
    return viz_dir / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML visualization from a graph JSON file."
    )
    parser.add_argument(
        "input",
        help="Path to the graph JSON file (e.g., data/graphs/2211.11689.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write the HTML visualization. Defaults to data/viz/arxiv_<id>_graph.html.",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the generated HTML file in the default web browser.",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open the generated HTML in the browser.",
    )
    parser.set_defaults(open_browser=True)

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        parser.error(f"Input JSON not found: {input_path}")

    try:
        graph_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        parser.error(f"Failed to read/parse JSON: {e}")

    # Resolve output path
    output_path = args.output or _default_viz_output(input_path, graph_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the HTML viz
    try:
        graph_viz.create_visualization_html(graph_data, output_path)
        logger.success(f"Visualization written to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        raise

    # Optionally open in browser
    if args.open_browser:
        try:
            webbrowser.open(output_path.resolve().as_uri())
            logger.info(f"Opening in browser: {output_path.resolve().as_uri()}")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")


if __name__ == "__main__":
    main()
