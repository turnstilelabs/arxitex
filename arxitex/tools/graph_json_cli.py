"""CLI wrapper that runs the full arxitex extraction pipeline and
prints a JSON graph to stdout.

This is designed to be called from the ArxiGraph Next.js API route via:

    python -m arxitex.tools.graph_json_cli 2211.11689 --infer-deps --enrich-content

Stdout contains a single JSON object with the following shape:

{
  "graph": { ... DocumentGraph.to_dict(...) ... },
  "definition_bank": { ... } | null,
  "artifact_to_terms_map": { "artifact_id": ["term1", ...], ... }
}

All logging and progress messages are emitted on stderr so they can be
forwarded to the client as status updates without corrupting the JSON
payload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict

from loguru import logger

from arxitex.extractor.models import ArxivExtractorError
from arxitex.extractor.pipeline import agenerate_artifact_graph


def _configure_logging(verbose: bool = False) -> None:
    """Configure loguru to log only to stderr.

    The Next.js layer treats stderr as status messages and stdout as the
    JSON payload.
    """

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


async def _run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the async graph extraction pipeline and shape the JSON payload."""

    # Derive final flags (support --all-enhancements like pipeline.py)
    infer_deps = bool(args.infer_deps)
    enrich_content = bool(args.enrich_content)

    if getattr(args, "all_enhancements", False):
        infer_deps = True
        enrich_content = True

    logger.info(
        f"Starting extraction for {args.arxiv_id} | infer_deps={infer_deps}, enrich_content={enrich_content}"
    )

    results = await agenerate_artifact_graph(
        arxiv_id=args.arxiv_id,
        infer_dependencies=infer_deps,
        enrich_content=enrich_content,
        source_dir=None,
    )

    graph = results.get("graph")
    bank = results.get("bank")
    artifact_to_terms_map = results.get("artifact_to_terms_map", {})

    if not graph or not graph.nodes:
        logger.warning("No artifacts were extracted; returning empty graph.")

    # Mirror the extractor_mode computation from extractor/pipeline.py
    if infer_deps and enrich_content:
        extractor_mode = "full-hybrid (deps + content)"
    elif infer_deps:
        extractor_mode = "hybrid (deps-only)"
    elif enrich_content:
        extractor_mode = "hybrid (content-only)"
    else:
        extractor_mode = "regex-only"

    graph_dict = graph.to_dict(arxiv_id=args.arxiv_id, extractor_mode=extractor_mode)

    bank_dict = None
    if bank is not None:
        try:
            bank_dict = await bank.to_dict()
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to serialize definition bank: {e}", exc_info=True)

    payload: Dict[str, Any] = {
        "graph": graph_dict,
        "definition_bank": bank_dict,
        "artifact_to_terms_map": artifact_to_terms_map,
    }

    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the arxitex extraction pipeline for a single arXiv ID and "
            "emit a JSON graph on stdout."
        )
    )
    parser.add_argument(
        "arxiv_id",
        help="arXiv identifier (e.g. '2103.14030', 'math.AG/0601001')",
    )
    parser.add_argument(
        "--infer-deps",
        action="store_true",
        help=(
            "Enable dependency inference between artifacts (LLM-based). "
            "Automatically activates content enrichment for best results in the "
            "underlying pipeline."
        ),
    )
    parser.add_argument(
        "--enrich-content",
        action="store_true",
        help="Enable content enrichment via definition/symbol extraction.",
    )
    parser.add_argument(
        "--all-enhancements",
        action="store_true",
        help="Convenience flag to enable both --infer-deps and --enrich-content.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose)

    # If enhancements are requested, ensure an LLM key is configured, similar
    # to extractor/pipeline.py. We don't enforce which backend; that logic
    # lives in arxitex.llms.
    if (args.infer_deps or args.enrich_content or args.all_enhancements) and not (
        os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    ):
        logger.error(
            "Enhancements requested but no LLM API key detected in the environment. "
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )
        sys.exit(2)

    try:
        payload = asyncio.run(_run_pipeline(args))
        json.dump(payload, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except (ArxivExtractorError, FileNotFoundError, ValueError) as e:
        logger.error(f"A processing error occurred: {e}")
        sys.exit(1)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
