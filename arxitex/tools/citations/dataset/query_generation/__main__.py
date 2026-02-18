#!/usr/bin/env python3
"""Generate synthetic researcher queries from mention contexts (generic)."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Optional

from loguru import logger

from arxitex.tools.citations.dataset.query_generation import QueryGenerator
from arxitex.tools.citations.dataset.utils import ensure_dir, read_jsonl


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic researcher queries from mentions."
    )
    parser.add_argument(
        "--target-id", default="target", help="Output file prefix (target id)."
    )
    parser.add_argument(
        "--target-name", default="the target work", help="Human-friendly target name."
    )
    parser.add_argument("--mentions-file", default=None, help="Input mentions JSONL.")
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument(
        "--model",
        default="gpt-5-mini-2025-08-07",
        help="LLM model name (OpenAI).",
    )
    parser.add_argument(
        "--max-mentions", type=int, default=0, help="Limit mentions processed."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omit for models that don't support it).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Max concurrent LLM calls."
    )
    args = parser.parse_args(argv)

    ensure_dir(args.out_dir)
    ensure_dir("./prompts_cache")

    target_id = args.target_id
    target_name = args.target_name

    if not args.mentions_file:
        args.mentions_file = os.path.join(args.out_dir, f"{target_id}_mentions.jsonl")

    out_path = os.path.join(args.out_dir, f"{target_id}_queries.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    mentions = list(read_jsonl(args.mentions_file))
    if args.max_mentions:
        mentions = mentions[: args.max_mentions]

    logger.info("Loaded {} mentions from {}", len(mentions), args.mentions_file)
    logger.info("Writing queries to {}", out_path)

    generator = QueryGenerator(
        model=args.model,
        target_name=target_name,
        temperature=args.temperature,
        concurrency=args.concurrency,
    )
    counters = await generator.generate_from_mentions(mentions, out_path)

    logger.info(
        "Done. Processed {} mentions (failed {}). Wrote {} queries to {}",
        counters["processed"],
        counters["failed"],
        counters["queries"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
