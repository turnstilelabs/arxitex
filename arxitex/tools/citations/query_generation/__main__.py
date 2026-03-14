#!/usr/bin/env python3
"""Generate synthetic researcher queries from mention contexts (generic)."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Optional

from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.arxiv_utils import parse_arxiv_id
from arxitex.tools.citations.query_generation import QueryGenerator
from arxitex.tools.citations.target_resolution import OpenAlexTargetResolver
from arxitex.utils import ensure_dir, read_jsonl


class QueryGenerationStage:
    def __init__(
        self,
        *,
        target_id: str,
        target_name: str,
        mentions_file: str,
        out_dir: str,
        model: str,
        max_mentions: int,
        temperature: Optional[float],
        concurrency: int,
    ) -> None:
        self.target_id = target_id
        self.target_name = target_name
        self.mentions_file = mentions_file
        self.out_dir = out_dir
        self.model = model
        self.max_mentions = max_mentions
        self.temperature = temperature
        self.concurrency = concurrency

    async def run(self) -> int:
        ensure_dir(self.out_dir)
        ensure_dir("./prompts_cache")

        out_path = os.path.join(self.out_dir, f"{self.target_id}_queries.jsonl")
        if os.path.exists(out_path):
            os.remove(out_path)

        mentions = list(read_jsonl(self.mentions_file))
        if self.max_mentions:
            mentions = mentions[: self.max_mentions]

        logger.info("Loaded {} mentions from {}", len(mentions), self.mentions_file)
        logger.info("Writing queries to {}", out_path)

        generator = QueryGenerator(
            model=self.model,
            target_name=self.target_name,
            temperature=self.temperature,
            concurrency=self.concurrency,
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


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic researcher queries from mentions."
    )
    parser.add_argument(
        "--target-id", default=None, help="Output file prefix (target id)."
    )
    parser.add_argument(
        "--target-name", default=None, help="Human-friendly target name."
    )
    parser.add_argument(
        "--target-arxiv",
        "--target-arxiv-url",
        dest="target_arxiv",
        default=None,
        help="Target arXiv id or URL (to derive target name).",
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

    resolver = OpenAlexTargetResolver(cache_dir=args.out_dir)

    target_id = args.target_id
    target_name = args.target_name

    if not target_id and args.target_arxiv:
        target_id = resolver.derive_target_id(parse_arxiv_id(args.target_arxiv))
    if not target_id:
        target_id = "target"

    if not target_name and args.target_arxiv:
        meta = resolver.fetch_arxiv_metadata(
            parse_arxiv_id(args.target_arxiv), ArxivAPI()
        )
        target_name = meta.title
    if not target_name:
        target_name = "the target work"

    if not args.mentions_file:
        args.mentions_file = os.path.join(args.out_dir, f"{target_id}_mentions.jsonl")

    stage = QueryGenerationStage(
        target_id=target_id,
        target_name=target_name,
        mentions_file=args.mentions_file,
        out_dir=args.out_dir,
        model=args.model,
        max_mentions=args.max_mentions,
        temperature=args.temperature,
        concurrency=args.concurrency,
    )
    return await stage.run()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
