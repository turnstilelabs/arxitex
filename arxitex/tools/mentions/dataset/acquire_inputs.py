#!/usr/bin/env python3
"""Acquire network-dependent inputs for the mentions dataset pipeline.

This stage performs:
- target metadata enrichment (arXiv/OpenAlex)
- statement extraction for targets (if missing)
- mention extraction from citing papers (if missing)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arxitex.tools.mentions.acquisition.target_resolution import OpenAlexTargetResolver
from arxitex.tools.mentions.dataset.acquisition_pipeline import (
    ensure_statements_for_targets,
    ensure_target_metadata,
    extract_mentions_for_targets,
    load_targets,
)
from arxitex.utils import ensure_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire network-dependent mention pipeline inputs."
    )
    parser.add_argument("--targets-json", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--out-dir", default="data/mentions")
    parser.add_argument("--statements-dir", default="data/statements/mentions")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--max-works", type=int, default=0)
    parser.add_argument("--rate-limit", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fallback-arxiv", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    statements_dir = Path(args.statements_dir)
    cache_dir = Path(args.cache_dir)
    ensure_dir(str(out_dir))
    ensure_dir(str(cache_dir))
    ensure_dir(str(statements_dir))

    resolver = OpenAlexTargetResolver(
        cache_dir=str(cache_dir),
        mailto=args.mailto,
        api_key=args.api_key,
    )
    targets = load_targets(args.targets_json, args.targets)
    ensure_target_metadata(targets=targets, resolver=resolver)
    ensure_statements_for_targets(targets=targets, statements_dir=statements_dir)
    extract_mentions_for_targets(
        targets=targets,
        resolver=resolver,
        out_dir=out_dir,
        cache_dir=cache_dir,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
