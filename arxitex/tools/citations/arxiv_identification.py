#!/usr/bin/env python3
"""Stage 1: build list of works that cite a target work via OpenAlex.

Optional fallback:
- If --fallback-arxiv is enabled and a citing work has no arXiv signal,
  the script uses the shared arXiv matcher utility to query the arXiv API
  by title/author and mark the work as arXiv-available when matched.

Outputs:
- data/{target}_target_ids.json
- data/{target}_works.jsonl
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.arxiv_utils import parse_arxiv_id
from arxitex.tools.citations.openalex_citations import (
    OpenAlexCitingWorksStage,
    normalize_openalex_work_id,
)
from arxitex.tools.citations.target_resolution import OpenAlexTargetResolver
from arxitex.tools.citations.utils import ensure_dir


def _load_target_ids_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [normalize_openalex_work_id(s) for s in data if isinstance(s, str)]


def _resolve_target_ids_from_arxiv(
    *,
    target_arxiv: str,
    target_title: Optional[str],
    target_authors: Optional[List[str]],
    cache_dir: str,
    mailto: Optional[str],
    api_key: Optional[str],
) -> List[str]:
    resolver = OpenAlexTargetResolver(
        cache_dir=cache_dir, mailto=mailto, api_key=api_key
    )
    arxiv_id = parse_arxiv_id(target_arxiv)
    title = target_title
    authors = target_authors
    if not title or authors is None:
        meta = resolver.fetch_arxiv_metadata(arxiv_id, ArxivAPI())
        if not title:
            title = meta.title
        if authors is None:
            authors = meta.authors

    resolved = resolver.resolve_openalex_work_id(
        title=title or "", authors=authors or []
    )
    if not resolved:
        raise RuntimeError("Unable to resolve OpenAlex work id from arXiv metadata.")
    return [normalize_openalex_work_id(resolved)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build list of OpenAlex works citing a target work."
    )
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument(
        "--target-id", default=None, help="Output file prefix (target id)."
    )
    parser.add_argument(
        "--target-arxiv",
        "--target-arxiv-url",
        dest="target_arxiv",
        default=None,
        help="Target arXiv id or URL (used to resolve OpenAlex work id).",
    )
    parser.add_argument(
        "--target-title",
        default=None,
        help="Override title used for OpenAlex resolution.",
    )
    parser.add_argument(
        "--target-authors",
        nargs="*",
        default=None,
        help="Override author list used for OpenAlex resolution.",
    )
    parser.add_argument(
        "--cache-dir", default="data/cache", help="Cache directory for API responses."
    )
    parser.add_argument(
        "--mailto", default=None, help="Contact email for OpenAlex polite usage."
    )
    parser.add_argument("--api-key", default=None, help="OpenAlex API key (optional).")
    parser.add_argument(
        "--target-ids",
        nargs="*",
        default=None,
        help="OpenAlex Work IDs to use as citation targets.",
    )
    parser.add_argument(
        "--target-ids-file",
        default=None,
        help="Path to a JSON file containing a list of OpenAlex Work IDs.",
    )
    parser.add_argument(
        "--seed-ids",
        nargs="*",
        default=None,
        help="Deprecated. Use --target-ids instead.",
    )
    parser.add_argument(
        "--seed-ids-file",
        default=None,
        help="Deprecated. Use --target-ids-file instead.",
    )
    parser.add_argument(
        "--per-page", type=int, default=200, help="OpenAlex per-page size (max 200)."
    )
    parser.add_argument(
        "--max-works",
        type=int,
        default=0,
        help="Stop after N citing works (0 = no limit).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds to sleep between OpenAlex requests.",
    )
    parser.add_argument(
        "--fallback-arxiv",
        action="store_true",
        help="Try to find an arXiv version via the arXiv matcher when not marked arXiv.",
    )
    parser.add_argument(
        "--fallback-cache-db",
        default=None,
        help="SQLite cache DB for arXiv fallback (default: {cache_dir}/arxiv_fallback_cache.db).",
    )
    parser.add_argument(
        "--fallback-refresh-days",
        type=int,
        default=30,
        help="Refresh cached arXiv matches older than N days.",
    )
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)
    if args.fallback_cache_db is None:
        args.fallback_cache_db = os.path.join(args.cache_dir, "arxiv_fallback_cache.db")

    if args.seed_ids or args.seed_ids_file:
        logger.warning(
            "Warning: --seed-ids/--seed-ids-file are deprecated. Use --target-ids/--target-ids-file.",
        )

    target_ids: List[str] = []
    target_id = args.target_id

    ids_file = args.target_ids_file or args.seed_ids_file
    if ids_file:
        target_ids = _load_target_ids_file(ids_file)
    elif args.target_ids or args.seed_ids:
        raw_ids = args.target_ids or args.seed_ids or []
        target_ids = [normalize_openalex_work_id(s) for s in raw_ids]
    elif args.target_arxiv:
        if not target_id:
            resolver = OpenAlexTargetResolver(cache_dir=args.cache_dir)
            target_id = resolver.derive_target_id(parse_arxiv_id(args.target_arxiv))
        target_ids = _resolve_target_ids_from_arxiv(
            target_arxiv=args.target_arxiv,
            target_title=args.target_title,
            target_authors=args.target_authors,
            cache_dir=args.cache_dir,
            mailto=args.mailto,
            api_key=args.api_key,
        )
    else:
        logger.error(
            "No target IDs provided. Pass --target-ids/--target-ids-file or --target-arxiv.",
        )
        return 2

    if not target_id:
        target_id = "target"

    stage = OpenAlexCitingWorksStage(
        target_ids=target_ids,
        target_id=target_id,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        mailto=args.mailto,
        api_key=args.api_key,
        per_page=args.per_page,
        max_works=args.max_works,
        rate_limit=args.rate_limit,
        fallback_arxiv=args.fallback_arxiv,
        fallback_cache_db=args.fallback_cache_db,
        fallback_refresh_days=args.fallback_refresh_days,
    )
    logger.info(
        "Stage 1: target_id={} targets={} cache_dir={}",
        target_id,
        len(target_ids),
        args.cache_dir,
    )
    return stage.run()


if __name__ == "__main__":
    raise SystemExit(main())
