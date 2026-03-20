#!/usr/bin/env python3
"""Fetch arXiv sources for many targets via the export API.

This:
1) Queries the arXiv API for IDs (by category or raw query).
2) Downloads source archives from export.arxiv.org/e-print/<id>.
3) Extracts into data/sources/<id>/.
4) Appends new targets to data/mentions/targets.json.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Set

from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arxitex.arxiv_utils import parse_arxiv_id  # noqa: E402
from arxitex.tools.mentions.dataset.sources.source_utils import (  # noqa: E402
    canonicalize_arxiv_id,
    download_source_to_dir,
    load_targets,
    save_targets,
)

ARXIV_API = "http://export.arxiv.org/api/query"


def _fetch_ids(
    query: str, start: int, max_results: int, *, max_retries: int = 5
) -> List[str]:
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    logger.info("Fetching arXiv API: {}", url)
    data = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as f:
                data = f.read()
            break
        except Exception as exc:
            if getattr(exc, "code", None) == 429:
                wait = 5 * (2**attempt)
                if attempt + 1 < max_retries:
                    logger.warning(
                        "arXiv API rate-limited (429). Retrying in {}s...", wait
                    )
                    time.sleep(wait)
                    continue
                logger.warning("arXiv API still rate-limited after retries.")
                return []
            raise
    if data is None:
        return []
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    ids: List[str] = []
    for entry in root.findall("atom:entry", ns):
        id_node = entry.find("atom:id", ns)
        if id_node is None or not id_node.text:
            continue
        try:
            ids.append(parse_arxiv_id(id_node.text))
        except ValueError:
            continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch arXiv sources in bulk.")
    parser.add_argument(
        "--categories",
        default="math.AP",
        help="Comma-separated arXiv categories (e.g., math.AP,math.DG).",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Raw arXiv API query (overrides categories if set).",
    )
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--out-dir", default="data/sources", help="Directory for extracted sources."
    )
    parser.add_argument(
        "--targets-json",
        default="data/mentions/targets.json",
        help="Targets JSON to update.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=50,
        help="Maximum number of new sources to download.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep seconds between downloads.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    targets_path = Path(args.targets_json)
    targets = load_targets(targets_path)
    existing_ids: Set[str] = set()
    for target in targets:
        raw_id = target.get("arxiv_id")
        if not raw_id:
            continue
        parsed = canonicalize_arxiv_id(raw_id)
        if parsed:
            existing_ids.add(parsed)

    if args.query.strip():
        queries = [args.query.strip()]
    else:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        if not cats:
            logger.error("No categories specified.")
            return 1
        queries = [f"cat:{c}" for c in cats]

    fetched: List[str] = []
    for q in queries:
        fetched.extend(_fetch_ids(q, args.start, args.max_results))

    deduped: List[str] = []
    seen: Set[str] = set()
    for arxiv_id in fetched:
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        deduped.append(arxiv_id)

    cache_dir = out_dir / "_cache"
    new_count = 0
    for arxiv_id in deduped:
        if new_count >= args.max_new:
            break
        if arxiv_id in existing_ids:
            continue
        dest_dir = out_dir / arxiv_id.replace("/", "_")
        if dest_dir.exists() and any(dest_dir.iterdir()):
            logger.info("Source already exists for {}", arxiv_id)
            existing_ids.add(arxiv_id)
            targets.append({"arxiv_id": arxiv_id, "local_source_dir": str(dest_dir)})
            new_count += 1
            continue
        ok = download_source_to_dir(
            arxiv_id, dest_dir, cache_dir=cache_dir, sleep_sec=args.sleep
        )
        if not ok:
            continue
        existing_ids.add(arxiv_id)
        targets.append({"arxiv_id": arxiv_id, "local_source_dir": str(dest_dir)})
        new_count += 1

    if new_count:
        save_targets(targets_path, targets)
        logger.info("Added {} new targets to {}", new_count, targets_path)
    else:
        logger.info("No new targets added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
