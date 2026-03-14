#!/usr/bin/env python3
"""Expand targets.json by downloading arXiv sources referenced in *_works.jsonl files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set

from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arxitex.tools.mentions.dataset.sources.source_utils import (  # noqa: E402
    canonicalize_arxiv_id,
    download_source_to_dir,
    iter_arxiv_ids_from_works,
    load_targets,
    save_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand targets using cached OpenAlex works files."
    )
    parser.add_argument("--works-dir", default="data/mentions")
    parser.add_argument("--out-dir", default="data/sources")
    parser.add_argument(
        "--targets-json",
        default="data/mentions/targets.json",
        help="Targets JSON to update.",
    )
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    works_dir = Path(args.works_dir)
    works_paths = list(works_dir.glob("*_works.jsonl"))
    if not works_paths:
        logger.error("No *_works.jsonl files found under {}", works_dir)
        return 1

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

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "_cache"
    new_count = 0
    for arxiv_id in iter_arxiv_ids_from_works(works_paths):
        if new_count >= args.max_new:
            break
        if arxiv_id in existing_ids:
            continue
        dest_dir = out_dir / arxiv_id.replace("/", "_")
        if dest_dir.exists() and any(dest_dir.iterdir()):
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
