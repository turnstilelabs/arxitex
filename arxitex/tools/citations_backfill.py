from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

from loguru import logger

from arxitex.tools.citations_openalex import (
    backfill_citations_openalex,
    strip_arxiv_version,
)


def _iter_arxiv_ids_from_db(db_path: str | Path) -> list[str]:
    """Load arXiv IDs from discovery + processed tables.

    The pipeline DB contains two different schemas depending on usage:
    - Legacy indices: discovered_papers, processed_papers
    - Normalized schema: papers

    We primarily care about the discovery queue and already-processed papers.
    """

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        ids: list[str] = []

        # discovered_papers (legacy queue)
        try:
            rows = conn.execute("SELECT arxiv_id FROM discovered_papers").fetchall()
            ids.extend([r[0] for r in rows if r and r[0]])
        except sqlite3.OperationalError:
            pass

        # processed_papers (legacy)
        try:
            rows = conn.execute("SELECT arxiv_id FROM processed_papers").fetchall()
            ids.extend([r[0] for r in rows if r and r[0]])
        except sqlite3.OperationalError:
            pass

        # normalized papers table (if you use persistence)
        try:
            rows = conn.execute("SELECT paper_id FROM papers").fetchall()
            ids.extend([r[0] for r in rows if r and r[0]])
        except sqlite3.OperationalError:
            pass

        # stable unique
        seen = set()
        out = []
        for a in ids:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out
    finally:
        conn.close()


def _iter_arxiv_ids_from_discovery_only(db_path: str | Path) -> list[str]:
    """Load arXiv IDs from discovery queue only."""

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT arxiv_id FROM discovered_papers").fetchall()
        except sqlite3.OperationalError:
            return []
        ids = [r[0] for r in rows if r and r[0]]
        # stable unique
        seen = set()
        out = []
        for a in ids:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out
    finally:
        conn.close()


def _load_paper_metadata_map(db_path: str | Path) -> dict[str, dict]:
    """Build base_id -> metadata map from discovered_papers.

    Metadata is stored as JSON in discovered_papers.metadata.
    This provides title/authors for OpenAlex matching.
    """

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        out: dict[str, dict] = {}
        try:
            rows = conn.execute("SELECT metadata FROM discovered_papers").fetchall()
        except sqlite3.OperationalError:
            return out

        import json

        for r in rows:
            try:
                m = json.loads(r[0])
            except Exception:
                continue
            arxiv_id = m.get("arxiv_id")
            if not arxiv_id:
                continue
            base_id = strip_arxiv_version(arxiv_id)
            # Keep the first seen; that's fine.
            out.setdefault(base_id, m)
        return out
    finally:
        conn.close()


async def run_backfill(args) -> int:
    if getattr(args, "only_discovery", False):
        arxiv_ids = _iter_arxiv_ids_from_discovery_only(args.db_path)
    else:
        arxiv_ids = _iter_arxiv_ids_from_db(args.db_path)
    logger.info(f"Loaded {len(arxiv_ids)} arXiv ids from DB")

    meta_map = _load_paper_metadata_map(args.db_path)
    logger.info(f"Loaded metadata for {len(meta_map)} base arXiv ids")

    stats = await backfill_citations_openalex(
        db_path=args.db_path,
        arxiv_ids=arxiv_ids,
        paper_metadata_by_id=meta_map,
        workers=args.workers,
        refresh_days=args.refresh_days,
        mailto=args.mailto,
        qps=args.qps,
        max_papers=args.max_papers,
    )
    logger.info(f"OpenAlex backfill done: {stats}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill OpenAlex total citation counts for all arXiv ids in the pipeline DB",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(Path.cwd() / "pipeline_output" / "arxitex_indices.db"),
        help="Path to arxitex_indices.db",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of unique base arXiv ids to fetch (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent OpenAlex requests",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help=(
            "Global request rate limit (requests/second) across all workers. "
            "Use something conservative like 0.5–1.0 for long runs."
        ),
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=30,
        help="Refetch citations if older than this many days",
    )
    parser.add_argument(
        "--mailto",
        type=str,
        default=None,
        help="Optional mailto parameter recommended by OpenAlex (contact email)",
    )

    parser.add_argument(
        "--only-discovery",
        action="store_true",
        help="Only backfill citation counts for discovery-queue papers (ignore processed/papers tables)",
    )

    args = parser.parse_args()
    return asyncio.run(run_backfill(args))


if __name__ == "__main__":
    raise SystemExit(main())
