"""Rollback processed papers after a cutoff timestamp.

This is a maintenance utility for the pipeline SQLite DB (usually
`pipeline_output/arxitex_indices.db`).

It will:
  1) Select all rows from `processed_papers` with processed_timestamp_utc > cutoff.
  2) (Optionally) delete them from `processed_papers`.
  3) (Optionally) re-add those arXiv IDs to the discovery queue (`discovered_papers`).

Notes
-----
- `processed_papers.arxiv_id` is PRIMARY KEY, so it is unique by definition.
- The discovery queue is `discovered_papers` (legacy name, but it's the queue).
- `discovered_papers` typically stores `metadata` as JSON string; if a paper is
  re-queued without metadata, processing may still work but ordering/filtering
  features may behave worse. This script therefore tries to recover metadata
  from the normalized `papers` table if available.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Plan:
    db_path: Path
    cutoff_iso: str
    apply: bool
    requeue: bool


def _find_default_db() -> Path | None:
    """Best-effort locator for the default pipeline DB."""
    root = Path("pipeline_output")
    if not root.exists():
        return None
    matches = sorted(root.rglob("arxitex_indices.db"))
    return matches[0] if matches else None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def _select_affected(conn: sqlite3.Connection, cutoff_iso: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row

    cols = _get_columns(conn, "processed_papers")
    if "processed_timestamp_utc" not in cols:
        raise RuntimeError(
            "processed_papers.processed_timestamp_utc column not found. "
            f"Columns are: {cols}"
        )

    return conn.execute(
        """
        SELECT arxiv_id, status, processed_timestamp_utc
        FROM processed_papers
        WHERE processed_timestamp_utc > ?
        ORDER BY processed_timestamp_utc ASC
        """,
        (cutoff_iso,),
    ).fetchall()


def _metadata_for_requeue(conn: sqlite3.Connection, arxiv_id: str) -> str:
    """Try to reconstruct discovery-queue-style metadata JSON.

    Preference order:
    1) normalized `papers` table (if it exists)
    2) minimal stub with just arxiv_id
    """
    if _table_exists(conn, "papers"):
        cols = _get_columns(conn, "papers")
        # Expect at least paper_id; may also include title/authors/categories.
        if "paper_id" in cols:
            row = conn.execute(
                "SELECT * FROM papers WHERE paper_id = ?", (arxiv_id,)
            ).fetchone()
            if row:
                # Keep it conservative: include whatever is there but ensure arxiv_id key.
                d = dict(row)
                d.setdefault("arxiv_id", arxiv_id)
                return json.dumps(d)

    return json.dumps({"arxiv_id": arxiv_id})


def _ensure_discovered_papers_row(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    """Insert into discovered_papers if absent.

    Returns True if inserted, False if already present.
    """
    cols = _get_columns(conn, "discovered_papers")
    if "arxiv_id" not in cols:
        raise RuntimeError(
            "discovered_papers.arxiv_id column not found. " f"Columns are: {cols}"
        )
    if "metadata" not in cols:
        raise RuntimeError(
            "discovered_papers.metadata column not found. " f"Columns are: {cols}"
        )

    exists = conn.execute(
        "SELECT 1 FROM discovered_papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    if exists:
        return False

    meta = _metadata_for_requeue(conn, arxiv_id)
    conn.execute(
        "INSERT INTO discovered_papers (arxiv_id, metadata) VALUES (?, ?)",
        (arxiv_id, meta),
    )
    return True


def _delete_processed(conn: sqlite3.Connection, arxiv_ids: Iterable[str]) -> int:
    arxiv_ids = list(arxiv_ids)
    if not arxiv_ids:
        return 0
    conn.executemany(
        "DELETE FROM processed_papers WHERE arxiv_id = ?",
        [(aid,) for aid in arxiv_ids],
    )
    return len(arxiv_ids)


def run(plan: Plan) -> int:
    if not plan.db_path.exists():
        raise SystemExit(f"DB not found: {plan.db_path}")

    conn = sqlite3.connect(str(plan.db_path))
    try:
        affected = _select_affected(conn, plan.cutoff_iso)
        arxiv_ids = [r["arxiv_id"] for r in affected]

        print(f"DB: {plan.db_path}")
        print(f"Cutoff: {plan.cutoff_iso}")
        print(f"Affected processed_papers rows: {len(arxiv_ids)}")
        if affected[:10]:
            print("Sample (first 10):")
            for r in affected[:10]:
                print(
                    f"  {r['processed_timestamp_utc']}  {r['arxiv_id']}  {r['status']}"
                )

        if not plan.apply:
            print("\nDry-run only. Re-run with --apply to modify the DB.")
            return 0

        # Apply changes transactionally.
        inserted = 0
        deleted = 0
        try:
            conn.execute("BEGIN")

            if plan.requeue:
                # Ensure discovered_papers exists
                if not _table_exists(conn, "discovered_papers"):
                    raise RuntimeError(
                        "discovered_papers table not found; cannot requeue."
                    )
                for aid in arxiv_ids:
                    if _ensure_discovered_papers_row(conn, aid):
                        inserted += 1

            deleted = _delete_processed(conn, arxiv_ids)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        print("\nApplied successfully:")
        print(f"  requeued into discovered_papers: {inserted}")
        print(f"  deleted from processed_papers : {deleted}")
        return 0
    finally:
        conn.close()


def main() -> int:
    default_db = _find_default_db()

    p = argparse.ArgumentParser(
        description=(
            "Rollback rows in processed_papers after a cutoff timestamp and optionally requeue them."
        )
    )
    p.add_argument(
        "--db",
        type=str,
        default=str(default_db) if default_db else None,
        help=(
            "Path to arxitex_indices.db (default: first match under pipeline_output/)"
        ),
    )
    p.add_argument(
        "--cutoff",
        required=True,
        help=(
            "ISO timestamp cutoff. Rows with processed_timestamp_utc > cutoff will be rolled back. "
            "Example: 2026-01-06T21:09:16.582679+00:00"
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify the DB (default is dry-run).",
    )
    p.add_argument(
        "--no-requeue",
        action="store_true",
        help="Do not re-add papers to discovered_papers; only delete from processed_papers.",
    )

    args = p.parse_args()
    if not args.db:
        raise SystemExit(
            "Could not infer --db (no pipeline_output/**/arxitex_indices.db found). "
            "Provide --db explicitly."
        )

    plan = Plan(
        db_path=Path(args.db),
        cutoff_iso=args.cutoff,
        apply=bool(args.apply),
        requeue=(not bool(args.no_requeue)),
    )
    return run(plan)


if __name__ == "__main__":
    raise SystemExit(main())
