"""Utilities for inspecting and deduplicating the discovery queue.

The discovery queue lives in the shared SQLite DB (`arxitex_indices.db`) in the
`discovered_papers` table.

This module provides a safe deduplication routine that:
- groups by base arXiv id (stripping a trailing `vN` version suffix)
- keeps the highest version per base id
- optionally creates a timestamped backup before applying changes
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_ARXIV_VERSION_RE = re.compile(r"^(?P<base>.+?)v(?P<ver>\d+)$")


def split_arxiv_id(arxiv_id: str) -> tuple[str, int]:
    """Return (base_id, version_number).

    Examples
    --------
    - "2406.01082v2" -> ("2406.01082", 2)
    - "2406.01082" -> ("2406.01082", 0)
    """

    m = _ARXIV_VERSION_RE.match(arxiv_id)
    if not m:
        return (arxiv_id, 0)
    return (m.group("base"), int(m.group("ver")))


@dataclass(frozen=True)
class DiscoveryQueueDedupReport:
    db_path: Path
    backup_path: Optional[Path]
    rows_before: int
    rows_to_delete: int
    rows_deleted: int
    base_ids_duplicated_before: int
    base_ids_duplicated_after: int
    deleted_arxiv_ids: Sequence[str]


def _fetch_all_arxiv_ids(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT arxiv_id FROM discovered_papers ORDER BY arxiv_id")
    return [row[0] for row in cur.fetchall()]


def _count_rows(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM discovered_papers").fetchone()[0])


def _count_base_id_dupes_from_ids(arxiv_ids: Iterable[str]) -> int:
    """Count base IDs that appear more than once.

    We do this in Python to ensure the same normalization semantics as
    :func:`split_arxiv_id`, rather than relying on SQL substring heuristics.
    """

    counts: dict[str, int] = {}
    for aid in arxiv_ids:
        if not aid:
            continue
        base, _ = split_arxiv_id(aid)
        counts[base] = counts.get(base, 0) + 1
    return sum(1 for c in counts.values() if c > 1)


def _compute_deletions(arxiv_ids: Iterable[str]) -> Tuple[int, List[str]]:
    """Compute which IDs to delete; keep only highest version per base id."""

    best: dict[str, tuple[int, str]] = {}
    all_ids: list[str] = []
    for aid in arxiv_ids:
        if not aid:
            continue
        all_ids.append(aid)
        base, ver = split_arxiv_id(aid)
        prev = best.get(base)
        if prev is None or ver > prev[0]:
            best[base] = (ver, aid)

    keep_ids = {aid for (_, aid) in best.values()}
    to_delete = [aid for aid in all_ids if aid not in keep_ids]
    return (len(all_ids), sorted(set(to_delete)))


def _backup_db(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(db_path.suffix + f".bak.{ts}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def dedup_discovery_queue(
    db_path: str | Path,
    *,
    dry_run: bool = True,
    make_backup: bool = True,
) -> DiscoveryQueueDedupReport:
    """Deduplicate `discovered_papers` by base arXiv id.

    Policy: keep only the highest `vN` for each base id.
    """

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        rows_before = _count_rows(conn)
        arxiv_ids = _fetch_all_arxiv_ids(conn)
        base_dupes_before = _count_base_id_dupes_from_ids(arxiv_ids)

        _, to_delete = _compute_deletions(arxiv_ids)

        backup_path: Optional[Path] = None
        rows_deleted = 0
        if not dry_run and to_delete:
            if make_backup:
                backup_path = _backup_db(db_path)

            placeholders = ",".join(["?"] * len(to_delete))
            conn.execute(
                f"DELETE FROM discovered_papers WHERE arxiv_id IN ({placeholders})",
                to_delete,
            )
            rows_deleted = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()

        remaining_arxiv_ids = _fetch_all_arxiv_ids(conn)
        base_dupes_after = _count_base_id_dupes_from_ids(remaining_arxiv_ids)
        return DiscoveryQueueDedupReport(
            db_path=db_path,
            backup_path=backup_path,
            rows_before=rows_before,
            rows_to_delete=len(to_delete),
            rows_deleted=int(rows_deleted),
            base_ids_duplicated_before=base_dupes_before,
            base_ids_duplicated_after=base_dupes_after,
            deleted_arxiv_ids=to_delete,
        )
    finally:
        conn.close()
