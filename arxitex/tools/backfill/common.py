"""Shared helpers for backfill workflows."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema


def load_existing_timestamps(
    db_path: str | Path,
    *,
    table: str,
    key_cols: Sequence[str],
    ts_col: str,
) -> dict:
    """Return {key -> timestamp} for the given table and columns."""

    ensure_schema(str(db_path))
    keys = ", ".join(key_cols)
    # Generic helper to cache timestamps for refresh decisions.
    sql = f"SELECT {keys}, {ts_col} FROM {table}"
    conn = connect(str(db_path))
    try:
        rows = conn.execute(sql).fetchall()
        out: dict = {}
        for r in rows:
            if not r:
                continue
            key_vals = [r[i] for i in range(len(key_cols))]
            key = key_vals[0] if len(key_vals) == 1 else tuple(key_vals)
            out[key] = r[len(key_cols)]
        return out
    finally:
        conn.close()


def should_refresh(ts: str | None, *, refresh_days: int, force: bool = False) -> bool:
    if force:
        return True
    if not ts:
        return True
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(refresh_days)))
    return dt < cutoff


def make_throttle(qps: float):
    """Return an awaitable throttle function enforcing a global QPS limit."""

    lock = asyncio.Lock()
    last_request_at = 0.0

    async def throttle() -> None:
        nonlocal last_request_at
        min_interval = 1.0 / max(0.05, float(qps))
        async with lock:
            now = asyncio.get_event_loop().time()
            wait = (last_request_at + min_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            last_request_at = asyncio.get_event_loop().time()

    return throttle


def iter_arxiv_ids_from_db(db_path: str | Path) -> list[str]:
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
