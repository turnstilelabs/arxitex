from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: str | Path, *, timeout_s: float = 30) -> sqlite3.Connection:
    """Create a SQLite connection with production-friendly PRAGMAs.

    Notes
    -----
    - WAL significantly improves concurrency for multi-worker processing.
    - foreign_keys must be enabled per-connection in SQLite.
    """

    conn = sqlite3.connect(str(db_path), timeout=timeout_s)
    conn.row_factory = sqlite3.Row

    # PRAGMAs are connection-local except journal_mode.
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")

    # Best-effort: journal_mode returns a row; ignore failures.
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass

    # NORMAL is a reasonable tradeoff for WAL.
    conn.execute("PRAGMA synchronous = NORMAL;")

    return conn
