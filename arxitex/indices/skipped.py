from datetime import datetime, timezone

from arxitex.indices.base_sqlite import BaseSQLiteIndex


class SkippedIndex(BaseSQLiteIndex):
    """Manages the index of papers skipped by heuristics, backed by SQLite."""

    def _create_table(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skipped_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    skipped_at_utc TEXT NOT NULL
                )
            """)
            conn.commit()

    def add(self, arxiv_id: str, reason: str):
        """Adds a paper to the skipped index if it's not already there."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            # INSERT OR IGNORE ensures we don't overwrite an existing entry.
            conn.execute(
                "INSERT OR IGNORE INTO skipped_papers (arxiv_id, reason, skipped_at_utc) VALUES (?, ?, ?)",
                (arxiv_id, reason, timestamp)
            )
            conn.commit()
            
    def __contains__(self, arxiv_id: str) -> bool:
        """Checks if a paper is in the skipped index."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM skipped_papers WHERE arxiv_id = ?", (arxiv_id,))
            return cursor.fetchone() is not None