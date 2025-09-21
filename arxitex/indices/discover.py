from typing import Any, Dict, List

from arxitex.indices.base_sqlite import BaseSQLiteIndex


class DiscoveryIndex(BaseSQLiteIndex):
    """Manages the index of discovered papers pending processing, backed by SQLite."""

    def _create_table(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discovered_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def add_papers(self, new_papers: List[Dict[str, Any]]) -> int:
        """Adds new, unique papers to the index. Returns the count of newly added papers."""
        if not new_papers:
            return 0

        papers_to_insert = [
            (p["arxiv_id"], self._serialize(p)) for p in new_papers if p.get("arxiv_id")
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # INSERT OR IGNORE will silently skip any papers whose arxiv_id already exists.
            cursor.executemany(
                "INSERT OR IGNORE INTO discovered_papers (arxiv_id, metadata) VALUES (?, ?)",
                papers_to_insert,
            )
            newly_added_count = cursor.rowcount
            conn.commit()

        return newly_added_count

    def get_pending_papers(self) -> List[Dict[str, Any]]:
        """Returns a list of all paper metadata dicts that are pending processing."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT metadata FROM discovered_papers ORDER BY arxiv_id"
            )
            return [self._deserialize(row["metadata"]) for row in cursor.fetchall()]

    def remove_paper(self, arxiv_id: str):
        """Removes a single paper from the discovery index by its ID."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM discovered_papers WHERE arxiv_id = ?", (arxiv_id,)
            )
            conn.commit()
