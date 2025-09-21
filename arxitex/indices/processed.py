from datetime import datetime, timezone
from typing import Dict, Optional

from arxitex.indices.base_sqlite import BaseSQLiteIndex


class ProcessedIndex(BaseSQLiteIndex):
    """Manages the index of processed papers and their status, backed by SQLite."""

    def _create_table(self):
        """
        Creates the processed_papers table with a corrected and clean schema.
        """
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS processed_papers (
                arxiv_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                processed_timestamp_utc TEXT NOT NULL,
                output_path TEXT,
                details TEXT
            )
        """
        with self._get_connection() as conn:
            conn.execute(create_table_sql)
            conn.commit()

    def update_processed_papers_status(self, arxiv_id: str, **kwargs):
        """Updates the status for a paper in a single, atomic transaction."""
        status = kwargs.pop("status", "success")
        output_path = kwargs.pop("output_path", None)
        timestamp = datetime.now(timezone.utc).isoformat()

        details_json = self._serialize(kwargs)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_papers (arxiv_id, status, processed_timestamp_utc, output_path, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (arxiv_id, status, timestamp, output_path, details_json),
            )
            conn.commit()

    def is_successfully_processed(self, arxiv_id: str) -> bool:
        """Checks the status of a paper with a fast, indexed lookup."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM processed_papers WHERE arxiv_id = ?", (arxiv_id,)
            )
            result = cursor.fetchone()
            return result["status"].startswith("success") if result else False

    def get_paper_status(self, arxiv_id: str) -> Optional[Dict]:
        """Retrieves all data for a processed paper."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM processed_papers WHERE arxiv_id = ?", (arxiv_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            status_dict = dict(row)
            details = self._deserialize(status_dict.pop("details"))
            status_dict.update(details)
            return status_dict
