from datetime import datetime, timezone
from loguru import logger

from arxitex.indices.base import BaseIndex

class SkippedIndex(BaseIndex):
    """
    Manages a persistent log of papers skipped by pre-processing heuristics.
    """
    def __init__(self, output_dir: str):
        super().__init__(output_dir, "skipped_papers.json")
        self.skipped_papers = self.data

    def add(self, arxiv_id: str, reason: str):
        """Adds a paper to the skipped index if it's not already there."""
        with self._lock:
            if arxiv_id not in self.skipped_papers:
                self.skipped_papers[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "reason": reason,
                    "skipped_at_utc": datetime.now(timezone.utc).isoformat()
                }
                self._save()
                logger.info(f"Added {arxiv_id} to skipped index. Reason: {reason}")