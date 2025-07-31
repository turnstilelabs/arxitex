from typing import Dict, Optional
from datetime import datetime, timezone
from loguru import logger

from arxitex.indices.base import BaseIndex

class ProcessedIndex(BaseIndex):
    """
    Manages a persistent, on-disk index of all attempted papers and their final status.
    """
    def __init__(self, output_dir: str):
        super().__init__(output_dir, "processed_papers.json")
        self.processed_papers = self.data

    def _get_default_data(self) -> Dict:
        return {}
        
    def update_processed_papers_status(self, arxiv_id: str, **kwargs):
        """Updates the status of a paper in the index and saves to disk."""
        with self._lock:
            if 'status' not in kwargs:
                kwargs['status'] = 'success'

            entry = self.processed_papers.get(arxiv_id, {})
            
            if kwargs['status'] == 'failure':
                entry['retry_count'] = entry.get('retry_count', 0) + 1

            entry.update({
                "processed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **kwargs
            })
            self.processed_papers[arxiv_id] = entry
            self._save()
        
        logger.debug(f"Updated index for {arxiv_id} (status: {kwargs['status']}) and saved to disk.")

    def get_paper_status(self, arxiv_id: str) -> Optional[Dict]:
        """Returns the full status dictionary for a paper, or None if not found."""
        with self._lock:
            return self.processed_papers.get(arxiv_id)

    def is_successfully_processed(self, arxiv_id: str) -> bool:
        with self._lock:
            entry = self.processed_papers.get(arxiv_id)
            if not entry:
                return False
            return entry.get('status', '').startswith('success')