import os
import json
from threading import Lock
from typing import Dict, Any
from datetime import datetime, timezone
from loguru import logger


class SkippedIndex:
    """
    Manages a persistent, on-disk log of papers that were skipped by
    pre-processing heuristics, along with the reason for skipping.
    """
    def __init__(self, output_dir: str):
        self.index_file_path = os.path.join(output_dir, "skipped_papers.json")
        self._lock = Lock()
        self.skipped_papers: Dict[str, Dict[str, Any]] = self._load()
        logger.info(f"Skipped index initialized. Loaded {len(self.skipped_papers)} entries from '{self.index_file_path}'.")

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Loads the dictionary of skipped papers from the JSON file."""
        with self._lock:
            if not os.path.exists(self.index_file_path):
                return {}
            try:
                with open(self.index_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}

    def _save(self):
        """Saves the current dictionary of skipped papers to disk."""
        try:
            sorted_papers = {k: self.skipped_papers[k] for k in sorted(self.skipped_papers.keys())}
            with open(self.index_file_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_papers, f, indent=2)
        except IOError as e:
            logger.error(f"CRITICAL: Could not save skipped index: {e}")

    def add_skipped(self, arxiv_id: str, reason: str):
        """
        Adds a paper to the skipped index if it's not already there.
        """
        with self._lock:
            if arxiv_id not in self.skipped_papers:
                self.skipped_papers[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "reason": reason,
                    "skipped_at_utc": datetime.now(timezone.utc).isoformat()
                }
                self._save()
                logger.error(f"Added {arxiv_id} to skipped index. Reason: {reason}")