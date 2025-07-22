import os
import json
from threading import Lock
from loguru import logger

class DiscoveryIndex:
    """
    Manages a persistent, thread-safe, on-disk list of discovered arXiv IDs
    that are pending processing.
    """
    def __init__(self, output_dir: str):
        self.index_file_path = os.path.join(output_dir, "discovered_papers.json")
        self._lock = Lock()
        self.discovered_ids = self._load()
        logger.info(f"Discovery index initialized. Loaded {len(self.discovered_ids)} pending IDs from '{self.index_file_path}'.")

    def _load(self) -> set:
        """Loads the set of IDs from the JSON file."""
        with self._lock:
            if not os.path.exists(self.index_file_path):
                return set()
            try:
                with open(self.index_file_path, 'r', encoding='utf-8') as f:
                    # Storing as a list in JSON, but using a set in memory for efficiency
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not load or parse discovery index '{self.index_file_path}', starting fresh: {e}")
                return set()

    def _save(self):
        """Saves the current set of IDs to disk. Assumes lock is already held."""
        try:
            with open(self.index_file_path, 'w', encoding='utf-8') as f:
                # Convert set to a sorted list for clean, deterministic JSON output
                json.dump(sorted(list(self.discovered_ids)), f, indent=2)
        except IOError as e:
            logger.error(f"CRITICAL: Could not save discovery index to '{self.index_file_path}': {e}")

    def add_ids(self, new_ids: list[str]) -> int:
        """
        Adds new, unique IDs to the index and returns the count of newly added IDs.
        Saves to disk if changes were made.
        """
        if not new_ids:
            return 0
        
        with self._lock:
            initial_count = len(self.discovered_ids)
            self.discovered_ids.update(new_ids)
            newly_added_count = len(self.discovered_ids) - initial_count
            if newly_added_count > 0:
                self._save()
        
        return newly_added_count

    def get_pending_ids(self, limit: int = None) -> list[str]:
        """Returns a list of IDs pending processing, sorted for deterministic order."""
        with self._lock:
            pending = sorted(list(self.discovered_ids))
            return pending[:limit] if limit is not None else pending

    def remove_id(self, arxiv_id: str):
        """
        Removes a single ID from the index, typically after it has been processed
        (either successfully or with a terminal failure).
        """
        with self._lock:
            if arxiv_id in self.discovered_ids:
                self.discovered_ids.remove(arxiv_id)
                self._save()