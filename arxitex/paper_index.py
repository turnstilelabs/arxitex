import os
import json
from threading import Lock
from datetime import datetime, timezone
from loguru import logger

class PaperIndex:
    """
    Manages a persistent, thread-safe, on-disk index of processed papers.
    """
    
    def __init__(self, output_dir):
        self.index_file_path = os.path.join(output_dir, "processed_papers.json")
        self._lock = Lock()
        self.processed_papers = self.load_processed_papers_index()

    def load_processed_papers_index(self) -> dict:
        """
        Loads the index from the JSON file.
        This method now returns a dictionary instead of a set and handles automatic migration.
        """
        with self._lock:
            if not os.path.exists(self.index_file_path):
                logger.info("No existing paper index found. Starting a new one.")
                return {}
            
            try:
                with open(self.index_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not load or parse index file '{self.index_file_path}', starting fresh: {e}")
                return {}

            if isinstance(data, list):
                logger.warning(f"Old index format (list) detected. Migrating to new dictionary format.")
                migrated_data = {
                    arxiv_id: {
                        "status": "migrated",
                        "processed_timestamp_utc": datetime.now(timezone.utc).isoformat()
                    } for arxiv_id in data
                }
                self._save(migrated_data)
                return migrated_data
            
            elif isinstance(data, dict):
                logger.info(f"Loaded existing paper index with {len(data)} entries from '{self.index_file_path}'.")
                return data

            else:
                logger.error(f"Unknown format in index file '{self.index_file_path}'. Starting fresh.")
                return {}

    def update_processed_papers_index(self, arxiv_id: str, **kwargs):
        """
        Marks a paper as processed with rich metadata and saves the index to disk.
        """
        with self._lock:
            if 'status' not in kwargs:
                kwargs['status'] = 'success'

            self.processed_papers[arxiv_id] = {
                "processed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
            self._save(self.processed_papers)
        
        logger.debug(f"Updated index for {arxiv_id} and saved to disk.")

    def is_paper_processed(self, arxiv_id: str) -> bool:
        with self._lock:
            return arxiv_id in self.processed_papers

    def _save(self, data_dict: dict):
        """Saves a dictionary to the index file. Assumes lock is already held."""
        try:
            with open(self.index_file_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2)
        except IOError as e:
            logger.error(f"CRITICAL: Could not save paper index to disk at '{self.index_file_path}': {e}")

