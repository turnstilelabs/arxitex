import os
import json
from typing import Dict, List, Any
from threading import Lock
from loguru import logger

class DiscoveryIndex:
    """
    Manages a persistent, on-disk dictionary of discovered arXiv papers
    and their associated metadata, keyed by arxiv_id.
    """
    def __init__(self, output_dir: str):
        self.index_file_path = os.path.join(output_dir, "discovered_papers.json")
        self._lock = Lock()
        self.papers: Dict[str, Dict[str, Any]] = self._load()
        logger.info(f"Discovery index initialized. Loaded metadata for {len(self.papers)} papers from '{self.index_file_path}'.")

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Loads the dictionary of paper metadata from the JSON file."""
        with self._lock:
            if not os.path.exists(self.index_file_path):
                return {}
            try:
                with open(self.index_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not load or parse discovery index '{self.index_file_path}', starting fresh: {e}")
                return {}

    def _save(self):
        """Saves the current dictionary of papers to disk. Assumes lock is already held."""
        try:
            # Sort keys for deterministic output, making file diffs meaningful
            sorted_papers = {k: self.papers[k] for k in sorted(self.papers.keys())}
            with open(self.index_file_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_papers, f, indent=2)
        except IOError as e:
            logger.error(f"CRITICAL: Could not save discovery index to '{self.index_file_path}': {e}")

    def add_papers(self, new_papers: List[Dict[str, Any]]) -> int:
        """
        Adds new, unique papers to the index.
        Returns the count of newly added papers.
        """
        if not new_papers:
            return 0
        
        newly_added_count = 0
        with self._lock:
            for paper in new_papers:
                arxiv_id = paper.get('arxiv_id')
                if arxiv_id and arxiv_id not in self.papers:
                    self.papers[arxiv_id] = paper
                    newly_added_count += 1
            
            if newly_added_count > 0:
                self._save()
        
        return newly_added_count

    def get_pending_papers(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Returns a list of paper metadata dicts that are pending processing.
        """    
        with self._lock:
            all_paper_ids = sorted(list(self.papers.keys()))
            
            if limit is not None:
                all_paper_ids = all_paper_ids[:limit]
                
            return [self.papers[pid] for pid in all_paper_ids]

    def remove_paper(self, arxiv_id: str):
        """
        Removes a single paper from the discovery index by its ID.
        """
        with self._lock:
            if arxiv_id in self.papers:
                del self.papers[arxiv_id]
                self._save()