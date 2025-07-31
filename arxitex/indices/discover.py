from typing import Any, Dict, List

from arxitex.indices.base import BaseIndex

class DiscoveryIndex(BaseIndex):
    """
    Manages a persistent queue of discovered arXiv papers pending processing.
    """
    def __init__(self, output_dir: str):
        super().__init__(output_dir, "discovered_papers.json")
        self.papers = self.data

    def add_papers(self, new_papers: List[Dict[str, Any]]) -> int:
        """Adds new, unique papers to the index. Returns the count of newly added papers."""
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
        with self._lock:
            all_paper_ids = sorted(list(self.papers.keys()))
            
            if limit is not None:
                all_paper_ids = all_paper_ids[:limit]
                
            return [self.papers[pid] for pid in all_paper_ids]

    def remove_paper(self, arxiv_id: str):
        with self._lock:
            if arxiv_id in self.papers:
                del self.papers[arxiv_id]
                self._save()