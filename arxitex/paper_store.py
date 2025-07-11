
import os
import json

class PaperStore:
    """Handles storing and retrieving of paper data"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_paper_results(self, paper, paper_index):
        """Save the results for a single paper"""
        arxiv_id = paper['arxiv_id']
        output_file = os.path.join(self.output_dir, f"{arxiv_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(paper, f, indent=2, ensure_ascii=False)
        paper_index.update_processed_papers_index(arxiv_id)
