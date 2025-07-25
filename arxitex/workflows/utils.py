import json
from pathlib import Path
from typing import Dict, List, Any

from arxitex.extractor.utils import ArtifactNode

def save_graph_data(arxiv_id: str, graphs_output_dir: str, graph_data: dict) -> Path:
        """Saves the generated graph data to a persistent JSON file."""
        safe_paper_id = arxiv_id.replace('/', '_')
        graph_filename = f"{safe_paper_id}.json"
        graph_filepath = Path(graphs_output_dir) / graph_filename
        
        with open(graph_filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        return graph_filepath

def transform_graph_to_search_format(
    arxiv_id: str, 
    graph_nodes: List[ArtifactNode],
    artifact_to_terms_map: Dict[str, List[str]],
    paper_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Transforms the raw artifacts from a graph into a flat list of dictionaries
    optimized for search indexing.
    
    Args:
        arxiv_id: The ID of the source paper.
        graph_nodes: A list of ArtifactNode objects from the DocumentGraph.
        artifact_to_terms_map: A map from artifact ID to its extracted terms.

    Returns:
        A list of search-ready artifact dictionaries.
    """
    searchable_artifacts = []
    
    for node in graph_nodes:
        if node.is_external:
            continue

        # The prerequisite block is identified by a consistent header.
        content_full = node.content
        prereq_header = "--- Prerequisite Definitions ---"
        if prereq_header in node.content:
            content_full = node.content.replace(prereq_header, "").strip()
       
        paper_title = paper_metadata.get("title")
        paper_authors = paper_metadata.get("authors", [])
        abstract = paper_metadata.get("abstract", "")
        
        search_doc = {
            "title": paper_title,  
            "authors": paper_authors,  
            "arxiv_id": arxiv_id,
            "abstract": abstract,
            "artifact_id": node.id,
            "artifact_type": node.type.value,
            "content":  content_full,
            "terms": artifact_to_terms_map.get(node.id, []),
    
        }
        searchable_artifacts.append(search_doc)

    return searchable_artifacts