import json
from pathlib import Path
from typing import Any, Dict, List

from arxitex.extractor.utils import ArtifactNode


def save_graph_data(arxiv_id: str, graphs_output_dir: str, graph_data: dict) -> Path:
    """Saves the generated graph data to a persistent JSON file."""
    safe_paper_id = arxiv_id.replace("/", "_")
    graph_filename = f"{safe_paper_id}.json"
    graph_filepath = Path(graphs_output_dir) / graph_filename

    with open(graph_filepath, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    return graph_filepath


def transform_graph_to_search_format(
    graph_nodes: List[ArtifactNode],
    artifact_to_terms_map: Dict[str, List[str]] = None,
    paper_metadata: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Transforms the raw artifacts from a graph into a flat list of dictionaries
    optimized for search indexing.

    Args:
        graph_nodes: A list of ArtifactNode objects from the DocumentGraph.
        artifact_to_terms_map: A map from artifact ID to its extracted terms.
        paper_metadata: Metadata about the paper, used to enrich each artifact's search document.

    Returns:
        A list of search-ready artifact dictionaries.
    """
    searchable_artifacts = []

    artifact_to_terms_map = artifact_to_terms_map or {}
    paper_metadata = paper_metadata or {}

    base_paper_info = {
        key: value for key, value in paper_metadata.items() if key != "id"
    }

    for node in graph_nodes:
        if node.is_external:
            continue

        # The prerequisite block is identified by a consistent header.
        # See document_enhancer.py for where this is added.
        content_full = node.content
        prereq_header = "--- Prerequisite Definitions ---"
        if prereq_header in node.content:
            content_full = node.content.split(prereq_header, 1)[-1].strip()
            content_full = content_full.split("---\n\n", 1)[-1].strip()

        search_doc = {
            **base_paper_info,
            "artifact_id": node.id,
            "artifact_type": node.type.value,
            "content": content_full,
            "terms": (artifact_to_terms_map or {}).get(node.id, []),
            "proof": node.proof,
        }
        searchable_artifacts.append(search_doc)

    return searchable_artifacts
