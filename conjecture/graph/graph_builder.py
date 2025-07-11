# arxiv_graph_extractor/graph_builder.py
"""
Builds the graph structure from LaTeX source using regex.
"""

import re
import uuid
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

from conjecture.graph.utils import (
    ArtifactNode, Edge, DocumentGraph, Position, Reference,
    ArtifactType, ReferenceType
)

ARTIFACT_TYPES: Set[str] = {
    'theorem', 'lemma', 'proposition', 'corollary',
    'definition', 'remark', 'example',
    'claim', 'observation', 'fact', 'conjecture','unknown'
}

PROOF_ENV_TYPE = 'proof'
ARTIFACT_TYPE_MAP = {name: ArtifactType(name) for name in ARTIFACT_TYPES}

def _find_matching_end(content: str, env_type: str, star: str, start_pos: int) -> int:
    """Finds the matching \\end{env} tag, handling nested environments."""
    begin_tag = f"\\begin{{{env_type}{star}}}"
    end_tag = f"\\end{{{env_type}{star}}}"
    nesting_level = 1
    cursor = start_pos
    
    while nesting_level > 0:
        next_begin = content.find(begin_tag, cursor)
        next_end = content.find(end_tag, cursor)
        
        if next_end == -1:
            logger.warning(f"Could not find matching {end_tag} for environment starting near position {start_pos}.")
            return -1
            
        if next_begin != -1 and next_begin < next_end:
            nesting_level += 1
            cursor = next_begin + len(begin_tag)
        else:
            nesting_level -= 1
            if nesting_level == 0:
                return next_end
            cursor = next_end + len(end_tag)
    
    return -1

def _extract_references_from_content(content: str, proof_content: str) -> List[Reference]:
    """Extract all references from artifact content."""
    references = []
    ref_pattern = re.compile(r'\\(?:[cC]ref|[vV]ref|[Aa]utoref|ref|eqref)\s*\{([^}]+)\}')
    
    content = content + (proof_content if proof_content else '')
    for match in ref_pattern.finditer(content):
        target_labels = [label.strip() for label in match.group(1).split(',')]
        
        for target_label in target_labels:
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].strip()
            
            references.append(Reference(
                target_id=target_label,  # Will be resolved later
                reference_type=ReferenceType.INTERNAL,  # Default assumption
                context=context,
                position=Position(line_start=0)  # Could be enhanced
            ))
    
    return references


def _resolve_and_create_graph_links(
    nodes: List[ArtifactNode], 
    label_to_node_id_map: Dict[str, str]
) -> Tuple[List[Edge], List[ArtifactNode]]:
    """
    Uses the label map to resolve references and create edges and external nodes.

    Args:
        nodes: The list of already parsed artifact nodes.
        label_to_node_id_map: A dictionary mapping LaTeX labels to unique node IDs.

    Returns:
        A tuple containing:
        - A list of Edge objects to be added to the graph.
        - A list of new external ArtifactNode objects to be added to the graph.
    """
    edges: List[Edge] = []
    new_external_nodes: List[ArtifactNode] = []
    created_external_labels: Set[str] = set()

    for source_node in nodes:
        if source_node.is_external:
            continue
            
        for ref in source_node.references:
            target_label = ref.target_id # This is the LaTeX label from \ref{...}

            # Case 1: The reference is INTERNAL and can be resolved.
            if target_label in label_to_node_id_map:
                target_node_id = label_to_node_id_map[target_label]
                ref.reference_type = ReferenceType.INTERNAL
                
                if target_node_id != source_node.id: # Avoid self-loops
                    edge = Edge(
                        source_id=source_node.id,
                        target_id=target_node_id, # The edge uses the REAL node ID.
                        context=ref.context,
                        reference_type=ReferenceType.INTERNAL
                    )
                    edges.append(edge)

            # Case 2: The reference is EXTERNAL.
            else:
                ref.reference_type = ReferenceType.EXTERNAL
                
                if target_label not in created_external_labels:
                    external_node_id = f"external_{target_label}"
                    external_node = ArtifactNode(
                        id=external_node_id,
                        label=target_label,
                        type=ArtifactType.UNKNOWN,
                        content=f"External reference to '{target_label}'",
                        is_external=True
                    )
                    new_external_nodes.append(external_node)
                    created_external_labels.add(target_label)
                
                edge = Edge(
                    source_id=source_node.id,
                    target_id=f"external_{target_label}", # The edge uses the external node's ID.
                    context=ref.context,
                    reference_type=ReferenceType.EXTERNAL
                )
                edges.append(edge)
                
    return edges, new_external_nodes

def build_graph_from_latex(latex_content: str, source_file: Optional[str] = None) -> DocumentGraph:
    """
    Extract artifacts and build a document graph from LaTeX source.
    """
    graph = DocumentGraph(source_file=source_file)
    label_to_node_id_map: Dict[str, str] = {}
    
    content = re.sub(r'(?<!\\)%.*', '', latex_content, flags=re.MULTILINE)
    
    # --- PASS 1: Parse all environments and create nodes ---
    nodes_to_process: List[ArtifactNode] = []
    artifact_types = '|'.join(ARTIFACT_TYPES)
    pattern = re.compile(rf'\\begin\{{({artifact_types})(\*?)\}}')
    
    artifact_counter = 0
    cursor = 0
    while cursor < len(content):
        match = pattern.search(content, cursor)
        if not match: break
            
        env_type, star = match.group(1).lower(), match.group(2) or ""
        if env_type not in ARTIFACT_TYPE_MAP:
            cursor = match.end()
            continue
            
        block_start = match.end()
        end_tag_str = f"\\end{{{env_type}{star}}}"
        end_tag_pos = _find_matching_end(content, env_type, star, block_start)
        
        if end_tag_pos == -1:
            cursor = match.end()
            continue
            
        next_cursor_pos = end_tag_pos + len(end_tag_str)
        raw_content = content[block_start:end_tag_pos].strip()
        
        # --- Proof Extraction Logic ---
        proof_content: Optional[str] = None
        proof_pattern = re.compile(r'\s*\\begin\{proof(\*?)\}')
        proof_match = proof_pattern.match(content, next_cursor_pos)
        if proof_match:
            proof_star = proof_match.group(1) or ""
            proof_content_start = proof_match.end()
            proof_end_pos = _find_matching_end(content, PROOF_ENV_TYPE, proof_star, proof_content_start)
            if proof_end_pos != -1:
                proof_content = content[proof_content_start:proof_end_pos].strip()
                proof_end_tag_str = f"\\end{{{PROOF_ENV_TYPE}{proof_star}}}"
                next_cursor_pos = proof_end_pos + len(proof_end_tag_str)

        # --- ID and Label Handling ---
        artifact_counter += 1
        node_id = f"{env_type}-{artifact_counter}-{str(uuid.uuid4())[:8]}"
        label_match = re.search(r'\\label\s*\{([^}]+)\}', raw_content)
        label = label_match.group(1).strip() if label_match else None
        
        if label:
            if label in label_to_node_id_map:
                logger.warning(f"Duplicate LaTeX label '{label}' found. References may be ambiguous.")
            label_to_node_id_map[label] = node_id

        node = ArtifactNode(
            id=node_id,
            type=ARTIFACT_TYPE_MAP[env_type],
            content=raw_content,
            label=label,
            position=Position(line_start=content[:match.start()].count('\n') + 1, line_end=content[:end_tag_pos].count('\n') + 1),
            references=_extract_references_from_content(raw_content, proof_content), #SHOULD extract form content and proof
            is_external=False,
            proof=proof_content
        )
        nodes_to_process.append(node)
        cursor = next_cursor_pos
        
    for node in nodes_to_process:
        graph.add_node(node)

    # --- PASS 2: Resolve references and create all links (edges) ---
    edges_to_add, external_nodes_to_add = _resolve_and_create_graph_links(
        graph.nodes, 
        label_to_node_id_map
    )
    
    for node in external_nodes_to_add:
        graph.add_node(node)
    for edge in edges_to_add:
        graph.add_edge(edge)
    
    logger.info(f"Graph extraction complete. Found {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph
