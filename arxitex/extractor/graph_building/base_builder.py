import re
import uuid
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

from arxitex.extractor.utils import (
    ArtifactNode, Edge, DocumentGraph, Position, Reference,
    ArtifactType, ReferenceType
)

class BaseGraphBuilder:
    """
    Parses LaTeX source code to build a DocumentGraph based on environments
    and explicit references (e.g., \\ref, \\cref).
    
    This class handles Pass 1 of the extraction process, focusing on high-recall
    structural parsing.
    """
    ARTIFACT_TYPES: Set[str] = {
        'theorem', 'lemma', 'proposition', 'corollary',
        'definition', 'remark', 'example',
        'claim', 'observation', 'fact', 'conjecture', 'unknown'
    }
    PROOF_ENV_TYPE = 'proof'
    
    def __init__(self):
        self.artifact_type_map = {name: ArtifactType(name) for name in self.ARTIFACT_TYPES}
        self.content: str = ""
        self.label_to_node_id_map: Dict[str, str] = {}

    def build_graph(self, latex_content: str, source_file: Optional[str] = None) -> DocumentGraph:
        logger.debug("Starting LaTeX graph extraction.")
        self.content = re.sub(r'(?<!\\)%.*', '', latex_content)
        
        graph = DocumentGraph(source_file=source_file)
        self.label_to_node_id_map.clear()

        # --- Pass 1: Parse all environments and create nodes ---
        nodes_to_process = self._parse_all_environments()
        for node in nodes_to_process:
            graph.add_node(node)
        logger.info(f"Pass 1: Parsed {len(graph.nodes)} initial artifact nodes.")

        # --- Pass 2: Resolve references and create all links (edges) ---
        edges_to_add, external_nodes_to_add = self._resolve_and_create_graph_links(graph.nodes)
        
        for node in external_nodes_to_add:
            graph.add_node(node)
        for edge in edges_to_add:
            graph.add_edge(edge)
        logger.info(f"Pass 2: Resolved references, creating {len(edges_to_add)} edges.")
        
        logger.success(f"Regex-based graph extraction complete. Found {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph

    def _parse_all_environments(self) -> List[ArtifactNode]:
        """Iterates through the content to find and parse all known artifact environments."""
        nodes: List[ArtifactNode] = []
        pattern = re.compile(rf'\\begin\{{({"|".join(self.ARTIFACT_TYPES)})(\*?)\}}')

        artifact_counter = 0
        cursor = 0
        while cursor < len(self.content):
            match = pattern.search(self.content, cursor)
            if not match:
                break

            env_type, star = match.group(1).lower(), match.group(2) or ""
            block_start = match.end()
            end_tag_pos = self._find_matching_end(env_type, star, block_start)
            
            if end_tag_pos == -1:
                cursor = match.end()
                continue
            
            next_cursor_pos = end_tag_pos + len(f"\\end{{{env_type}{star}}}")
            raw_content = self.content[block_start:end_tag_pos].strip()
            
            proof_content, next_cursor_pos = self._extract_following_proof(next_cursor_pos)
            
            artifact_counter += 1
            node_id = f"{env_type}-{artifact_counter}-{uuid.uuid4().hex[:6]}"
            label = self._extract_label(raw_content)

            if label:
                if label in self.label_to_node_id_map:
                    logger.warning(f"Duplicate LaTeX label '{label}' found. References may be ambiguous.")
                self.label_to_node_id_map[label] = node_id

            node = ArtifactNode(
                id=node_id,
                type=self.artifact_type_map[env_type],
                content=raw_content,
                label=label,
                position=self._calculate_position(match.start(), end_tag_pos),
                references=self._extract_references_from_content(raw_content, proof_content),
                is_external=False,
                proof=proof_content
            )
            nodes.append(node)
            cursor = next_cursor_pos
        
        return nodes

    def _find_matching_end(self, env_type: str, star: str, start_pos: int) -> int:
        """Finds the matching \\end{env} tag, handling nested environments."""
        begin_tag = f"\\begin{{{env_type}{star}}}"
        end_tag = f"\\end{{{env_type}{star}}}"
        nesting_level = 1
        cursor = start_pos
        
        while nesting_level > 0:
            next_begin = self.content.find(begin_tag, cursor)
            next_end = self.content.find(end_tag, cursor)
            
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

    def _extract_following_proof(self, start_pos: int) -> Tuple[Optional[str], int]:
        """Looks for a proof environment immediately following an artifact."""
        proof_pattern = re.compile(r'\s*\\begin\{proof(\*?)\}')
        proof_match = proof_pattern.match(self.content, start_pos)
        if not proof_match:
            return None, start_pos
        
        proof_star = proof_match.group(1) or ""
        proof_content_start = proof_match.end()
        proof_end_pos = self._find_matching_end(self.PROOF_ENV_TYPE, proof_star, proof_content_start)
        
        if proof_end_pos != -1:
            proof_content = self.content[proof_content_start:proof_end_pos].strip()
            new_cursor_pos = proof_end_pos + len(f"\\end{{{self.PROOF_ENV_TYPE}{proof_star}}}")
            return proof_content, new_cursor_pos
        
        return None, start_pos

    def _extract_label(self, content: str) -> Optional[str]:
        """Extracts the first \\label from a content string."""
        label_match = re.search(r'\\label\s*\{([^}]+)\}', content)
        return label_match.group(1).strip() if label_match else None

    def _calculate_position(self, start_offset: int, end_offset: int) -> Position:
        """Calculates line numbers from character offsets."""
        line_start = self.content[:start_offset].count('\n') + 1
        line_end = self.content[:end_offset].count('\n') + 1

        # Find the position of the last newline before the start of the artifact
        last_newline_before_start = self.content.rfind('\n', 0, start_offset)
        if last_newline_before_start == -1:
            col_start = start_offset + 1
        else:
            col_start = start_offset - last_newline_before_start

        # Find the position of the last newline before the end of the artifact
        last_newline_before_end = self.content.rfind('\n', 0, end_offset)
        if last_newline_before_end == -1:
            col_end = end_offset + 1
        else:
            col_end = end_offset - last_newline_before_end

        return Position(
            line_start=line_start,
            line_end=line_end,
            col_start=col_start,
            col_end=col_end
        )

    def _extract_references_from_content(self, content: str, proof_content: Optional[str]) -> List[Reference]:
        """Extract all references from artifact and proof content."""
        references = []
        ref_pattern = re.compile(r'\\(?:[cC]ref|[vV]ref|[Aa]utoref|ref|eqref)\s*\{([^}]+)\}')
        
        full_content = content + (f"\n\n{proof_content}" if proof_content else '')
        for match in ref_pattern.finditer(full_content):
            target_labels = [label.strip() for label in match.group(1).split(',')]
            
            for target_label in target_labels:
                start = max(0, match.start() - 50)
                end = min(len(full_content), match.end() + 50)
                context = full_content[start:end].replace('\n', ' ').strip()
                
                references.append(Reference(
                    target_id=target_label,
                    reference_type=ReferenceType.INTERNAL,  # Assume internal until resolved
                    context=context,
                    position=self._calculate_position(match.start(), match.end())
                ))
        return references
    
    def _resolve_and_create_graph_links(self, nodes: List[ArtifactNode]) -> Tuple[List[Edge], List[ArtifactNode]]:
        """Uses the label map to resolve references and create graph components."""
        edges: List[Edge] = []
        new_external_nodes: List[ArtifactNode] = []
        created_external_labels: Set[str] = set()

        for source_node in nodes:
            if source_node.is_external:
                continue
            
            for ref in source_node.references:
                target_label = ref.target_id
                
                if target_label in self.label_to_node_id_map:
                    target_node_id = self.label_to_node_id_map[target_label]
                    ref.reference_type = ReferenceType.INTERNAL
                    
                    if target_node_id != source_node.id:
                        edges.append(Edge(
                            source_id=source_node.id,
                            target_id=target_node_id,
                            context=ref.context,
                            reference_type=ReferenceType.INTERNAL
                        ))
                else:
                    ref.reference_type = ReferenceType.EXTERNAL
                    external_node_id = f"external_{target_label}"
                    
                    if target_label not in created_external_labels:
                        external_node = ArtifactNode(
                            id=external_node_id,
                            label=target_label,
                            type=ArtifactType.UNKNOWN,
                            content=f"External reference to '{target_label}'",
                            is_external=True
                        )
                        new_external_nodes.append(external_node)
                        created_external_labels.add(target_label)
                    
                    edges.append(Edge(
                        source_id=source_node.id,
                        target_id=external_node_id,
                        context=ref.context,
                        reference_type=ReferenceType.EXTERNAL
                    ))
        return edges, new_external_nodes