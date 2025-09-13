import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

from arxitex.extractor.utils import (
    ArtifactNode, Edge, DocumentGraph, Position, Reference,
    ArtifactType, ReferenceType
)

from arxitex.extractor.graph_building.proof_linker import ProofLinker
from arxitex.extractor.graph_building.citation_enricher import CitationEnricher
from arxitex.downloaders.utils import read_and_combine_tex_files

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

        self.proof_linker = ProofLinker()
        self.citation_enricher = CitationEnricher()

    def build_graph(
        self, 
        project_dir: Optional[Path] = None,
        source_file: Optional[str] = None,
    ) -> DocumentGraph:
        logger.debug("Starting LaTeX graph extraction.")
        self.content = read_and_combine_tex_files(project_dir)
        self.content = re.sub(r'(?<!\\)%.*', '', self.content)
        
        graph = DocumentGraph(source_file=source_file)
        self.label_to_node_id_map.clear()

        # 1. Discover all artifacts AND all proof environments
        nodes, detached_proofs, node_char_offsets = self._parse_all_environments_and_proofs()
        
        # 2. Link detached proofs to their statements
        self.proof_linker.link_proofs(nodes, detached_proofs, node_char_offsets, self.content)

        # 3. Enrich citations using .bbl or .bib files if available
        self.citation_enricher.enrich(project_dir, nodes)

        # 4. Extract references from all content (including proofs)
        logger.info("Extracting internal references from all artifact content (including proofs)...")
        for node in nodes:
            node.references = self._extract_references_from_content(node.content, node.proof)
            
        for node in nodes:
            graph.add_node(node)
        logger.info(f"Pass 1: Parsed {len(graph.nodes)} artifacts and linked detached proofs.")

        # 5. Resolve references and create all links (edges) ---
        edges_to_add, external_nodes_to_add = self._resolve_and_create_graph_links(graph.nodes)
        
        for node in external_nodes_to_add:
            graph.add_node(node)
        for edge in edges_to_add:
            graph.add_edge(edge)
        logger.info(f"Pass 2: Resolved references, creating {len(edges_to_add)} edges.")
        
        logger.success(f"Regex-based graph extraction complete. Found {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph

    def _parse_all_environments_and_proofs(self) -> Tuple[List[ArtifactNode], List[Dict], Dict[str, Tuple[int, int]]]:
        """
        Iterates through the content to find all artifacts and all proof environments.
        Returns nodes, temporary proof dictionaries, and a temporary map of node character offsets.
        """
        nodes: List[ArtifactNode] = []
        detached_proofs: List[Dict] = []
        node_char_offsets: Dict[str, Tuple[int, int]] = {}
        
        all_env_types = "|".join(list(self.ARTIFACT_TYPES) + [self.PROOF_ENV_TYPE])
        pattern = re.compile(rf'\\begin\{{({all_env_types})(\*?)\}}(?:\[([^\]]*)\])?')

        artifact_counter = 0
        cursor = 0
        while cursor < len(self.content):
            match = pattern.search(self.content, cursor)
            if not match:
                break

            env_type, star, optional_arg = match.group(1).lower(), match.group(2) or "", match.group(3)
            block_start = match.end()
            end_tag_pos = self._find_matching_end(env_type, star, block_start)
            
            if end_tag_pos == -1:
                cursor = match.end()
                continue
            
            full_end_pos = end_tag_pos + len(f"\\end{{{env_type}{star}}}")
            raw_content = self.content[block_start:end_tag_pos].strip()
            
            if env_type == self.PROOF_ENV_TYPE:
                proof_block = {
                    "content": raw_content, "optional_arg": optional_arg,
                    "start_char": match.start(), "end_char": full_end_pos,
                    "used": False
                }
                detached_proofs.append(proof_block)
            else:
                artifact_counter += 1
                node_id = f"{env_type}-{artifact_counter}-{uuid.uuid4().hex[:6]}"
                label = self._extract_label(raw_content)

                if label: self.label_to_node_id_map[label] = node_id

                node = ArtifactNode(
                    id=node_id, type=self.artifact_type_map[env_type],
                    content=raw_content, label=label,
                    position=self._calculate_position(match.start(), full_end_pos),
                    is_external=False, proof=None
                )
                nodes.append(node)
                # Store the character offsets in our temporary dictionary.
                node_char_offsets[node.id] = (match.start(), full_end_pos)
            cursor = full_end_pos

        return nodes, detached_proofs, node_char_offsets 
    
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
        
        full_content = (content or "") + (f"\n\n{proof_content}" if proof_content else '')
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