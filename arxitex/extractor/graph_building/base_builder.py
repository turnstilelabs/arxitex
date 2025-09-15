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

        # 1. Discover all artifacts and proof environments
        nodes, detached_proofs, node_char_offsets = self._parse_all_environments_and_proofs()
        
        # 2. Link proofs to their statements
        self.proof_linker.link_proofs(nodes, detached_proofs, node_char_offsets, self.content)

        # 3. Parse the bibliography ONCE to create the lookup map.
        bibliography_map = self._find_and_parse_bibliography(project_dir)

        # 4. Extract references from all content
        logger.info("Extracting references from artifact content...")
        for node in nodes:
            node.references = self._extract_references(node, bibliography_map)
            
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

    def _find_and_parse_bibliography(self, project_dir: Path) -> Dict[str, Dict]:
        """
        Finds and parses ALL bibliography files in the project, prioritizing .bbl files
        and merging the contents of all found files.
        """
        # Strategy 1: Search for an embedded bibliography first
        embedded_bib_match = re.search(r'\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}', self.content, re.DOTALL)
        if embedded_bib_match:
            logger.info("Found embedded 'thebibliography' environment. Parsing it.")
            embedded_bbl_content = embedded_bib_match.group(1)
            return self._parse_bbl_content(embedded_bbl_content)
        
        # Strategy 2: Look for .bbl files
        bbl_files = list(project_dir.rglob('*.bbl'))
        if bbl_files:
            logger.info(f"Found {len(bbl_files)} .bbl file(s). Parsing all of them.")
            final_bib_map = {}
            for bbl_file in bbl_files:
                try:
                    bbl_content = bbl_file.read_text(encoding='utf-8', errors='ignore')
                    parsed_map = self._parse_bbl_content(bbl_content)
                    final_bib_map.update(parsed_map)
                except Exception as e:
                    logger.warning(f"Could not parse .bbl file {bbl_file.name}: {e}")
            return final_bib_map

        # Strategy 3: Fallback to .bib files
        bib_files = list(project_dir.rglob('*.bib'))
        if bib_files:
            logger.info(f"No .bbl files found. Found {len(bib_files)} .bib file(s). Parsing all of them.")
            final_bib_map = {}
            for bib_file in bib_files:
                try:
                    bib_content = bib_file.read_text(encoding='utf-8', errors='ignore')
                    parsed_map = self._parse_bib_content(bib_content)
                    final_bib_map.update(parsed_map)
                except Exception as e:
                    logger.warning(f"Could not parse .bib file {bib_file.name}: {e}")
            return final_bib_map
        
        logger.warning("No .bbl or .bib files found in the project directory. Cannot parse bibliography.")
        return {}

    def _parse_bbl_content(self, bbl_content: str) -> Dict[str, Dict]:
        bib_map = {}
        pattern = re.compile(r'\\bibitem(?:\[(.*?)\])?\{(.*?)\}(.*?)(?=\\bibitem|\s*\\end)', re.DOTALL)
        for match in pattern.finditer(bbl_content):
            optional_key, mandatory_key, ref_text = match.groups()
            ref_text = re.sub(r'\s+', ' ', ref_text).strip()
            arxiv_match = re.search(r'(?:arxiv[:\s]*|eprint\s*=\s*\{s*)([\d\.\/v-]+)', ref_text, re.IGNORECASE)
            
            reference_data = {
                "full_reference": ref_text,
                "arxiv_id": arxiv_match.group(1).strip() if arxiv_match else None
            }
            
            if mandatory_key:
                bib_map[mandatory_key.strip()] = reference_data
            if optional_key:
                bib_map[optional_key.strip()] = reference_data
                
        logger.info(f"Parsed {len(bib_map)} unique cite keys from .bbl content.")
        return bib_map

    def _parse_bib_content(self, bib_content: str) -> Dict[str, Dict]:
        bib_map = {}
        pattern = re.compile(r'@\w+\s*\{(.*?),(.*?)(?=\n@|\Z)', re.DOTALL)
        for match in pattern.finditer(bib_content):
            cite_key, fields_str = match.groups()
            ref_text = f"{cite_key}: {fields_str.strip()}"
            arxiv_match = re.search(r'(?:archivePrefix|eprint)\s*=\s*.*?([\d\.\/v-]+)', fields_str)
            bib_map[cite_key.strip()] = { "full_reference": ref_text, "arxiv_id": arxiv_match.group(1).strip() if arxiv_match else None }
        return bib_map
    
    def _extract_references(self, node: ArtifactNode, bib_map: Dict[str, Dict]) -> List[Reference]:
        """Extract all references (\ref-style and \cite-style) from artifact and proof content."""
        references = []
        parts = []
        if node.content: parts.append(node.content)
        if node.proof: parts.append(node.proof)
        full_content = "\n\n".join(parts)

        if not full_content:
            logger.warning(f"Node {node.id} has no content or proof to extract references from.")
            return references
        
        explicit_pattern = re.compile(
            r'\\(?P<ref_cmd>[cC]ref|[vV]ref|[Aa]utoref|ref|eqref)\s*\{(?P<ref_keys>[^}]+)\}|'
            r'\\(?P<cite_cmd>cite[pt]?\*?)(?:\[(?P<cite_note>[^\]]*)\])?\{(?P<cite_keys>[^}]+)\}'
        )
        
        found_cite_keys = set()
        
        for match in explicit_pattern.finditer(full_content):
            context_start = max(0, match.start() - 50)
            context_end = min(len(full_content), match.end() + 50)
            context = full_content[context_start:context_end].replace('\n', ' ').strip()
            position = self._calculate_position(match.start(), match.end())

            # Case 1: It was an internal reference (\ref, \Cref, etc.)
            if match.group('ref_cmd'):
                for key in match.group('ref_keys').split(','):
                    references.append(Reference(
                        target_id=key.strip(),
                        reference_type=ReferenceType.INTERNAL,
                        context=context,
                        position=position
                    ))
            
            # Case 2: It was an external citation (\cite, \citep, etc.)
            elif match.group('cite_cmd'):
                note = match.group('cite_note')
                for key in match.group('cite_keys').split(','):
                    key = key.strip()
                    found_cite_keys.add(key) 
                    
                    if key in bib_map:
                        bib_entry = bib_map[key]
                        references.append(Reference(
                            target_id=key,
                            reference_type=ReferenceType.EXTERNAL,
                            context=context,
                            position=position,
                            full_reference=bib_entry["full_reference"],
                            arxiv_id=bib_entry["arxiv_id"],
                            note=note.strip() if note else None
                        ))
                    else:
                        logger.warning(
                            f"Unresolved citation: Found cite key '{key}' in the text, "
                            f"but it was not found in the parsed bibliography."
                        )
                        references.append(Reference(
                            target_id=key,
                            reference_type=ReferenceType.EXTERNAL,
                            context=context,
                            position=position,
                            full_reference=f"UNRESOLVED: Citation key '{key}' not found in bibliography.",
                            arxiv_id=None,
                            note=note.strip() if note else None
                        ))

        # Now, search for any bib keys that were NOT found via an explicit \cite command.
        # This ensures we match "Rou01" before we match "Rou".
        sorted_bib_keys = sorted(bib_map.keys(), key=len, reverse=True)

        for cite_key in sorted_bib_keys:
            if cite_key in found_cite_keys:
                continue

            # This finds the key when it's used as a "word" like [AuthorYear, Theorem X].
            escaped_key = re.escape(cite_key)
            manual_pattern = re.compile(
                r'([\(\[][^[\]\(\)]*\b' + escaped_key + r'\b[^[\]\(\)]*[\)\]])'
            )

            for match in manual_pattern.finditer(full_content):
                logger.debug(f"Found manual citation for key '{cite_key}' in node {node.id}")
                
                context_start = max(0, match.start() - 50)
                context_end = min(len(full_content), match.end() + 50)
                context = full_content[context_start:context_end].replace('\n', ' ').strip()
                position = self._calculate_position(match.start(), match.end())
                bib_entry = bib_map[cite_key]
                
                full_match_text = match.group(1)            
                inner_content = full_match_text.strip("[]()")           
                note_text = re.sub(r'\b' + escaped_key + r'\b', '', inner_content)
                note = note_text.strip(" ,")
                
                if not note:
                    note = None

                if not any(r.target_id == cite_key and r.note == note for r in references):
                    references.append(Reference(
                        target_id=cite_key, 
                        reference_type=ReferenceType.EXTERNAL,
                        context=context, 
                        position=position,
                        full_reference=bib_entry["full_reference"], 
                        arxiv_id=bib_entry["arxiv_id"],
                        note=note
                    ))
                break
            
        return references
    
    def _resolve_and_create_graph_links(self, nodes: List[ArtifactNode]) -> Tuple[List[Edge], List[ArtifactNode]]:
        """Uses the label map to resolve references and create graph components."""
        edges: List[Edge] = []
        new_external_nodes: List[ArtifactNode] = []
        created_external_nodes_map: Dict[str, str] = {} # Maps target_id to node_id

        for source_node in nodes:
            if source_node.is_external:
                continue
            
            for ref in source_node.references:
                if ref.reference_type == ReferenceType.INTERNAL:
                    target_label = ref.target_id
                    if target_label in self.label_to_node_id_map:
                        target_node_id = self.label_to_node_id_map[target_label]
                        if target_node_id != source_node.id:
                            edges.append(Edge(
                                source_id=source_node.id,
                                target_id=target_node_id,
                                context=ref.context,
                                reference_type=ReferenceType.INTERNAL
                            ))
                    else:
                        logger.warning(f"Dangling internal reference found: \
            Node '{source_node.id}' refers to missing label '{target_label}'.")
                        if target_label not in created_external_nodes_map:
                            external_node_id = f"external_{target_label}"
                            external_node = ArtifactNode(
                                id=external_node_id,
                                label=target_label,
                                type=ArtifactType.UNKNOWN,
                                content=f"Dangling reference to missing internal label: '{target_label}'",
                                is_external=True
                            )
                            new_external_nodes.append(external_node)
                            created_external_nodes_map[target_label] = external_node_id
                        
                        edges.append(Edge(
                            source_id=source_node.id,
                            target_id=created_external_nodes_map[target_label],
                            context=ref.context,
                            reference_type=ReferenceType.EXTERNAL
                        ))
                                        
                elif ref.reference_type == ReferenceType.EXTERNAL:
                    target_key = ref.target_id                    
                    if target_key in created_external_nodes_map:
                        external_node_id = created_external_nodes_map[target_key]
                    else:
                        external_node_id = f"external_{target_key}"
                        external_node = ArtifactNode(
                            id=external_node_id,
                            label=target_key,
                            type=ArtifactType.UNKNOWN,
                            is_external=True
                        )
                        new_external_nodes.append(external_node)
                        created_external_nodes_map[target_key] = external_node_id

                    edges.append(Edge(
                        source_id=source_node.id,
                        target_id=external_node_id,
                        context=ref.context,
                        reference_type=ReferenceType.EXTERNAL
                    ))

        return edges, new_external_nodes