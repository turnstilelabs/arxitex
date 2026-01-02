import hashlib
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from arxitex.downloaders.utils import read_and_combine_tex_files
from arxitex.extractor.graph_building.newtheorem_scanner import NewTheoremScanner
from arxitex.extractor.graph_building.proof_linker import ProofLinker
from arxitex.extractor.graph_building.reference_resolver import ReferenceResolver
from arxitex.extractor.models import ArtifactNode, ArtifactType, DocumentGraph, Position


class BaseGraphBuilder:
    """
    Parses LaTeX source code to build a DocumentGraph based on environments
    and explicit references (e.g., \\ref, \\cref).

    This class handles Pass 1 of the extraction process, focusing on high-recall
    structural parsing.
    """

    ARTIFACT_TYPES: Set[str] = {
        "theorem",
        "lemma",
        "proposition",
        "corollary",
        "definition",
        "remark",
        "example",
        "claim",
        "observation",
        "fact",
        "conjecture",
        "unknown",
    }
    PROOF_ENV_TYPE = "proof"

    # Mapping from LaTeX environment names (often custom/abbreviated) to
    # canonical artifact type names used in ARTIFACT_TYPES.
    ENV_NAME_ALIASES: Dict[str, str] = {
        # Theorems
        "thm": "theorem",
        "Thm": "theorem",
        "THEOREM": "theorem",
        # Definitions
        "defn": "definition",
        "defi": "definition",
        "Def": "definition",
        # Propositions
        "prop": "proposition",
        "Prop": "proposition",
        # Lemmas (short aliases; canonical "lemma" is already in ARTIFACT_TYPES)
        "lem": "lemma",
        "Lem": "lemma",
        # Corollaries
        "cor": "corollary",
        "Cor": "corollary",
        # Claims (short alias; canonical "claim" already present)
        "clm": "claim",
        # Observations
        "obs": "observation",
        # Conjectures
        "conj": "conjecture",
        # Remarks
        "Rem": "remark",
    }

    def __init__(self):
        self.artifact_type_map = {
            name: ArtifactType(name) for name in self.ARTIFACT_TYPES
        }
        self.content: str = ""
        self.label_to_node_id_map: Dict[str, str] = {}
        self._newline_offsets: List[int] = []

    def build_graph(
        self,
        project_dir: Optional[Path] = None,
        source_file: Optional[str] = None,
    ) -> DocumentGraph:
        logger.debug("Starting LaTeX graph extraction.")

        content = read_and_combine_tex_files(project_dir)
        return self.build_graph_from_content(
            content=content, source_file=source_file, project_dir=project_dir
        )

    def build_graph_from_content(
        self,
        content: str,
        source_file: Optional[str] = None,
        project_dir: Optional[Path] = None,
    ) -> DocumentGraph:
        """Build a graph from an already combined LaTeX string.

        This lets upstream orchestration (e.g., GraphEnhancer) avoid reading and
        concatenating TeX sources multiple times.
        """
        self.content = content
        self.content = re.sub(r"(?<!\\)%.*", "", self.content)
        self._newline_offsets = [m.start() for m in re.finditer("\n", self.content)]

        # Build a per-document alias map by combining static aliases with
        # aliases discovered from \newtheorem declarations.
        dynamic_aliases = NewTheoremScanner.scan(self.content)
        self.env_name_aliases: Dict[str, str] = {
            **self.ENV_NAME_ALIASES,
            **dynamic_aliases,
        }

        graph = DocumentGraph(source_file=source_file)
        self.label_to_node_id_map.clear()

        self.proof_linker = ProofLinker(self.content)
        self.reference_resolver = ReferenceResolver(self.content)

        # 1. Discover all artifacts and proof environments
        nodes, detached_proofs, node_char_offsets = (
            self._parse_all_environments_and_proofs()
        )

        # 2. Link proofs to their statements
        self.proof_linker.link_proofs(nodes, detached_proofs, node_char_offsets)

        # 3. Delegate ALL reference and citation handling to the new resolver.
        if project_dir is None:
            raise ValueError(
                "project_dir must be provided to resolve bibliography/citations."
            )

        edges, external_nodes = self.reference_resolver.resolve_all_references(
            project_dir, nodes, self.label_to_node_id_map
        )

        # 4. Build the final graph.
        for node in nodes:
            graph.add_node(node)
        for node in external_nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)

        logger.success(
            f"Regex-based graph extraction complete. Found {len(graph.nodes)} nodes and {len(graph.edges)} edges."
        )
        return graph

    def _parse_all_environments_and_proofs(
        self,
    ) -> Tuple[List[ArtifactNode], List[Dict], Dict[str, Tuple[int, int]]]:
        """
        Iterates through the content to find all artifacts and all proof environments.
        Returns nodes, temporary proof dictionaries, and a temporary map of node character offsets.
        """
        nodes: List[ArtifactNode] = []
        detached_proofs: List[Dict] = []
        node_char_offsets: Dict[str, Tuple[int, int]] = {}

        all_env_types = "|".join(
            sorted(
                set(
                    list(self.ARTIFACT_TYPES)
                    + list(self.env_name_aliases.keys())
                    + [self.PROOF_ENV_TYPE]
                )
            )
        )
        pattern = re.compile(rf"\\begin\{{({all_env_types})(\*?)\}}(?:\[([^\]]*)\])?")

        artifact_counter = 0
        cursor = 0
        while cursor < len(self.content):
            match = pattern.search(self.content, cursor)
            if not match:
                break

            raw_env_type = match.group(1)
            env_type = self.env_name_aliases.get(raw_env_type, raw_env_type).lower()
            star = match.group(2) or ""
            optional_arg = match.group(3)

            block_start = match.end()
            end_tag_pos = self._find_matching_end(raw_env_type, star, block_start)

            if end_tag_pos == -1:
                cursor = match.end()
                continue

            # Use the *raw* environment name when computing the end tag length so
            # that aliases like "thm" (canonicalized to "theorem") still produce
            # correct character offsets.
            full_end_pos = end_tag_pos + len(f"\\end{{{raw_env_type}{star}}}")
            raw_content = self.content[block_start:end_tag_pos].strip()

            if env_type == self.PROOF_ENV_TYPE:
                proof_block = {
                    "content": raw_content,
                    "optional_arg": optional_arg,
                    "start_char": match.start(),
                    "end_char": full_end_pos,
                    "used": False,
                }
                detached_proofs.append(proof_block)
            else:
                artifact_counter += 1
                label = self._extract_label(raw_content)

                # Stable IDs are critical for the webapp streaming use-case.
                # Prefer label-based IDs when available, otherwise fall back to a
                # deterministic hash based on environment type + document position.
                if label:
                    safe_label = re.sub(r"[^A-Za-z0-9_.:-]+", "_", label)
                    node_id = f"{env_type}:{safe_label}"
                else:
                    # Use raw env type + artifact counter + source offsets to
                    # make IDs stable across runs for the same document.
                    pos_sig = f"{env_type}|{match.start()}|{full_end_pos}"
                    digest = hashlib.sha1(pos_sig.encode("utf-8")).hexdigest()[:8]
                    node_id = f"{env_type}-{artifact_counter}-{digest}"

                if label:
                    self.label_to_node_id_map[label] = node_id

                node = ArtifactNode(
                    id=node_id,
                    type=self.artifact_type_map[env_type],
                    content=raw_content,
                    label=label,
                    position=self._calculate_position(match.start(), full_end_pos),
                    is_external=False,
                    proof=None,
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
                logger.warning(
                    f"Could not find matching {end_tag} for environment starting near position {start_pos}."
                )
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
        label_match = re.search(r"\\label\s*\{([^}]+)\}", content)
        return label_match.group(1).strip() if label_match else None

    def _calculate_position(self, start_offset: int, end_offset: int) -> Position:
        """Calculates line/column numbers from character offsets.

        Uses a precomputed newline index to avoid O(doc_length) work per node.
        """

        # Number of newlines strictly before the offset.
        line_start = bisect_right(self._newline_offsets, start_offset - 1) + 1
        line_end = bisect_right(self._newline_offsets, end_offset - 1) + 1

        def col_at(offset: int) -> int:
            idx = bisect_right(self._newline_offsets, offset - 1) - 1
            last_nl = self._newline_offsets[idx] if idx >= 0 else -1
            return offset - last_nl

        col_start = col_at(start_offset)
        col_end = col_at(end_offset)

        return Position(
            line_start=line_start,
            line_end=line_end,
            col_start=col_start,
            col_end=col_end,
        )
