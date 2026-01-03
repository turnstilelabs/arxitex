import re
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from arxitex.extractor.models import (
    ArtifactNode,
    ArtifactType,
    Edge,
    Reference,
    ReferenceType,
)


class ReferenceResolver:
    """A self-contained class for all reference and citation processing."""

    def __init__(self, content: str):
        """Initializes the resolver with the document content."""
        self.content = content
        # Index all LaTeX labels present in the document to distinguish
        # artifact vs. non-artifact internal labels.
        self.all_labels_in_doc = self._index_all_labels(self.content)

    def resolve_all_references(
        self,
        project_dir: Path,
        nodes: List[ArtifactNode],
        label_to_node_id_map: Dict[str, str],
    ) -> Tuple[List[Edge], List[ArtifactNode]]:
        """Main entry point to find, parse, and resolve all references."""
        # 1. Parse the bibliography to create a lookup map.
        bib_map = self._find_and_parse_bibliography(project_dir)

        # 2. Extract a unified list of all Reference objects from every node.
        for node in nodes:
            node.references = self._extract_references_from_node(node, bib_map)

        # 3. Resolve these Reference objects into graph Edges and new external nodes.
        return self._create_graph_links(nodes, label_to_node_id_map)

    def _find_and_parse_bibliography(self, project_dir: Path) -> Dict[str, Dict]:
        """
        Finds and parses ALL bibliography files in the project, prioritizing .bbl files
        and merging the contents of all found files.
        """
        # Strategy 1: Search for an embedded bibliography first
        embedded_bib_match = re.search(
            r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}",
            self.content,
            re.DOTALL,
        )
        if embedded_bib_match:
            logger.info("Found embedded 'thebibliography' environment. Parsing it.")
            embedded_bbl_content = embedded_bib_match.group(1)
            return self._parse_bbl_content(embedded_bbl_content)

        # Strategy 2: Look for .bbl files
        bbl_files = list(project_dir.rglob("*.bbl"))
        if bbl_files:
            logger.info(f"Found {len(bbl_files)} .bbl file(s). Parsing all of them.")
            final_bib_map = {}
            for bbl_file in bbl_files:
                try:
                    bbl_content = bbl_file.read_text(encoding="utf-8", errors="ignore")
                    parsed_map = self._parse_bbl_content(bbl_content)
                    final_bib_map.update(parsed_map)
                except Exception as e:
                    logger.warning(f"Could not parse .bbl file {bbl_file.name}: {e}")
            return final_bib_map

        # Strategy 3: Fallback to .bib files
        bib_files = list(project_dir.rglob("*.bib"))
        if bib_files:
            logger.info(
                f"No .bbl files found. Found {len(bib_files)} .bib file(s). Parsing all of them."
            )
            final_bib_map = {}
            for bib_file in bib_files:
                try:
                    bib_content = bib_file.read_text(encoding="utf-8", errors="ignore")
                    parsed_map = self._parse_bib_content(bib_content)
                    final_bib_map.update(parsed_map)
                except Exception as e:
                    logger.warning(f"Could not parse .bib file {bib_file.name}: {e}")
            return final_bib_map

        logger.warning(
            "No .bbl or .bib files found in the project directory. Cannot parse bibliography."
        )
        return {}

    def _parse_bbl_content(self, bbl_content: str) -> Dict[str, Dict]:
        bib_map = {}
        pattern = re.compile(
            r"\\bibitem(?:\[(.*?)\])?\{(.*?)\}(.*?)(?=\\bibitem|\s*\\end)", re.DOTALL
        )
        for match in pattern.finditer(bbl_content):
            optional_key, mandatory_key, ref_text = match.groups()
            ref_text = re.sub(r"\s+", " ", ref_text).strip()
            arxiv_match = re.search(
                r"(?:arxiv[:\s]*|eprint\s*=\s*\{s*)([\d\.\/v-]+)",
                ref_text,
                re.IGNORECASE,
            )

            reference_data = {
                "full_reference": ref_text,
                "arxiv_id": arxiv_match.group(1).strip() if arxiv_match else None,
            }

            if mandatory_key:
                bib_map[mandatory_key.strip()] = reference_data
            if optional_key:
                bib_map[optional_key.strip()] = reference_data

        logger.info(f"Parsed {len(bib_map)} unique cite keys from .bbl content.")
        return bib_map

    def _parse_bib_content(self, bib_content: str) -> Dict[str, Dict]:
        bib_map = {}
        pattern = re.compile(r"@\w+\s*\{(.*?),(.*?)(?=\n@|\Z)", re.DOTALL)
        for match in pattern.finditer(bib_content):
            cite_key, fields_str = match.groups()
            ref_text = f"{cite_key}: {fields_str.strip()}"
            arxiv_match = re.search(
                r"(?:archivePrefix|eprint)\s*=\s*.*?([\d\.\/v-]+)", fields_str
            )
            bib_map[cite_key.strip()] = {
                "full_reference": ref_text,
                "arxiv_id": arxiv_match.group(1).strip() if arxiv_match else None,
            }
        return bib_map

    def _index_all_labels(self, content: str) -> set:
        """Scan the document content and collect every LaTeX label declared with \\label{...}."""
        labels = set()
        try:
            for m in re.finditer(r"\\label\s*\{([^}]+)\}", content):
                labels.add(m.group(1).strip())
        except Exception as e:
            logger.debug(f"Label indexing failed: {e}")
        return labels

    def _normalize_label(self, s: str) -> str:
        """Normalize labels to tolerate minor formatting differences (case and separators)."""
        if not s:
            return ""
        t = s.strip().lower()
        # Collapse runs of :, -, _, or whitespace to a single colon and trim surrounding colons
        t = re.sub(r"[:\-\s_]+", ":", t)
        return t.strip(":")

    def _extract_references_from_node(
        self, node: ArtifactNode, bib_map: Dict[str, Dict]
    ) -> List[Reference]:
        """Extract all references (\\ref-style and \\cite-style) from artifact and proof content."""
        references = []
        parts = []
        if node.content:
            parts.append(node.content)
        if node.proof:
            parts.append(node.proof)
        full_content = "\n\n".join(parts)

        if not full_content:
            logger.warning(
                f"Node {node.id} has no content or proof to extract references from."
            )
            return references

        explicit_pattern = re.compile(
            r"\\(?P<ref_cmd>[cC]ref|[vV]ref|[Aa]utoref|ref|eqref)\s*\{(?P<ref_keys>[^}]+)\}|"
            r"\\(?P<cite_cmd>cite[pt]?\*?)(?:\[(?P<cite_note>[^\]]*)\])?\{(?P<cite_keys>[^}]+)\}"
        )

        found_cite_keys = set()

        for match in explicit_pattern.finditer(full_content):
            context_start = max(0, match.start() - 50)
            context_end = min(len(full_content), match.end() + 50)
            context = full_content[context_start:context_end].replace("\n", " ").strip()

            # Case 1: It was an internal reference (\ref, \Cref, etc.)
            if match.group("ref_cmd"):
                for key in match.group("ref_keys").split(","):
                    references.append(
                        Reference(
                            target_id=key.strip(),
                            reference_type=ReferenceType.INTERNAL,
                            context=context,
                        )
                    )

            # Case 2: It was an external citation (\cite, \citep, etc.)
            elif match.group("cite_cmd"):
                note = match.group("cite_note")
                for key in match.group("cite_keys").split(","):
                    key = key.strip()
                    found_cite_keys.add(key)

                    if key in bib_map:
                        bib_entry = bib_map[key]
                        references.append(
                            Reference(
                                target_id=key,
                                reference_type=ReferenceType.EXTERNAL,
                                context=context,
                                full_reference=bib_entry["full_reference"],
                                arxiv_id=bib_entry["arxiv_id"],
                                note=note.strip() if note else None,
                            )
                        )
                    else:
                        logger.warning(
                            f"Unresolved citation: Found cite key '{key}' in the text, "
                            f"but it was not found in the parsed bibliography."
                        )
                        references.append(
                            Reference(
                                target_id=key,
                                reference_type=ReferenceType.EXTERNAL,
                                context=context,
                                full_reference=f"UNRESOLVED: Citation key '{key}' not found in bibliography.",
                                arxiv_id=None,
                                note=note.strip() if note else None,
                            )
                        )

        # Manual citation scan (fast path): look for bracket/paren spans like
        #   [Rou01, Thm 2] or (Doe01, p. 3)
        # and extract *multiple* cite keys from the same span.
        # We avoid O(nodes * bib_keys) scanning.
        for span in self._iter_bracket_spans(full_content, max_span_chars=500):
            inner = span[1:-1]  # strip [] or ()
            # Find candidate tokens that look like bib keys.
            # Keep letters/numbers and common separators.
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9:_\-\.]*", inner)
            matched_keys = [
                t for t in tokens if t in bib_map and t not in found_cite_keys
            ]
            if not matched_keys:
                continue

            for cite_key in matched_keys:
                found_cite_keys.add(cite_key)
                bib_entry = bib_map[cite_key]

                # Build note by removing *all* matched keys from the inner span.
                note_text = inner
                for k in matched_keys:
                    note_text = re.sub(r"\b" + re.escape(k) + r"\b", "", note_text)
                note = note_text.strip(" ,") or None

                # Context window around the first occurrence of this key within the span.
                # (Approx: use the full span location in the concatenated content.)
                # We don't keep exact offsets for each token; the span itself is good enough.
                context = span

                if not any(
                    r.target_id == cite_key and r.note == note for r in references
                ):
                    references.append(
                        Reference(
                            target_id=cite_key,
                            reference_type=ReferenceType.EXTERNAL,
                            context=context,
                            full_reference=bib_entry["full_reference"],
                            arxiv_id=bib_entry["arxiv_id"],
                            note=note,
                        )
                    )

        return references

    def _iter_bracket_spans(self, text: str, *, max_span_chars: int) -> List[str]:
        """Yield non-nested bracket/paren spans like '[...]' or '(...)'.

        This is intentionally conservative (no nesting) and bounded by max_span_chars
        to avoid pathological regex backtracking or huge spans.
        """

        spans: List[str] = []
        for open_ch, close_ch in (("[", "]"), ("(", ")")):
            start = 0
            while True:
                i = text.find(open_ch, start)
                if i == -1:
                    break
                j = text.find(close_ch, i + 1)
                if j == -1:
                    break
                if j - i + 1 <= max_span_chars:
                    spans.append(text[i : j + 1])
                start = j + 1
        return spans

    def _create_graph_links(
        self, nodes: List[ArtifactNode], label_to_node_id_map: Dict[str, str]
    ) -> Tuple[List[Edge], List[ArtifactNode]]:
        edges: List[Edge] = []
        new_external_nodes: List[ArtifactNode] = []
        created_external_nodes_map: Dict[str, str] = {}
        # Build a normalized lookup for artifact labels to tolerate minor formatting differences.
        normalized_label_map: Dict[str, str] = {}
        normalized_collisions: set = set()
        for lbl, nid in label_to_node_id_map.items():
            norm = self._normalize_label(lbl)
            if norm in normalized_label_map:
                normalized_collisions.add(norm)
            else:
                normalized_label_map[norm] = nid

        for source_node in nodes:
            if source_node.is_external:
                continue

            for ref in source_node.references:
                if ref.reference_type == ReferenceType.INTERNAL:
                    target_label = ref.target_id
                    if target_label in label_to_node_id_map:
                        target_node_id = label_to_node_id_map[target_label]
                        if target_node_id != source_node.id:
                            edges.append(
                                Edge(
                                    source_id=source_node.id,
                                    target_id=target_node_id,
                                    context=ref.context,
                                    reference_type=ReferenceType.INTERNAL,
                                )
                            )
                    else:
                        # Try normalized lookup before deciding it's missing.
                        norm = self._normalize_label(target_label)
                        if (
                            norm
                            and norm in normalized_label_map
                            and norm not in normalized_collisions
                        ):
                            target_node_id = normalized_label_map[norm]
                            if target_node_id != source_node.id:
                                edges.append(
                                    Edge(
                                        source_id=source_node.id,
                                        target_id=target_node_id,
                                        context=ref.context,
                                        reference_type=ReferenceType.INTERNAL,
                                    )
                                )
                        else:
                            # If the label exists somewhere in the document, it is likely a non-artifact label
                            # (e.g., equation, section). Since we only build artifact graphs, ignore it quietly.
                            if (
                                hasattr(self, "all_labels_in_doc")
                                and target_label in self.all_labels_in_doc
                            ):
                                logger.debug(
                                    f"Ignoring non-artifact internal label '{target_label}' referenced by '{source_node.id}'."
                                )
                            else:
                                # True dangling internal reference
                                logger.warning(
                                    f"Dangling internal reference: Node '{source_node.id}' refers to missing label '{target_label}'."
                                )

                elif ref.reference_type == ReferenceType.EXTERNAL:
                    target_key = ref.target_id
                    if target_key in created_external_nodes_map:
                        external_node_id = created_external_nodes_map[target_key]
                    else:
                        external_node_id = f"external_{target_key}"

                        # Use bibliography metadata (when available) to give external
                        # references a meaningful preview. This avoids generic
                        # "Unknown" nodes in downstream visualizations.
                        if ref.full_reference:
                            node_content = ref.full_reference
                        else:
                            node_content = (
                                f"External reference {target_key} "
                                "(no bibliography entry found in project)."
                            )

                        external_node = ArtifactNode(
                            id=external_node_id,
                            label=target_key,
                            type=ArtifactType.EXTERNAL_REFERENCE,
                            is_external=True,
                            content=node_content,
                        )
                        new_external_nodes.append(external_node)
                        created_external_nodes_map[target_key] = external_node_id

                    edges.append(
                        Edge(
                            source_id=source_node.id,
                            target_id=external_node_id,
                            context=ref.context,
                            reference_type=ReferenceType.EXTERNAL,
                        )
                    )
        return edges, new_external_nodes
