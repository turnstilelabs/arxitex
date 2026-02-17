from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ArxivExtractorError(Exception):
    """Custom exception for all arxiv-extractor related errors."""

    pass


class ArtifactType(Enum):
    """Types of mathematical artifacts that can be extracted."""

    THEOREM = "theorem"
    LEMMA = "lemma"
    DEFINITION = "definition"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    EXAMPLE = "example"
    REMARK = "remark"
    CONJECTURE = "conjecture"
    CLAIM = "claim"
    FACT = "fact"
    OBSERVATION = "observation"
    EXTERNAL_REFERENCE = "external_reference"
    UNKNOWN = "unknown"


class ReferenceType(Enum):
    """Types of references between artifacts."""

    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class Position:
    """Represents a position in the source document."""

    line_start: int
    line_end: Optional[int] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        """Converts the Position object to a JSON-serializable dictionary."""
        return {
            "line_start": self.line_start,
            "line_end": self.line_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
        }


@dataclass
class Reference:
    """Represents a reference from one artifact to another."""

    target_id: str
    reference_type: ReferenceType
    context: Optional[str] = None  # Surrounding text for context
    full_reference: Optional[str] = None  # The full text from the bibliography entry.
    arxiv_id: Optional[str] = None  # The arXiv ID, if found.
    note: Optional[str] = None  # e.g., "Theorem 3.1" from \cite[Theorem 3.1]{...}

    def to_dict(self) -> Dict:
        return {
            "target_id": self.target_id,
            "reference_type": self.reference_type.value,
            "context": self.context,
            "full_reference": self.full_reference,
            "arxiv_id": self.arxiv_id,
            "note": self.note,
        }


@dataclass
class ArtifactNode:
    """
    Represents a mathematical artifact in the document graph.
    """

    id: str
    type: ArtifactType
    content: str = ""
    label: Optional[str] = None  # LaTeX label (\label{...}) for cross-references
    position: Position = field(default_factory=lambda: Position(line_start=0))
    references: List[Reference] = field(default_factory=list)
    is_external: bool = False
    proof: Optional[str] = None
    prerequisite_defs: Dict[str, str] = field(default_factory=dict)
    semantic_tag: Optional[str] = None
    source_file: Optional[str] = None
    source_line_start: Optional[int] = None
    pdf_page: Optional[int] = None
    pdf_label: Optional[str] = None
    pdf_label_type: Optional[str] = None
    pdf_label_number: Optional[str] = None

    @property
    def content_preview(self) -> str:
        """
        Generates a clean, MathJax-compatible preview of the content.
        """
        if not self.content:
            return ""

        # 1. Escape backticks, which can break JS template literals.
        # 2. Replace newlines with <br> for HTML rendering.
        # 3. Escape backslashes for JSON compatibility.
        clean_content = self.content.replace("`", "\\`")
        clean_content = clean_content.replace("\n", "<br>")
        # A double escape is often needed for JSON -> JS pipeline.
        clean_content = clean_content.replace("\\", "\\\\")

        # Truncate to preview length
        max_length = 250
        if len(clean_content) <= max_length:
            return clean_content

        truncated = clean_content[:max_length]
        last_space = truncated.rfind(" ")
        if last_space != -1:
            return truncated[:last_space] + "..."

        return truncated + "..."

    @property
    def prerequisites_preview(self) -> str:
        """Generates a clean, HTML-formatted list of prerequisite definitions."""
        if not self.prerequisite_defs:
            return ""

        items = []
        for term, definition in self.prerequisite_defs.items():
            # Sanitize each part for HTML/JS
            clean_term = term.replace("`", "\\`").replace("\\", "\\\\")
            clean_def = (
                definition.replace("`", "\\`")
                .replace("\n", "<br>")
                .replace("\\", "\\\\")
            )
            items.append(f"<b>{clean_term}</b>: {clean_def}")

        return "<br><br>".join(items)

    @property
    def display_name(self) -> str:
        """
        Generate a display name for the artifact.
        Uses the type capitalized (e.g., "Theorem", "Definition").
        For external references, try to surface a more human-friendly label.
        """
        if self.is_external and self.type == ArtifactType.EXTERNAL_REFERENCE:
            # Prefer an explicit label (usually the BibTeX key), falling back to id.
            base = self.label or self.id
            return f"Reference: {base}"

        # TODO: consider using label if available for internal artifacts as well.
        return self.type.value.capitalize()

    @property
    def raw_content(self) -> str:
        """
        Alias for content to maintain compatibility with dependency checker.
        """
        return self.content

    def to_dict(self) -> Dict:
        """
        Convert ArtifactNode to a fully JSON-serializable dictionary.
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "content_preview": self.content_preview,
            "prerequisites_preview": self.prerequisites_preview,
            "display_name": self.display_name,
            "label": self.label,
            "position": self.position.to_dict(),
            "references": [ref.to_dict() for ref in self.references],
            "proof": self.proof,
            "semantic_tag": self.semantic_tag,
            "source_file": self.source_file,
            "source_line_start": self.source_line_start,
            "pdf_page": self.pdf_page,
            "pdf_label": self.pdf_label,
            "pdf_label_type": self.pdf_label_type,
            "pdf_label_number": self.pdf_label_number,
        }


class DependencyType(str, Enum):
    """Simplified dependency taxonomy between two artifacts.

    Internal (extractor) convention:
        - `source_id` is the dependent.
        - `target_id` is the prerequisite.

    Note: the frontend visualization prefers arrows to point prerequisite → dependent.
    We normalize that direction at serialization time in `Edge.to_dict()`.
    """

    USED_IN = "used_in"
    """Target is used as a prerequisite (definition/result/concept) by Source."""

    GENERALIZES = "generalizes"
    """Source generalizes Target (Target is a special case of Source)."""


@dataclass
class Edge:
    """
    Represents a directed relationship between two artifacts.
    """

    source_id: str
    target_id: str
    dependency_type: Optional[DependencyType] = None
    dependency: Optional[str] = None
    context: Optional[str] = None
    reference_type: Optional[ReferenceType] = None

    def to_dict(self) -> Dict:
        """
        Converts the Edge object to a fully JSON-serializable dictionary.
        """
        dep_type_str = self.dependency_type.value if self.dependency_type else None
        ref_type_str = self.reference_type.value if self.reference_type else None

        # Frontend/UI semantics:
        # The UI renders arrows from `source` -> `target`.
        # For `used_in`, we want the arrow to mean:
        #   prerequisite -> dependent
        #
        # Internally, `used_in` edges are stored as:
        #   dependent (source_id) -> prerequisite (target_id)
        # so we swap the serialized direction for `used_in`.
        source_id = self.source_id
        target_id = self.target_id
        if self.dependency_type == DependencyType.USED_IN:
            source_id, target_id = target_id, source_id

        return {
            "source": source_id,
            "target": target_id,
            "context": self.context,
            "reference_type": ref_type_str,
            "dependency_type": dep_type_str,
            "dependency": self.dependency,
            "type": dep_type_str or ref_type_str or "generic_dependency",
        }


@dataclass
class DocumentGraph:
    """
    Container for the complete document graph.
    """

    nodes: List[ArtifactNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    source_file: Optional[str] = None

    def get_node_by_id(self, node_id: str) -> Optional[ArtifactNode]:
        """Get a node by its ID."""
        return next((node for node in self.nodes if node.id == node_id), None)

    def get_node_by_label(self, label: str) -> Optional[ArtifactNode]:
        """Get a node by its LaTeX label."""
        return next((node for node in self.nodes if node.label == label), None)

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges if edge.source_id == node_id]

    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges pointing to a node."""
        return [edge for edge in self.edges if edge.target_id == node_id]

    def find_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        """
        Finds and returns a specific edge between two nodes, if it exists.
        """
        return next(
            (
                edge
                for edge in self.edges
                if edge.source_id == source_id and edge.target_id == target_id
            ),
            None,
        )

    def add_node(self, node: ArtifactNode) -> None:
        """Add a node to the graph."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph, avoiding duplicates."""
        if not any(
            e.source_id == edge.source_id and e.target_id == edge.target_id
            for e in self.edges
        ):
            self.edges.append(edge)

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the graph."""
        return {"total_nodes": len(self.nodes), "total_edges": len(self.edges)}

    def to_dict(self, arxiv_id: str, extractor_mode: str | None = None) -> Dict:
        """
        Serializes the entire graph, including nodes and edges, into a
        JSON-serializable dictionary for output.
        """
        serialized_nodes = [node.to_dict() for node in self.nodes]
        serialized_edges = [edge.to_dict() for edge in self.edges]

        return {
            "arxiv_id": arxiv_id,
            "extractor_mode": extractor_mode or "unspecified",
            "stats": {"node_count": len(self.nodes), "edge_count": len(self.edges)},
            "nodes": serialized_nodes,
            "edges": serialized_edges,
        }
