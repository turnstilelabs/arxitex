from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

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
            "col_end": self.col_end
        }

@dataclass
class Reference:
    """Represents a reference from one artifact to another."""
    target_id: str
    reference_type: ReferenceType
    context: Optional[str] = None  # Surrounding text for context
    full_reference: Optional[str] = None # The full text from the bibliography entry.
    arxiv_id: Optional[str] = None # The arXiv ID, if found.
    note: Optional[str] = None # e.g., "Theorem 3.1" from \cite[Theorem 3.1]{...}

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
class Citation:
    """Represents a citation to an external work, including contextual notes."""
    cite_key: str
    full_reference: str
    arxiv_id: Optional[str] = None
    note: Optional[str] = None # Captures text from e.g., \cite[Theorem 3.1]{...}

    def to_dict(self) -> Dict:
        return {
            "cite_key": self.cite_key,
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
        clean_content = self.content.replace('`', '\\`')
        clean_content = clean_content.replace('\n', '<br>')
        # A double escape is often needed for JSON -> JS pipeline.
        clean_content = clean_content.replace('\\', '\\\\')
            
        # Truncate to preview length
        max_length = 250
        if len(clean_content) <= max_length:
            return clean_content
        
        truncated = clean_content[:max_length]
        last_space = truncated.rfind(' ')
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
            clean_term = term.replace('`', '\\`').replace('\\', '\\\\')
            clean_def = definition.replace('`', '\\`').replace('\n', '<br>').replace('\\', '\\\\')
            items.append(f"<b>{clean_term}</b>: {clean_def}")
            
        return "<br><br>".join(items)

    @property
    def display_name(self) -> str:
        """
        Generate a display name for the artifact.
        Uses the type capitalized (e.g., "Theorem", "Definition").
        """
        #TODO: consider using label if available
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
            "proof": self.proof
        }

class DependencyType(str, Enum):
    """
    Defines the types of logical or structural dependencies between two artifacts in a document.
    """
    
    # --- Logical Dependencies ---
    USES_RESULT = "uses_result"
    """The source artifact's proof relies on a theorem, lemma, or proposition from the target."""
    
    USES_DEFINITION = "uses_definition"
    """The source artifact uses a term, notation, or concept formally defined in the target."""

    PROVES = "proves"
    """The source artifact is the formal proof of the statement made in the target."""

    # --- Illustrative Dependencies ---
    PROVIDES_EXAMPLE = "provides_example"
    """The source artifact is a concrete example illustrating the concept from the target."""

    PROVIDES_REMARK = "provides_remark"
    """The source artifact is a remark that provides context or commentary on the target."""

    # --- Hierarchical Dependencies ---
    IS_COROLLARY_OF = "is_corollary_of"
    """The source artifact is a direct and immediate consequence of the target theorem."""

    IS_SPECIAL_CASE_OF = "is_special_case_of"
    """The source artifact is a more specific version of a general result in the target."""
    
    IS_GENERALIZATION_OF = "is_generalization_of"
    """The source artifact presents a result that extends a more specific result from the target."""
    
@dataclass
class Edge:
    """
    Represents a directed relationship between two artifacts.
    """
    source_id: str
    target_id: str
    dependency_type: DependencyType = None
    dependency: Optional[str] = None
    context: Optional[str] = None
    reference_type: Optional[ReferenceType] = None

    def to_dict(self) -> Dict:
        """
        Converts the Edge object to a fully JSON-serializable dictionary.
        """
        dep_type_str = self.dependency_type.value if self.dependency_type else None
        ref_type_str = self.reference_type.value if self.reference_type else None

        return {
            "source": self.source_id,
            "target": self.target_id,
            "context": self.context,
            "reference_type": ref_type_str,
            "dependency_type": dep_type_str,
            "dependency": self.dependency,
            "type": dep_type_str or ref_type_str or "generic_dependency"
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
        return next((
            edge for edge in self.edges 
            if edge.source_id == source_id and edge.target_id == target_id
        ), None)
    
    def add_node(self, node: ArtifactNode) -> None:
        """Add a node to the graph."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph, avoiding duplicates."""
        if not any(e.source_id == edge.source_id and e.target_id == edge.target_id
                  for e in self.edges):
            self.edges.append(edge)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the graph."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }
    
    def to_dict(self, arxiv_id: str, extractor_mode: str) -> Dict:
        """
        Serializes the entire graph, including nodes and edges, into a
        JSON-serializable dictionary for output.
        """
        serialized_nodes = [node.to_dict() for node in self.nodes]
        serialized_edges = [edge.to_dict() for edge in self.edges]

        return {
            "arxiv_id": arxiv_id,
            "extractor_mode": extractor_mode,
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges)
            },
            "nodes": serialized_nodes,
            "edges": serialized_edges
        }
