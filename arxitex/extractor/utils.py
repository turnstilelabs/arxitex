from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import re

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

#Sometimes proofs are separated from their statement -- how to deal with this?

class ReferenceType(Enum):
    """Types of references between artifacts."""
    INTERNAL = "internal"  # Reference within the same document
    EXTERNAL = "external"  # Reference to external document/citation

@dataclass
class Position:
    """Represents a position in the source document."""
    line_start: int
    line_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        """Converts the Position object to a JSON-serializable dictionary."""
        return {
            "line_start": self.line_start,
            "line_end": self.line_end
        }

@dataclass
class Reference:
    """Represents a reference from one artifact to another."""
    target_id: str
    reference_type: ReferenceType
    context: Optional[str] = None  # Surrounding text for context
    position: Optional[Position] = None  # Where the reference appears

    def to_dict(self) -> Dict:
        """Converts the Reference object to a JSON-serializable dictionary."""
        return {
            "target_id": self.target_id,
            "reference_type": self.reference_type.value,
            "context": self.context,
            "position": self.position.to_dict() if self.position else None,
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

    @property
    def content_preview(self) -> str:
        """
        Generate a preview of the content (first 150 characters).
        Strips LaTeX commands and formatting for cleaner display.
        """
        if not self.content:
            return ""
        
        # Remove common LaTeX commands and formatting
        clean_content = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', self.content)
        clean_content = re.sub(r'[{}$\\]', '', clean_content)  # Remove braces, dollar signs, backslashes
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Normalize whitespace
        clean_content = clean_content.strip()
        
        # Truncate to preview length
        max_length = 150
        if len(clean_content) <= max_length:
            return clean_content
        
        truncated = clean_content[:max_length]
        
        # Try to break at sentence end
        last_sentence = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_sentence > max_length * 0.7:  # If sentence break is reasonably close
            return truncated[:last_sentence + 1]
        
        # Otherwise break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:
            return truncated[:last_space] + "..."
        
        return truncated + "..."
    
    @property
    def display_name(self) -> str:
        """
        Generate a display name for the artifact.
        Uses the type capitalized (e.g., "Theorem", "Definition").
        """
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
            "display_name": self.display_name,
            "label": self.label,
            "position": self.position.to_dict(),
            "references": [ref.to_dict() for ref in self.references],
            "proof": self.proof
        }

class DependencyType(str, Enum):
    """Enumerates the specific types of relationships between artifacts."""
    PROVES = "PROVES"
    USES_DEFINITION = "USES_DEFINITION"
    BUILDS_UPON = "BUILDS_UPON"
    CITES = "CITES"
    PROVIDES_EXAMPLE_FOR = "PROVIDES_EXAMPLE_FOR"
    CONTRADICTS = "CONTRADICTS"
    
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

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "dependency_type": self.dependency_type.value if self.dependency_type else None,
            "dependency": self.dependency,
            "context": self.context,
            "reference_type": self.reference_type.value,
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
    
    def to_dict(self):
        return {
            "source_file": self.source_file,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges]
        }
