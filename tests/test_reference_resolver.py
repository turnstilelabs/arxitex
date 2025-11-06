from pathlib import Path

from arxitex.extractor.graph_building.reference_resolver import ReferenceResolver
from arxitex.extractor.models import ArtifactNode, ArtifactType, Position, ReferenceType


def test_resolve_inline_and_citation(tmp_path):
    # Create a fake project dir and a .bbl file
    proj = Path(tmp_path)
    bbl = proj / "refs.bbl"
    bbl.write_text(
        r"""
    \begin{thebibliography}{9}
    \bibitem{Doe01} J. Doe, Some paper, arXiv:1234.5678
    \bibitem[OptKey]{OptKey} Author, Another paper
    \end{thebibliography}
    """,
        encoding="utf-8",
    )

    # Document content (contains embedded bibliography already, but having .bbl is fine)
    content = "Some intro text.\n\nSee \\ref{thm:a} and also \\cite{Doe01} in the text."

    # Nodes: one referenced target with label 'thm:a' and one node that references it.
    target = ArtifactNode(
        id="target1",
        type=ArtifactType.THEOREM,
        content="Theorem content",
        label="thm:a",
        position=Position(line_start=1),
    )
    referrer = ArtifactNode(
        id="n1",
        type=ArtifactType.REMARK,
        content="See \\ref{thm:a} and \\cite{Doe01}",
        label=None,
        position=Position(line_start=5),
    )

    resolver = ReferenceResolver(content)
    edges, external_nodes = resolver.resolve_all_references(
        proj, [target, referrer], {"thm:a": "target1"}
    )

    # There should be an internal reference edge from referrer -> target1
    internal_edges = [e for e in edges if e.reference_type == ReferenceType.INTERNAL]
    assert any(e.source_id == "n1" and e.target_id == "target1" for e in internal_edges)

    # There should be an external edge for the citation (arXiv-based)
    external_edges = [e for e in edges if e.reference_type == ReferenceType.EXTERNAL]
    assert any(e.source_id == "n1" for e in external_edges)

    # External nodes should include at least the cited key
    assert any(n.is_external for n in external_nodes)


def test_unresolved_citation_creates_placeholder(tmp_path):
    proj = Path(tmp_path)
    # Use an explicit \\cite command so the resolver's explicit-pattern picks it up even without .bbl
    content = r"Text mentioning \cite{Rou01} but no .bbl present."
    node = ArtifactNode(
        id="n2",
        type=ArtifactType.REMARK,
        content=r"See \cite{Rou01}",
        position=Position(line_start=1),
    )
    resolver = ReferenceResolver(content)
    edges, external_nodes = resolver.resolve_all_references(proj, [node], {})

    # Should create an external node for the unresolved citation and an external edge
    assert len(external_nodes) == 1
    assert any(e.reference_type == ReferenceType.EXTERNAL for e in edges)
