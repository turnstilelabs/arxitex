from pathlib import Path

from arxitex.extractor.graph_building.base_builder import BaseGraphBuilder


def test_build_graph_parses_environments_and_attaches_proof(monkeypatch):
    latex = r"""
\section{Intro}

\begin{theorem}\label{thm:one}
This is theorem statement.
\end{theorem}

\begin{proof}
Proof content here.
\end{proof}

\begin{definition}\label{def:one}
Definition content.
\end{definition}
"""

    # Monkeypatch read_and_combine_tex_files used inside BaseGraphBuilder
    monkeypatch.setattr(
        "arxitex.extractor.graph_building.base_builder.read_and_combine_tex_files",
        lambda project_dir: latex,
    )

    builder = BaseGraphBuilder()
    graph = builder.build_graph(project_dir=Path("/tmp/fake"))

    # Should find two artifact nodes (theorem and definition) plus no extra nodes
    node_types = {n.type.value for n in graph.nodes}
    assert "theorem" in node_types
    assert "definition" in node_types

    # Find theorem node and check label and proof attached
    thm_nodes = [n for n in graph.nodes if n.type.value == "theorem"]
    assert len(thm_nodes) == 1
    thm = thm_nodes[0]
    assert thm.label == "thm:one"
    # Proof should have been attached by ProofLinker via proximity
    assert thm.proof is not None
    assert "Proof content here" in thm.proof

    # Definition node present with label
    def_nodes = [n for n in graph.nodes if n.type.value == "definition"]
    assert len(def_nodes) == 1
    d = def_nodes[0]
    assert d.label == "def:one"


def test_build_graph_handles_broken_environment(monkeypatch):
    # Start an environment but never close it; builder should skip it without raising
    latex = r"""
\begin{lemma}\label{lem:bad}
This lemma has no end tag...
% missing \end{lemma}
\begin{theorem}\label{thm:ok}
A well-formed theorem.
\end{theorem}
"""
    monkeypatch.setattr(
        "arxitex.extractor.graph_building.base_builder.read_and_combine_tex_files",
        lambda project_dir: latex,
    )

    builder = BaseGraphBuilder()
    graph = builder.build_graph(project_dir=Path("/tmp/fake2"))

    # Should still detect the well-formed theorem
    assert any(n.label == "thm:ok" for n in graph.nodes)
    # The broken lemma should not produce a node (no crash)
    assert not any(getattr(n, "label", "") == "lem:bad" for n in graph.nodes)
