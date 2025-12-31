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


def test_build_graph_handles_custom_theorem_like_envs(monkeypatch):
    """Custom theorem/definition env names (thm, Def, etc.) should be normalized.

    This asserts that our ENV_NAME_ALIASES mapping in BaseGraphBuilder is wired
    correctly and that downstream types are the canonical ones (theorem, definition).
    """

    latex = r"""
\section{Intro}

% Custom theorem-like environments that should still be treated as theorems/definitions
\begin{thm}\label{thm:alias-one}
First custom theorem.
\end{thm}

\begin{Thm}\label{thm:alias-two}
Second custom theorem (capitalized env name).
\end{Thm}

\begin{Def}\label{def:alias-one}
Custom definition env.
\end{Def}
"""

    monkeypatch.setattr(
        "arxitex.extractor.graph_building.base_builder.read_and_combine_tex_files",
        lambda project_dir: latex,
    )

    builder = BaseGraphBuilder()
    graph = builder.build_graph(project_dir=Path("/tmp/fake3"))

    # The custom theorem envs should be normalized to canonical type "theorem"
    thm_nodes = [n for n in graph.nodes if n.type.value == "theorem"]
    assert {n.label for n in thm_nodes} == {"thm:alias-one", "thm:alias-two"}

    # The custom definition env should be normalized to "definition"
    def_nodes = [n for n in graph.nodes if n.type.value == "definition"]
    assert {n.label for n in def_nodes} == {"def:alias-one"}


def test_build_graph_uses_newtheorem_scanner_for_shared_counter_envs(monkeypatch):
    """Envs defined via \newtheorem (thm1, lem1, etc.) should be recognized.

    This specifically tests the pattern:

        \newtheorem{thm1}{Theorem}[section]
        \newtheorem{lem1}[thm1]{Lemma}
        \newtheorem{rem1}[thm1]{Remark}
        \newtheorem{def1}[thm1]{Definition}
        \newtheorem{cor1}[thm1]{Corollary}
        \newtheorem{defn1}[thm1]{Definition}
        \newtheorem{prop1}[thm1]{Proposition}
        \newtheorem{ex1}[thm1]{Example}
    """

    latex = r"""
\section{Intro}

\newtheorem{thm1}{Theorem}[section]
\newtheorem{lem1}[thm1]{Lemma}
\newtheorem{rem1}[thm1]{Remark}
\newtheorem{def1}[thm1]{Definition}
\newtheorem{cor1}[thm1]{Corollary}
\newtheorem{defn1}[thm1]{Definition}
\newtheorem{prop1}[thm1]{Proposition}
\newtheorem{ex1}[thm1]{Example}

\begin{thm1}\label{thm:first}
First theorem.
\end{thm1}

\begin{lem1}\label{lem:first}
First lemma.
\end{lem1}

\begin{rem1}\label{rem:first}
First remark.
\end{rem1}

\begin{def1}\label{def:first}
First definition.
\end{def1}

\begin{cor1}\label{cor:first}
First corollary.
\end{cor1}

\begin{defn1}\label{defn:first}
Second definition env style.
\end{defn1}

\begin{prop1}\label{prop:first}
First proposition.
\end{prop1}

\begin{ex1}\label{ex:first}
First example.
\end{ex1}
"""

    monkeypatch.setattr(
        "arxitex.extractor.graph_building.base_builder.read_and_combine_tex_files",
        lambda project_dir: latex,
    )

    builder = BaseGraphBuilder()
    graph = builder.build_graph(project_dir=Path("/tmp/fake_newthm"))

    types_by_label = {n.label: n.type.value for n in graph.nodes}

    assert types_by_label["thm:first"] == "theorem"
    assert types_by_label["lem:first"] == "lemma"
    assert types_by_label["rem:first"] == "remark"
    assert types_by_label["def:first"] == "definition"
    assert types_by_label["cor:first"] == "corollary"
    assert types_by_label["defn:first"] == "definition"
    assert types_by_label["prop:first"] == "proposition"
    assert types_by_label["ex:first"] == "example"


def test_build_graph_from_content_position_matches_naive(monkeypatch):
    """The optimized position calculator should match the naive implementation."""

    content = "a\n" * 5 + "X" * 10 + "\n" + "b\n" * 3

    builder = BaseGraphBuilder()

    # We need project_dir for bibliography resolution; point to empty temp.
    class DummyDir(Path):
        _flavour = type(Path())._flavour

    proj = DummyDir("/tmp")

    # Make resolver not depend on actual files.
    monkeypatch.setattr(
        "arxitex.extractor.graph_building.reference_resolver.ReferenceResolver._find_and_parse_bibliography",
        lambda self, project_dir: {},
    )

    builder.build_graph_from_content(content=content, project_dir=proj)

    def naive_pos(start: int, end: int):
        line_start = content[:start].count("\n") + 1
        line_end = content[:end].count("\n") + 1
        last_nl_start = content.rfind("\n", 0, start)
        col_start = start + 1 if last_nl_start == -1 else start - last_nl_start
        last_nl_end = content.rfind("\n", 0, end)
        col_end = end + 1 if last_nl_end == -1 else end - last_nl_end
        return line_start, line_end, col_start, col_end

    for start, end in [(0, 1), (2, 3), (10, 12), (len(content) - 5, len(content) - 1)]:
        p = builder._calculate_position(start, end)
        exp = naive_pos(start, end)
        assert (p.line_start, p.line_end, p.col_start, p.col_end) == exp
