import asyncio
from pathlib import Path

from arxitex.extractor.graph_building.graph_enhancer import GraphEnhancer
from arxitex.tex.dialect import TeXDialect, detect_tex_dialect
from arxitex.tex.normalize import normalize_tex


def test_detect_tex_dialect_latex():
    content = r"""\documentclass{article}
\begin{document}
Hi
\end{document}
"""
    assert detect_tex_dialect(content) == TeXDialect.LATEX


def test_detect_tex_dialect_ams_tex():
    content = r"""% AMS-TeX style
\input amstex
\proclaim{Theorem 1.} Foo.\endproclaim
\demo Proof.\enddemo
"""
    assert detect_tex_dialect(content) == TeXDialect.AMS_TEX


def test_normalize_ams_tex_proclaim_and_demo():
    content = r"""\proclaim{Theorem 1.}
Let $X$ be a set.
\demo Proof goes here.\enddemo
\endproclaim
"""
    res = normalize_tex(content, TeXDialect.AMS_TEX)
    assert res.changed is True
    assert "\\begin{theorem}" in res.content
    assert "\\end{theorem}" in res.content
    assert "\\begin{proof}" in res.content
    assert "\\end{proof}" in res.content


def test_graph_enhancer_extracts_theorem_from_ams_tex(monkeypatch, tmp_path):
    """End-to-end-ish regression: GraphEnhancer should extract nodes after normalization."""

    from arxitex.extractor.graph_building import graph_enhancer as ge_mod

    ams_tex = r"""
\proclaim{Theorem 1.}\label{thm:one}
Let $X$ be a set.
\demo Proof goes here.\enddemo
\endproclaim
"""

    monkeypatch.setattr(
        ge_mod, "read_and_combine_tex_files", lambda project_dir: ams_tex
    )

    ge = GraphEnhancer()
    graph, bank, artifact_to_terms_map, latex_macros = asyncio.run(
        ge.build_graph(
            project_dir=Path(tmp_path),
            source_file="ams",
            infer_dependencies=False,
            enrich_content=False,
        )
    )

    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.type.value == "theorem"
    assert node.proof is not None
    assert "Proof goes here" in node.proof
