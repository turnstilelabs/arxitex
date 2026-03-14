from pathlib import Path

from arxitex.tools.retrieval.colgrep_chunks import build_chunks


def test_build_chunks_prev_paragraph(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir(parents=True)
    tex = """Intro paragraph line1.

Another paragraph line.

\\begin{lemma}
Lemma statement.
\\end{lemma}
"""
    (source_dir / "main.tex").write_text(tex, encoding="utf-8")

    graph = {
        "nodes": [
            {
                "id": "lemma:test",
                "type": "lemma",
                "content": "Lemma statement.",
                "position": {"line_start": 7},
                "pdf_label_number": "1.1",
            }
        ]
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(__import__("json").dumps(graph), encoding="utf-8")

    out_dir = tmp_path / "chunks"
    build_chunks(
        graph_path=graph_path,
        source_dir=source_dir,
        out_dir=out_dir,
        arxiv_id="1111.1111",
        title="Test Paper",
    )

    files = list(out_dir.glob("*.md"))
    assert files, "No chunk files written"
    content = files[0].read_text(encoding="utf-8")
    assert "PREV_PARAGRAPH" in content
    assert "Another paragraph line." in content

    manifest = (out_dir / "manifest.jsonl").read_text(encoding="utf-8")
    assert "lemma:test" in manifest
