from pathlib import Path

from arxitex.extractor import pdf_labels as pl
from arxitex.extractor.models import ArtifactNode, ArtifactType, Position


def test_strip_tex_to_anchor():
    text = r"\begin{thm} The absolute Galois groups of $K$ and $K^\flat$ are canonically isomorphic.\end{thm}"
    anchor = pl._strip_tex_to_anchor(text)
    assert "The absolute Galois groups" in anchor
    assert "$" not in anchor


def test_find_label_in_lines():
    lines = [
        ("Some intro text.", None),
        (
            "Theorem 1.1. The absolute Galois groups of K and K^flat are canonically isomorphic.",
            None,
        ),
        ("More text.", None),
    ]
    anchor = "the absolute galois groups of k and k flat are canonically isomorphic"
    found = pl._find_label_in_lines(lines, anchor, "Theorem")
    assert found == ("Theorem", "1.1")


def test_annotate_nodes_with_pdf_labels(monkeypatch, tmp_path):
    tex_root = tmp_path / "tex"
    tex_root.mkdir()
    tex_file = tex_root / "main.tex"
    tex_file.write_text(
        "\\begin{thm}\nThe absolute Galois groups of K and K^flat are canonically isomorphic.\n\\end{thm}\n",
        encoding="utf-8",
    )

    node = ArtifactNode(
        id="n1",
        type=ArtifactType.THEOREM,
        content="The absolute Galois groups of K and K^flat are canonically isomorphic.",
        position=Position(line_start=1),
    )

    def fake_load_pdf_text(_):
        return {
            2: [
                (
                    "Theorem 1.1. The absolute Galois groups of K and K^flat are canonically isomorphic.",
                    None,
                )
            ]
        }, False

    def fake_synctex_view(*_args, **_kwargs):
        return {"page": 2, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}

    monkeypatch.setattr(pl, "_load_pdf_text", fake_load_pdf_text)
    monkeypatch.setattr(pl, "_run_synctex_view", fake_synctex_view)

    updated = pl.annotate_nodes_with_pdf_labels(
        nodes=[node],
        tex_root=tex_root,
        pdf_path=Path("dummy.pdf"),
    )

    assert updated == 1
    assert node.pdf_label == "Theorem 1.1"
    assert node.pdf_label_number == "1.1"
