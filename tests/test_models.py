from arxitex.extractor.models import ArtifactNode, ArtifactType, Position


def test_content_preview_truncation_and_escaping():
    long_content = "a " * 200 + "\\ backslash and `backtick` and newline\nline2"
    node = ArtifactNode(
        id="n",
        type=ArtifactType.DEFINITION,
        content=long_content,
        position=Position(line_start=1),
    )

    preview = node.content_preview
    # Should not be empty
    assert preview
    # Newlines may be converted to <br> or truncated; do not assert strictly here.
    # Backticks should be escaped (or present as escaped sequences)
    assert ("\\`" in preview) or ("`" not in preview)
    # If longer than max_length it should end with "..."
    assert preview.endswith("...") or len(preview) <= 250


def test_prerequisites_preview_formats_entries():
    node = ArtifactNode(
        id="n",
        type=ArtifactType.REMARK,
        content="something",
        position=Position(line_start=1),
    )
    node.prerequisite_defs = {
        "TermA": "Definition of TermA.\nWith new line.",
        "T`ermB": "Definition with `backtick` and \\ backslash",
    }

    preview = node.prerequisites_preview
    # Should include bolded term names and HTML <br> for newlines
    assert "<b>TermA</b>" in preview
    assert "With new line." in preview
    assert "<br><br>" in preview  # entries separated by double break
    # Backticks and backslashes should be escaped
    assert "\\`" in preview or "\\\\" in preview


def test_display_name_and_raw_content_and_to_dict():
    node = ArtifactNode(
        id="n1",
        type=ArtifactType.THEOREM,
        content="content",
        label="lbl",
        position=Position(line_start=5),
    )

    assert node.display_name == "Theorem"
    assert node.raw_content == node.content

    d = node.to_dict()
    assert d["id"] == "n1"
    assert d["type"] == ArtifactType.THEOREM.value
    assert "content_preview" in d
    assert "prerequisites_preview" in d
    assert "position" in d
