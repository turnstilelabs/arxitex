from arxitex.tools.retrieval.colgrep_engine import parse_colgrep_output


def test_parse_colgrep_output_json() -> None:
    stdout = '{"path": "/tmp/a.md", "score": 0.9, "text": "hello"}'
    hits = parse_colgrep_output(stdout)
    assert hits
    assert hits[0].path == "/tmp/a.md"
    assert hits[0].score == 0.9


def test_parse_colgrep_output_text() -> None:
    stdout = "/tmp/a.md:12:3 score=0.5 some text"
    hits = parse_colgrep_output(stdout)
    assert hits
    assert hits[0].path == "/tmp/a.md"
    assert hits[0].score == 0.5
