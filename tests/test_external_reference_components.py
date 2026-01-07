from __future__ import annotations

import json
from pathlib import Path

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema
from arxitex.tools.external_reference_components import (
    extract_top_k_reference_components,
)


def _ensure_paper_and_artifact(
    conn, *, paper_id: str, external_artifact_id: str
) -> None:
    # external_reference_arxiv_matches has FK (paper_id, external_artifact_id)
    # -> artifacts(paper_id, artifact_id), and artifacts has FK -> papers.
    conn.execute(
        "INSERT OR IGNORE INTO papers(paper_id, title) VALUES (?, ?)",
        (paper_id, None),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO artifacts(
            paper_id, artifact_id, artifact_type, label, content_tex,
            proof_tex, line_start, line_end, col_start, col_end
        ) VALUES (?, ?, 'external_reference', NULL, '', NULL, 0, NULL, NULL, NULL)
        """,
        (paper_id, external_artifact_id),
    )


def _insert_match(
    conn, *, paper_id: str, external_artifact_id: str, matched_arxiv_id: str | None
) -> None:
    # Minimal insert; remaining columns are nullable but last_matched_at_utc is required.
    _ensure_paper_and_artifact(
        conn, paper_id=paper_id, external_artifact_id=external_artifact_id
    )
    conn.execute(
        """
        INSERT INTO external_reference_arxiv_matches (
            paper_id, external_artifact_id, matched_arxiv_id, match_method,
            extracted_title, extracted_authors_json,
            matched_title, matched_authors_json,
            title_score, author_overlap, arxiv_query,
            last_matched_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            external_artifact_id,
            matched_arxiv_id,
            "direct_regex" if matched_arxiv_id else "none",
            None,
            "[]",
            None,
            "[]",
            None,
            None,
            None,
            "2026-01-01T00:00:00+00:00",
        ),
    )


def test_extract_top_k_reference_components(tmp_path: Path):
    db_path = tmp_path / "t.db"
    ensure_schema(db_path)

    # Two components:
    # - Component A: a1 -- a2 -- a3 (2 edges)
    # - Component B: b1 -- b2 (1 edge)
    conn = connect(db_path)
    try:
        with conn:
            _insert_match(
                conn, paper_id="a1", external_artifact_id="r1", matched_arxiv_id="a2"
            )
            _insert_match(
                conn, paper_id="a2", external_artifact_id="r2", matched_arxiv_id="a3"
            )
            _insert_match(
                conn, paper_id="b1", external_artifact_id="r3", matched_arxiv_id="b2"
            )
            # Unmatched should be ignored
            _insert_match(
                conn, paper_id="x1", external_artifact_id="r4", matched_arxiv_id=None
            )
    finally:
        conn.close()

    comps = extract_top_k_reference_components(db_path=db_path, top_k=2)
    assert len(comps) == 2

    assert comps[0].node_count == 3
    assert set(comps[0].nodes) == {"a1", "a2", "a3"}
    # directed internal edges
    assert {(e.source, e.target) for e in comps[0].edges} == {
        ("a1", "a2"),
        ("a2", "a3"),
    }

    assert comps[1].node_count == 2
    assert set(comps[1].nodes) == {"b1", "b2"}
    assert {(e.source, e.target) for e in comps[1].edges} == {("b1", "b2")}


def test_component_json_shape(tmp_path: Path):
    db_path = tmp_path / "t.db"
    ensure_schema(db_path)

    conn = connect(db_path)
    try:
        with conn:
            _insert_match(
                conn, paper_id="p1", external_artifact_id="r1", matched_arxiv_id="p2"
            )
    finally:
        conn.close()

    comps = extract_top_k_reference_components(db_path=db_path, top_k=1)
    d = comps[0].to_json_dict()
    assert set(d.keys()) == {"rank", "stats", "nodes", "edges"}
    assert d["nodes"] == ["p1", "p2"]
    assert d["edges"] == [{"source": "p1", "target": "p2"}]

    # Ensure serializable
    json.dumps(d)


def test_normalizes_arxiv_versions_by_default(tmp_path: Path):
    db_path = tmp_path / "t.db"
    ensure_schema(db_path)

    conn = connect(db_path)
    try:
        with conn:
            _insert_match(
                conn,
                paper_id="1202.1159v1",
                external_artifact_id="r1",
                matched_arxiv_id="0706.4403v2",
            )
    finally:
        conn.close()

    comps = extract_top_k_reference_components(db_path=db_path, top_k=1)
    assert len(comps) == 1
    # version suffixes removed
    assert set(comps[0].nodes) == {"1202.1159", "0706.4403"}
    assert {(e.source, e.target) for e in comps[0].edges} == {
        ("1202.1159", "0706.4403")
    }
