import sqlite3

import pytest

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema


def test_schema_creates_tables_and_foreign_keys(tmp_path):
    db_path = tmp_path / "arxitex_indices.db"
    ensure_schema(db_path)

    conn = connect(db_path)
    try:
        # Ensure tables exist
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        expected = {
            "papers",
            "paper_ingestion_state",
            "artifacts",
            "artifact_edges",
            "definitions",
            "definition_aliases",
            "definition_dependencies",
            "artifact_terms",
            "artifact_definition_requirements",
        }

        assert expected.issubset(tables)

        # FK enforcement check: artifacts requires existing paper.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO artifacts (paper_id, artifact_id, artifact_type, content_tex) VALUES (?, ?, ?, ?)",
                ("nope", "a1", "lemma", "x"),
            )

        # Insert paper
        conn.execute(
            "INSERT INTO papers(paper_id, title) VALUES(?, ?)",
            ("0000.00000v1", "t"),
        )

        # Now artifact insert should succeed
        conn.execute(
            "INSERT INTO artifacts (paper_id, artifact_id, artifact_type, content_tex) VALUES (?, ?, ?, ?)",
            ("0000.00000v1", "a1", "lemma", "x"),
        )

        # definitions + requirements joinability
        conn.execute(
            "INSERT INTO definitions (paper_id, term_canonical, term_original, definition_text, is_synthesized) VALUES (?, ?, ?, ?, ?)",
            ("0000.00000v1", "g", "G", "a group", 0),
        )
        conn.execute(
            "INSERT INTO artifact_definition_requirements (paper_id, artifact_id, term_canonical) VALUES (?, ?, ?)",
            ("0000.00000v1", "a1", "g"),
        )

        row = conn.execute(
            """
            SELECT a.artifact_id, d.term_canonical
            FROM artifact_definition_requirements r
            JOIN artifacts a ON a.paper_id=r.paper_id AND a.artifact_id=r.artifact_id
            JOIN definitions d ON d.paper_id=r.paper_id AND d.term_canonical=r.term_canonical
            WHERE a.paper_id=?
            """,
            ("0000.00000v1",),
        ).fetchone()
        assert dict(row) == {"artifact_id": "a1", "term_canonical": "g"}

    finally:
        conn.close()
