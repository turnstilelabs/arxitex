import sqlite3

from arxitex.db.schema import ensure_schema
from arxitex.tools.citations.arxiv_backfill import (
    backfill_external_reference_arxiv_matches,
)


def test_backfill_creates_joinable_rows(monkeypatch, tmp_path):
    db_path = tmp_path / "db.sqlite"
    ensure_schema(db_path)

    # Insert a paper + an external reference artifact.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("INSERT INTO papers(paper_id) VALUES (?)", ("p1",))
        conn.execute(
            """
            INSERT INTO artifacts(paper_id, artifact_id, artifact_type, label, content_tex)
            VALUES (?, ?, 'external_reference', ?, ?)
            """,
            (
                "p1",
                "external_Doe01",
                "Doe01",
                'J. Doe, "A Great Paper", 2021.',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Mark the paper as successfully processed.
    connp = sqlite3.connect(str(db_path))
    try:
        # processed_papers table is managed by ProcessedIndex in production; create
        # a minimal compatible version here.
        connp.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_papers (
                arxiv_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                processed_timestamp_utc TEXT NOT NULL,
                output_path TEXT,
                details TEXT
            )
            """
        )
        connp.execute(
            "INSERT OR REPLACE INTO processed_papers(arxiv_id, status, processed_timestamp_utc) VALUES (?, ?, ?)",
            ("p1", "success", "2020-01-01T00:00:00+00:00"),
        )
        connp.commit()
    finally:
        connp.close()

    # Patch the matcher so the test doesn't perform network calls.
    from arxitex import tools as _tools_pkg  # noqa: F401
    from arxitex.tools.citations import arxiv_backfill as backfill_mod
    from arxitex.tools.citations.arxiv_matcher import MatchResult

    def fake_matcher(**kwargs):
        return MatchResult(
            matched_arxiv_id="1111.2222",
            match_method="search",
            extracted_title="A Great Paper",
            extracted_authors=["J. Doe"],
            matched_title="A Great Paper",
            matched_authors=["John Doe"],
            title_score=0.99,
            author_overlap=1.0,
            arxiv_query='ti:"A Great Paper"',
        )

    monkeypatch.setattr(backfill_mod, "match_external_reference_to_arxiv", fake_matcher)

    # Run backfill.
    import asyncio

    asyncio.run(
        backfill_external_reference_arxiv_matches(
            db_path=str(db_path),
            only_paper_ids=["p1"],
            only_processed_success=True,
            qps=1000,
            refresh_days=0,
            force=True,
        )
    )

    conn2 = sqlite3.connect(str(db_path))
    try:
        conn2.row_factory = sqlite3.Row
        row = conn2.execute(
            """
            SELECT a.paper_id, a.artifact_id, m.matched_arxiv_id
            FROM artifacts a
            JOIN external_reference_arxiv_matches m
              ON m.paper_id = a.paper_id
             AND m.external_artifact_id = a.artifact_id
            WHERE a.paper_id = ? AND a.artifact_id = ?
            """,
            ("p1", "external_Doe01"),
        ).fetchone()
        assert row is not None
        assert row["matched_arxiv_id"] == "1111.2222"
    finally:
        conn2.close()
