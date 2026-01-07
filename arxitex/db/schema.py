from __future__ import annotations

from pathlib import Path

from arxitex.db.connection import connect

SCHEMA_VERSION = 3


def _table_has_column(conn, table: str, column: str) -> bool:
    try:
        cols = [
            r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
        ]
    except Exception:
        return False
    return column in cols


def _migrate_paper_citations_drop_raw_json(conn) -> None:
    """Migration: drop `raw_json` column from paper_citations.

    SQLite doesn't support DROP COLUMN reliably across versions, so we rebuild.
    """

    if not _table_has_column(conn, "paper_citations", "raw_json"):
        return

    # Rebuild the table without raw_json.
    conn.execute("ALTER TABLE paper_citations RENAME TO paper_citations_old")

    conn.execute(
        """
        CREATE TABLE paper_citations (
            paper_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_work_id TEXT,
            citation_count INTEGER,
            last_fetched_at_utc TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                ON DELETE CASCADE
        );
        """
    )

    conn.execute(
        """
        INSERT INTO paper_citations (paper_id, source, source_work_id, citation_count, last_fetched_at_utc)
        SELECT paper_id, source, source_work_id, citation_count, last_fetched_at_utc
        FROM paper_citations_old
        """
    )

    conn.execute("DROP TABLE paper_citations_old")

    # Recreate indexes.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_citations_count ON paper_citations(citation_count);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_citations_fetched ON paper_citations(last_fetched_at_utc);"
    )


def ensure_schema(db_path: str | Path) -> None:
    """Create (if needed) the normalized persistence schema.

    We keep this idempotent (CREATE TABLE IF NOT EXISTS), so it's safe to call
    at every program start.
    """

    conn = connect(db_path)
    try:
        # A tiny schema-version table (optional but useful).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS arxitex_schema_version (
                version INTEGER NOT NULL
            );
            """
        )

        # -- Papers
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                comment TEXT,
                primary_category TEXT,
                all_categories_json TEXT,
                authors_json TEXT
            );
            """
        )

        # -- Minimal migrations (idempotent)
        # We currently use a small set of targeted migrations keyed off schema
        # inspection, since SQLite schemas can exist in legacy states.
        with conn:
            if _table_has_column(conn, "paper_citations", "raw_json"):
                _migrate_paper_citations_drop_raw_json(conn)

        # -- Per-paper processing state (tracked per mode)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_ingestion_state (
                paper_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (paper_id, mode),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                    ON DELETE CASCADE
            );
            """
        )

        # -- Artifacts (internal only)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                paper_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                label TEXT,
                content_tex TEXT NOT NULL,
                proof_tex TEXT,
                line_start INTEGER,
                line_end INTEGER,
                col_start INTEGER,
                col_end INTEGER,
                PRIMARY KEY (paper_id, artifact_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                    ON DELETE CASCADE
            );
            """
        )

        # -- Edges
        # NOTE: we don't FK source/target artifact ids because targets can be external.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_edges (
                paper_id TEXT NOT NULL,
                edge_kind TEXT NOT NULL, -- reference|dependency
                source_artifact_id TEXT NOT NULL,
                target_artifact_id TEXT NOT NULL,
                reference_type TEXT,
                dependency_type TEXT,
                context TEXT,
                justification TEXT,
                PRIMARY KEY (paper_id, edge_kind, source_artifact_id, target_artifact_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                    ON DELETE CASCADE
            );
            """
        )

        # -- Definitions (per paper)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS definitions (
                paper_id TEXT NOT NULL,
                term_canonical TEXT NOT NULL,
                term_original TEXT NOT NULL,
                definition_text TEXT NOT NULL,
                is_synthesized INTEGER NOT NULL,
                source_artifact_id TEXT,
                PRIMARY KEY (paper_id, term_canonical),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                    ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS definition_aliases (
                paper_id TEXT NOT NULL,
                term_canonical TEXT NOT NULL,
                alias TEXT NOT NULL,
                PRIMARY KEY (paper_id, term_canonical, alias),
                FOREIGN KEY (paper_id, term_canonical)
                    REFERENCES definitions(paper_id, term_canonical)
                    ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS definition_dependencies (
                paper_id TEXT NOT NULL,
                term_canonical TEXT NOT NULL,
                depends_on_term_canonical TEXT NOT NULL,
                PRIMARY KEY (paper_id, term_canonical, depends_on_term_canonical),
                FOREIGN KEY (paper_id, term_canonical)
                    REFERENCES definitions(paper_id, term_canonical)
                    ON DELETE CASCADE,
                FOREIGN KEY (paper_id, depends_on_term_canonical)
                    REFERENCES definitions(paper_id, term_canonical)
                    ON DELETE CASCADE
            );
            """
        )

        # -- Artifact term usage
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_terms (
                paper_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                term_canonical TEXT NOT NULL,
                term_raw TEXT NOT NULL,
                PRIMARY KEY (paper_id, artifact_id, term_canonical),
                FOREIGN KEY (paper_id, artifact_id)
                    REFERENCES artifacts(paper_id, artifact_id)
                    ON DELETE CASCADE
            );
            """
        )

        # -- Artifact -> required definitions
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_definition_requirements (
                paper_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                term_canonical TEXT NOT NULL,
                PRIMARY KEY (paper_id, artifact_id, term_canonical),
                FOREIGN KEY (paper_id, artifact_id)
                    REFERENCES artifacts(paper_id, artifact_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (paper_id, term_canonical)
                    REFERENCES definitions(paper_id, term_canonical)
                    ON DELETE CASCADE
            );
            """
        )

        # -- LLM usage (token accounting)
        # One row per LLM call. This is intentionally append-only.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                paper_id TEXT,
                mode TEXT,
                stage TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_id TEXT,
                context TEXT,
                cached INTEGER NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                    ON DELETE SET NULL
            );
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_usage_paper ON llm_usage(paper_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_usage_model ON llm_usage(model);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at_utc);"
        )

        # -- Per-paper citation counts (VIP selection)
        # We store per *base* arXiv id (strip vN). Total citations are sourced from OpenAlex.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_citations (
                paper_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_work_id TEXT,
                citation_count INTEGER,
                last_fetched_at_utc TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                    ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_citations_count ON paper_citations(citation_count);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_citations_fetched ON paper_citations(last_fetched_at_utc);"
        )

        # -- External reference -> matched arXiv id
        # This is a post-processing/backfill table to connect external
        # bibliography references to an arXiv paper id (when the bib entry does
        # not explicitly mention arXiv).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_reference_arxiv_matches (
                paper_id TEXT NOT NULL,
                external_artifact_id TEXT NOT NULL,
                matched_arxiv_id TEXT,
                match_method TEXT NOT NULL, -- direct_regex|search|none
                extracted_title TEXT,
                extracted_authors_json TEXT,
                matched_title TEXT,
                matched_authors_json TEXT,
                title_score REAL,
                author_overlap REAL,
                arxiv_query TEXT,
                last_matched_at_utc TEXT NOT NULL,
                PRIMARY KEY (paper_id, external_artifact_id),
                FOREIGN KEY (paper_id, external_artifact_id)
                    REFERENCES artifacts(paper_id, artifact_id)
                    ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_extref_matches_arxiv_id ON external_reference_arxiv_matches(matched_arxiv_id);"
        )

        # Cache to avoid repeatedly querying arXiv for the same normalized
        # (title, authors) tuple.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_reference_arxiv_search_cache (
                cache_key TEXT PRIMARY KEY,
                matched_arxiv_id TEXT,
                matched_title TEXT,
                matched_authors_json TEXT,
                title_score REAL,
                author_overlap REAL,
                arxiv_query TEXT,
                last_fetched_at_utc TEXT NOT NULL
            );
            """
        )

        # Useful indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_by_paper ON artifacts(paper_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_by_paper_source ON artifact_edges(paper_id, source_artifact_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_by_paper_target ON artifact_edges(paper_id, target_artifact_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_defs_by_paper ON definitions(paper_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_state_by_stage ON paper_ingestion_state(stage);"
        )

        # Initialize schema version row if missing.
        row = conn.execute(
            "SELECT version FROM arxitex_schema_version LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO arxitex_schema_version(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        else:
            # Best-effort update to the latest version (schema is maintained to
            # be backwards compatible via migrations above).
            try:
                if int(row["version"]) < SCHEMA_VERSION:
                    conn.execute(
                        "UPDATE arxitex_schema_version SET version = ?",
                        (SCHEMA_VERSION,),
                    )
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()
