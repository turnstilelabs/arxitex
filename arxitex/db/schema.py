from __future__ import annotations

from pathlib import Path

from arxitex.db.connection import connect

SCHEMA_VERSION = 1


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
        conn.commit()
    finally:
        conn.close()
