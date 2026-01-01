from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema
from arxitex.extractor.models import (
    ArtifactNode,
    ArtifactType,
    DependencyType,
    DocumentGraph,
    Edge,
    Position,
    ReferenceType,
)
from arxitex.symdef.definition_bank import DefinitionBank


def load_document_graph(
    *,
    db_path: str | Path,
    paper_id: str,
    include_prerequisites: bool = True,
) -> DocumentGraph:
    """Reconstruct a DocumentGraph from the normalized SQLite schema.

    Notes
    -----
    - Internal nodes come from `artifacts`.
    - External nodes are synthesized from `artifact_edges` targets/sources that
      are not present in `artifacts`.
    - If `include_prerequisites` is True, populate each node.prerequisite_defs
      using `artifact_definition_requirements` + `definitions`.
    """

    ensure_schema(db_path)

    conn = connect(db_path)
    try:
        # Nodes (internal)
        nodes_by_id: dict[str, ArtifactNode] = {}
        rows = conn.execute(
            """
            SELECT artifact_id, artifact_type, label, content_tex, proof_tex,
                   line_start, line_end, col_start, col_end
            FROM artifacts
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchall()

        for r in rows:
            pos = None
            if r["line_start"] is not None:
                pos = Position(
                    line_start=int(r["line_start"]),
                    line_end=int(r["line_end"]) if r["line_end"] is not None else None,
                    col_start=(
                        int(r["col_start"]) if r["col_start"] is not None else None
                    ),
                    col_end=int(r["col_end"]) if r["col_end"] is not None else None,
                )

            # Best-effort type mapping.
            try:
                atype = ArtifactType(str(r["artifact_type"]))
            except Exception:
                atype = ArtifactType.UNKNOWN

            nodes_by_id[str(r["artifact_id"])] = ArtifactNode(
                id=str(r["artifact_id"]),
                type=atype,
                content=str(r["content_tex"] or ""),
                label=str(r["label"]) if r["label"] is not None else None,
                position=pos or Position(line_start=0),
                is_external=False,
                proof=str(r["proof_tex"]) if r["proof_tex"] is not None else None,
            )

        # Edges
        edges: list[Edge] = []
        erows = conn.execute(
            """
            SELECT edge_kind, source_artifact_id, target_artifact_id,
                   reference_type, dependency_type, context, justification
            FROM artifact_edges
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchall()

        # Identify external nodes (appear in edges but not in artifacts)
        external_ids: set[str] = set()
        for r in erows:
            sid = str(r["source_artifact_id"])
            tid = str(r["target_artifact_id"])
            if sid not in nodes_by_id:
                external_ids.add(sid)
            if tid not in nodes_by_id:
                external_ids.add(tid)

        for eid in external_ids:
            nodes_by_id.setdefault(
                eid,
                ArtifactNode(
                    id=eid,
                    type=ArtifactType.UNKNOWN,
                    content="",
                    label=None,
                    position=Position(line_start=0),
                    is_external=True,
                    proof=None,
                ),
            )

        for r in erows:
            dep_type = None
            ref_type = None
            if r["dependency_type"] is not None:
                try:
                    dep_type = DependencyType(str(r["dependency_type"]))
                except Exception:
                    dep_type = None
            if r["reference_type"] is not None:
                try:
                    ref_type = ReferenceType(str(r["reference_type"]))
                except Exception:
                    ref_type = None

            edges.append(
                Edge(
                    source_id=str(r["source_artifact_id"]),
                    target_id=str(r["target_artifact_id"]),
                    dependency_type=dep_type,
                    dependency=(
                        str(r["justification"])
                        if r["justification"] is not None
                        else None
                    ),
                    context=str(r["context"]) if r["context"] is not None else None,
                    reference_type=ref_type,
                )
            )

        if include_prerequisites:
            # Map artifact -> (term -> definition)
            prows = conn.execute(
                """
                SELECT r.artifact_id, d.term_original, d.definition_text
                FROM artifact_definition_requirements r
                JOIN definitions d
                  ON d.paper_id = r.paper_id
                 AND d.term_canonical = r.term_canonical
                WHERE r.paper_id = ?
                """,
                (paper_id,),
            ).fetchall()

            for r in prows:
                aid = str(r["artifact_id"])
                term = str(r["term_original"])
                dtext = str(r["definition_text"])
                node = nodes_by_id.get(aid)
                if node is None:
                    continue
                node.prerequisite_defs[term] = dtext

        return DocumentGraph(
            nodes=list(nodes_by_id.values()), edges=edges, source_file=None
        )
    finally:
        conn.close()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_paper(conn, paper: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            paper_id, title, abstract, comment, primary_category,
            all_categories_json, authors_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title=excluded.title,
            abstract=excluded.abstract,
            comment=excluded.comment,
            primary_category=excluded.primary_category,
            all_categories_json=excluded.all_categories_json,
            authors_json=excluded.authors_json
        """,
        (
            paper.get("arxiv_id"),
            paper.get("title"),
            paper.get("abstract"),
            paper.get("comment"),
            paper.get("primary_category"),
            json.dumps(paper.get("all_categories") or []),
            json.dumps(paper.get("authors") or []),
        ),
    )


def mark_state_processing(conn, *, paper_id: str, mode: str) -> None:
    conn.execute(
        """
        INSERT INTO paper_ingestion_state (paper_id, mode, stage, attempt_count, last_error, updated_at_utc)
        VALUES (?, ?, 'processing', 1, NULL, ?)
        ON CONFLICT(paper_id, mode) DO UPDATE SET
            stage='processing',
            attempt_count=paper_ingestion_state.attempt_count + 1,
            last_error=NULL,
            updated_at_utc=excluded.updated_at_utc
        """,
        (paper_id, mode, _utc_now_iso()),
    )


def mark_state_complete(conn, *, paper_id: str, mode: str) -> None:
    conn.execute(
        """
        INSERT INTO paper_ingestion_state (paper_id, mode, stage, attempt_count, last_error, updated_at_utc)
        VALUES (?, ?, 'complete', 0, NULL, ?)
        ON CONFLICT(paper_id, mode) DO UPDATE SET
            stage='complete',
            last_error=NULL,
            updated_at_utc=excluded.updated_at_utc
        """,
        (paper_id, mode, _utc_now_iso()),
    )


def mark_state_failed(conn, *, paper_id: str, mode: str, error: str) -> None:
    conn.execute(
        """
        INSERT INTO paper_ingestion_state (paper_id, mode, stage, attempt_count, last_error, updated_at_utc)
        VALUES (?, ?, 'failed', 1, ?, ?)
        ON CONFLICT(paper_id, mode) DO UPDATE SET
            stage='failed',
            last_error=excluded.last_error,
            updated_at_utc=excluded.updated_at_utc
        """,
        (paper_id, mode, error, _utc_now_iso()),
    )


def upsert_artifacts_and_edges(conn, *, paper_id: str, graph: DocumentGraph) -> None:
    # Artifacts: internal only.
    artifact_rows = []
    for node in graph.nodes:
        if node.is_external:
            continue
        pos = node.position
        artifact_rows.append(
            (
                paper_id,
                node.id,
                node.type.value,
                node.label,
                node.content or "",
                node.proof,
                pos.line_start if pos else None,
                pos.line_end if pos else None,
                pos.col_start if pos else None,
                pos.col_end if pos else None,
            )
        )

    conn.executemany(
        """
        INSERT INTO artifacts (
            paper_id, artifact_id, artifact_type, label, content_tex, proof_tex,
            line_start, line_end, col_start, col_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, artifact_id) DO UPDATE SET
            artifact_type=excluded.artifact_type,
            label=excluded.label,
            content_tex=excluded.content_tex,
            proof_tex=excluded.proof_tex,
            line_start=excluded.line_start,
            line_end=excluded.line_end,
            col_start=excluded.col_start,
            col_end=excluded.col_end
        """,
        artifact_rows,
    )

    edge_rows = []
    for e in graph.edges:
        edge_kind = "dependency" if e.dependency_type else "reference"
        edge_rows.append(
            (
                paper_id,
                edge_kind,
                e.source_id,
                e.target_id,
                e.reference_type.value if e.reference_type else None,
                e.dependency_type.value if e.dependency_type else None,
                e.context,
                e.dependency,
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO artifact_edges (
            paper_id, edge_kind, source_artifact_id, target_artifact_id,
            reference_type, dependency_type, context, justification
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        edge_rows,
    )


async def replace_definitions_and_mappings(
    conn,
    *,
    paper_id: str,
    bank: DefinitionBank,
    artifact_to_terms_map: Dict[str, List[str]],
) -> None:
    # Clear existing definition-related rows for that paper.
    conn.execute(
        "DELETE FROM artifact_definition_requirements WHERE paper_id = ?", (paper_id,)
    )
    conn.execute("DELETE FROM artifact_terms WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM definition_dependencies WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM definition_aliases WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM definitions WHERE paper_id = ?", (paper_id,))

    bank_dict = await bank.to_dict()

    def_rows = []
    alias_rows = []
    dep_rows = []

    for term_canonical, d in bank_dict.items():
        source_artifact_id = d.get("source_artifact_id")
        is_synth = int(
            bool(
                source_artifact_id
                and str(source_artifact_id).startswith("synthesized_from_context_for_")
            )
        )
        def_rows.append(
            (
                paper_id,
                term_canonical,
                d.get("term") or term_canonical,
                d.get("definition_text") or "",
                is_synth,
                source_artifact_id,
            )
        )

        for alias in d.get("aliases") or []:
            alias_rows.append((paper_id, term_canonical, alias))

        # Dependencies are stored as term strings; store canonical.
        for dep in d.get("dependencies") or []:
            dep_rows.append((paper_id, term_canonical, bank._normalize_term(dep)))

    conn.executemany(
        """
        INSERT INTO definitions (
            paper_id, term_canonical, term_original, definition_text, is_synthesized, source_artifact_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        def_rows,
    )

    if alias_rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO definition_aliases (paper_id, term_canonical, alias)
            VALUES (?, ?, ?)
            """,
            alias_rows,
        )

    if dep_rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO definition_dependencies (
                paper_id, term_canonical, depends_on_term_canonical
            ) VALUES (?, ?, ?)
            """,
            dep_rows,
        )

    # artifact_terms + artifact_definition_requirements
    term_rows = []
    req_rows = []

    # Build a quick membership set for (paper_id, term_canonical) definitions.
    known_defs = set(bank_dict.keys())

    for artifact_id, terms in (artifact_to_terms_map or {}).items():
        for term_raw in terms:
            canonical = bank._normalize_term(term_raw)
            term_rows.append((paper_id, artifact_id, canonical, term_raw))
            if canonical in known_defs:
                req_rows.append((paper_id, artifact_id, canonical))

    if term_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO artifact_terms (paper_id, artifact_id, term_canonical, term_raw)
            VALUES (?, ?, ?, ?)
            """,
            term_rows,
        )

    if req_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO artifact_definition_requirements (paper_id, artifact_id, term_canonical)
            VALUES (?, ?, ?)
            """,
            req_rows,
        )


async def persist_extraction_result(
    *,
    db_path: str | Path,
    paper_metadata: Dict[str, Any],
    graph: DocumentGraph,
    mode: str,
    bank: Optional[DefinitionBank] = None,
    artifact_to_terms_map: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Persist one paper's extraction output into SQLite.

    This function is safe to call concurrently from multiple tasks. Each call
    uses its own connection and a single transaction.
    """

    ensure_schema(db_path)

    paper_id = paper_metadata.get("arxiv_id")
    if not paper_id:
        raise ValueError("paper_metadata missing 'arxiv_id'")

    conn = connect(db_path)
    try:
        with conn:  # transaction
            upsert_paper(conn, paper_metadata)
            mark_state_processing(conn, paper_id=paper_id, mode=mode)

            upsert_artifacts_and_edges(conn, paper_id=paper_id, graph=graph)

            if mode in {"defs", "full"}:
                if bank is None or artifact_to_terms_map is None:
                    raise ValueError(
                        "defs/full mode requires bank and artifact_to_terms_map"
                    )
                await replace_definitions_and_mappings(
                    conn,
                    paper_id=paper_id,
                    bank=bank,
                    artifact_to_terms_map=artifact_to_terms_map,
                )

            mark_state_complete(conn, paper_id=paper_id, mode=mode)

        logger.debug(f"Persisted paper {paper_id} (mode={mode}) to DB")

    except Exception as e:
        # Try to record failure state.
        try:
            with conn:
                upsert_paper(conn, paper_metadata)
                mark_state_failed(conn, paper_id=paper_id, mode=mode, error=str(e))
        except Exception:
            pass
        raise
    finally:
        conn.close()
