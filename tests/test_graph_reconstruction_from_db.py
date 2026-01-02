from arxitex.db.connection import connect
from arxitex.db.persistence import load_document_graph
from arxitex.db.schema import ensure_schema


def test_load_document_graph_reconstructs_edges_and_prereqs(tmp_path):
    db_path = tmp_path / "arxitex_indices.db"
    ensure_schema(db_path)

    conn = connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO papers(paper_id, title) VALUES(?, ?)", ("p1", "t")
            )

            # Two artifacts and one dependency edge
            conn.execute(
                """
                INSERT INTO artifacts(
                    paper_id, artifact_id, artifact_type, label, content_tex, proof_tex,
                    line_start, line_end, col_start, col_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("p1", "a1", "lemma", None, "lemma content", None, 10, 11, None, None),
            )
            conn.execute(
                """
                INSERT INTO artifacts(
                    paper_id, artifact_id, artifact_type, label, content_tex, proof_tex,
                    line_start, line_end, col_start, col_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "p1",
                    "a2",
                    "theorem",
                    None,
                    "theorem content",
                    None,
                    20,
                    21,
                    None,
                    None,
                ),
            )
            conn.execute(
                """
                INSERT INTO artifact_edges(
                    paper_id, edge_kind, source_artifact_id, target_artifact_id,
                    reference_type, dependency_type, context, justification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("p1", "dependency", "a2", "a1", None, "used_in", "ctx", "because"),
            )

            # Definition + requirement for a2
            conn.execute(
                """
                INSERT INTO definitions(
                    paper_id, term_canonical, term_original, definition_text, is_synthesized, source_artifact_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("p1", "group", "Group", "a group is ...", 0, None),
            )
            conn.execute(
                """
                INSERT INTO artifact_definition_requirements(paper_id, artifact_id, term_canonical)
                VALUES (?, ?, ?)
                """,
                ("p1", "a2", "group"),
            )
    finally:
        conn.close()

    g = load_document_graph(db_path=db_path, paper_id="p1", include_prerequisites=True)
    assert {n.id for n in g.nodes} == {"a1", "a2"}
    assert len(g.edges) == 1
    e = g.edges[0]
    assert (e.source_id, e.target_id) == ("a2", "a1")
    assert e.dependency_type is not None

    n2 = next(n for n in g.nodes if n.id == "a2")
    assert n2.prerequisite_defs.get("Group") == "a group is ..."
