from arxitex.tools.retrieval.qrels_audit import audit_qrels_alignment


def test_audit_qrels_alignment_counts_mismatches_and_ambiguous_keys():
    queries = [
        {
            "query_id": "q1",
            "explicit_refs": [{"kind": "theorem", "number": "5.2"}],
        },
        {
            "query_id": "q2",
            "explicit_refs": [{"kind": "definition", "number": "2.1"}],
        },
    ]
    qrels = {"q1": ["s1"], "q2": ["s2"]}
    nodes = [
        {
            "id": "s1",
            "arxiv_id": "paper",
            "type": "theorem",
            "pdf_label_number": "paper:5.10",
        },
        {
            "id": "s2",
            "arxiv_id": "paper",
            "type": "definition",
            "pdf_label_number": "paper:2.1",
        },
        {
            "id": "s3",
            "arxiv_id": "paper",
            "type": "proposition",
            "pdf_label_number": "paper:4.5",
        },
        {
            "id": "s4",
            "arxiv_id": "paper",
            "type": "proposition",
            "pdf_label_number": "paper:4.5",
        },
    ]
    report = audit_qrels_alignment(queries=queries, qrels=qrels, graph_nodes=nodes)
    assert report["qrels_count"] == 2
    assert report["mismatch_count"] == 1
    assert report["ambiguous_statement_key_count"] == 1
