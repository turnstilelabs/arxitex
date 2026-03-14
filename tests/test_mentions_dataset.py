import json
from pathlib import Path

from arxitex.tools.mentions.dataset.build_dataset import (
    _build_mentions_dataset,
    _build_statement_index,
    _prefix_label,
)


def test_prefix_label_idempotent():
    assert _prefix_label("paper", "1.2") == "paper:1.2"
    assert _prefix_label("paper", "paper:1.2") == "paper:1.2"


def test_build_mentions_dataset_maps_refs(tmp_path: Path):
    graph = {
        "nodes": [
            {
                "id": "paper:theorem-1",
                "type": "theorem",
                "pdf_label_number": "paper:1.1",
                "content": "If X then Y.",
            }
        ],
        "edges": [],
    }
    index = _build_statement_index(graph)

    mention_path = tmp_path / "mentions.jsonl"
    mention_path.write_text(
        json.dumps(
            {
                "context_sentence": "By Theorem 1.1 we have X.",
                "context_prev": None,
                "context_next": None,
                "explicit_refs": [{"kind": "theorem", "number": "1.1"}],
                "target_arxiv_id": "paper",
            }
        )
        + "\n"
    )

    out_path = tmp_path / "dataset.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.json"
    total = _build_mentions_dataset(
        mentions_paths=[mention_path],
        statement_index=index,
        out_path=out_path,
        queries_out_path=queries_path,
        qrels_out_path=qrels_path,
    )
    assert total == 1
    rows = out_path.read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["target_statement_id"] == "paper:theorem-1"
    qrels = json.loads(qrels_path.read_text())
    assert row["query_id"] in qrels
    assert qrels[row["query_id"]][0] == "paper:theorem-1"
