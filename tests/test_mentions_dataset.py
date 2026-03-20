import json
from pathlib import Path

from arxitex.tools.mentions.dataset.mapping_pipeline import build_mentions_dataset
from arxitex.tools.mentions.dataset.statements import (
    find_duplicate_statement_keys,
    merge_statements,
    prefix_label,
)
from arxitex.tools.mentions.mapping.ref_artifact_mapper import build_target_registry


def test_prefix_label_idempotent():
    assert prefix_label("paper", "1.2") == "paper:1.2"
    assert prefix_label("paper", "paper:1.2") == "paper:1.2"


def test_build_mentions_dataset_maps_refs(tmp_path: Path):
    nodes = [
        {
            "id": "paper:theorem-1",
            "type": "theorem",
            "pdf_label_number": "paper:1.1",
            "content": "If X then Y.",
            "arxiv_id": "paper",
        }
    ]
    target_registry = build_target_registry(nodes)

    mentions_rows = [
        {
            "context_sentence": "By Theorem 1.1 we have X.",
            "context_prev": None,
            "context_next": None,
            "explicit_refs": [{"kind": "theorem", "number": "1.1"}],
            "target_arxiv_id": "paper",
            "target_match_status": "exact_target",
        }
    ]

    contexts_path = tmp_path / "mention_contexts.jsonl"
    labels_path = tmp_path / "mention_gold_links.json"
    total = build_mentions_dataset(
        mentions_rows=mentions_rows,
        target_registry=target_registry,
        contexts_out_path=contexts_path,
        gold_links_out_path=labels_path,
        mapping_report_path=tmp_path / "mapping_report.jsonl",
        mapping_summary_path=tmp_path / "mapping_summary.json",
    )
    assert total == 1
    context_rows = [json.loads(line) for line in contexts_path.read_text().splitlines()]
    assert len(context_rows) == 1
    row = context_rows[0]
    assert row["context_text"] == "By Theorem 1.1 we have X."
    qrels = json.loads(labels_path.read_text())
    assert row["context_id"] in qrels
    assert qrels[row["context_id"]][0] == "paper:theorem-1"


def test_build_mentions_dataset_drops_non_target(tmp_path: Path):
    nodes = [
        {
            "id": "paper:def-1",
            "type": "definition",
            "pdf_label_number": "paper:1.1",
            "content": "Definition content.",
            "arxiv_id": "paper",
        }
    ]
    target_registry = build_target_registry(nodes)

    mentions_rows = [
        {
            "context_sentence": "By Definition 1.1 we use this notion.",
            "explicit_refs": [{"kind": "definition", "number": "1.1"}],
            "target_arxiv_id": "paper",
            "target_match_status": "non_target",
        }
    ]

    total = build_mentions_dataset(
        mentions_rows=mentions_rows,
        target_registry=target_registry,
        contexts_out_path=tmp_path / "mention_contexts.jsonl",
        gold_links_out_path=tmp_path / "mention_gold_links.json",
        mapping_report_path=tmp_path / "mapping_report.jsonl",
        mapping_summary_path=tmp_path / "mapping_summary.json",
    )
    assert total == 0
    qrels = json.loads((tmp_path / "mention_gold_links.json").read_text())
    assert qrels == {}
    assert (tmp_path / "mention_contexts.jsonl").read_text().strip() == ""


def test_find_duplicate_statement_keys_detects_kind_number_collisions():
    nodes = [
        {
            "id": "paper:theorem:A",
            "type": "theorem",
            "pdf_label_number": "paper:5.2",
            "arxiv_id": "paper",
        },
        {
            "id": "paper:theorem:B",
            "type": "theorem",
            "pdf_label_number": "paper:5.2",
            "arxiv_id": "paper",
        },
        {
            "id": "paper:lemma:C",
            "type": "lemma",
            "pdf_label_number": "paper:5.2",
            "arxiv_id": "paper",
        },
    ]
    duplicates = find_duplicate_statement_keys(nodes)
    assert duplicates == {
        ("paper", "theorem", "5.2"): ["paper:theorem:A", "paper:theorem:B"]
    }


def test_merge_statements_preserves_node_arxiv_and_id_prefix(tmp_path: Path):
    src = tmp_path / "input.json"
    src.write_text(
        json.dumps(
            {
                "arxiv_id": "combined",
                "nodes": [
                    {
                        "id": "perfectoid:theorem:Main",
                        "arxiv_id": "perfectoid",
                        "type": "theorem",
                        "pdf_label_number": "5.2",
                        "content": "Main statement.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "combined.json"
    merged = merge_statements([src], out, source_map={})
    node = merged["nodes"][0]
    assert node["id"] == "perfectoid:theorem:Main"
    assert node["arxiv_id"] == "perfectoid"
    assert node["pdf_label_number"] == "perfectoid:5.2"
