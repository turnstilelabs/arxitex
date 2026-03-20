from __future__ import annotations

import json

from arxitex.tools.mentions.mapping.ref_artifact_mapper import (
    build_gold_links,
    build_target_registry,
)


def _registry_with_two_versions():
    nodes = [
        {
            "id": "1111.4914:thm_63",
            "arxiv_id": "1111.4914",
            "type": "theorem",
            "pdf_label_number": "6.3",
            "content": "Main theorem from arXiv version.",
        },
        {
            "id": "pmihes2012:thm_63",
            "arxiv_id": "pmihes2012",
            "type": "theorem",
            "pdf_label_number": "6.3",
            "content": "Main theorem from PMIHES version.",
        },
        {
            "id": "pmihes2012:def_26",
            "arxiv_id": "pmihes2012",
            "type": "definition",
            "pdf_label_number": "2.6",
            "content": "Perfectoid definition.",
        },
    ]
    return build_target_registry(nodes)


def test_build_gold_links_exact_kind_number():
    registry = _registry_with_two_versions()
    rows = [
        {
            "context_id": "c1",
            "target_arxiv_id": "pmihes2012",
            "target_match_status": "exact_target",
            "explicit_refs": [{"kind": "definition", "number": "2.6"}],
            "context_text": "By Definition 2.6.",
        }
    ]
    mapped = build_gold_links(rows, registry, include_records=True)
    assert mapped.gold_links == {"c1": ["pmihes2012:def_26"]}
    assert mapped.records[0]["mapping_tier"] == "exact"


def test_build_gold_links_duplicate_exact_is_ambiguous_drop():
    nodes = [
        {
            "id": "perfectoid:theorem:A",
            "arxiv_id": "perfectoid",
            "type": "theorem",
            "pdf_label_number": "5.2",
            "content": "First duplicate.",
        },
        {
            "id": "perfectoid:theorem:B",
            "arxiv_id": "perfectoid",
            "type": "theorem",
            "pdf_label_number": "5.2",
            "content": "Second duplicate.",
        },
    ]
    registry = build_target_registry(nodes)
    rows = [
        {
            "context_id": "c1",
            "target_arxiv_id": "perfectoid",
            "target_match_status": "exact_target",
            "explicit_refs": [{"kind": "theorem", "number": "5.2"}],
            "context_text": "By Theorem 5.2.",
        }
    ]
    mapped = build_gold_links(rows, registry, include_records=True)
    assert mapped.gold_links == {}
    assert mapped.diagnostics.dropped_by_reason.get("exact_multi_hit") == 1


def test_build_gold_links_drops_unresolved_and_is_deterministic():
    registry = _registry_with_two_versions()
    rows = [
        {
            "context_id": "c_ok",
            "target_arxiv_id": "pmihes2012",
            "target_match_status": "exact_target",
            "explicit_refs": [{"kind": "definition", "number": "2.6"}],
            "source_arxiv_id": "srcA",
            "context_text": "By Definition 2.6, ...",
        },
        {
            "context_id": "c_drop",
            "target_arxiv_id": "pmihes2012",
            "target_match_status": "exact_target",
            "explicit_refs": [{"kind": "theorem", "number": "99.99"}],
            "source_arxiv_id": "srcB",
            "context_text": "By Theorem 99.99, ...",
        },
    ]
    res1 = build_gold_links(rows, registry)
    res2 = build_gold_links(rows, registry)
    assert res1.gold_links == res2.gold_links
    assert res1.gold_links == {"c_ok": ["pmihes2012:def_26"]}
    assert res1.diagnostics.mapped_rows == 1
    assert res1.diagnostics.dropped_rows == 1
    assert res1.diagnostics.dropped_by_reason.get("ref_not_found") == 1


def test_non_target_never_reaches_gold_links():
    registry = _registry_with_two_versions()
    row = {
        "context_id": "c_non_target",
        "target_arxiv_id": "pmihes2012",
        "target_match_status": "non_target",
        "explicit_refs": [{"kind": "definition", "number": "2.6"}],
    }
    res = build_gold_links([row], registry)
    assert res.gold_links == {}
    assert res.diagnostics.dropped_by_reason.get("non_target") == 1


def test_curated_alias_maps_same_work_alt_version(tmp_path):
    nodes = [
        {
            "id": "canon:def_2_7",
            "arxiv_id": "1111.4914",
            "type": "definition",
            "pdf_label_number": "2.7",
            "content": "A perfectoid algebra is uniform and complete for p-adic topology.",
        }
    ]
    registry = build_target_registry(nodes)

    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(
        json.dumps(
            [
                {
                    "version_id": "1111.4914",
                    "kind": "definition",
                    "alt_number": "2.6",
                    "statement_id": "canon:def_2_7",
                    "action": "allow",
                }
            ]
        ),
        encoding="utf-8",
    )

    row = {
        "context_id": "c_alias",
        "target_arxiv_id": "1111.4914",
        "target_match_status": "same_work_alt_version",
        "context_text": "Definition 2.6 says perfectoid algebra is uniform and complete p-adically.",
        "explicit_refs": [{"kind": "definition", "number": "2.6"}],
    }
    res = build_gold_links(
        [row],
        registry,
        alias_curated_path=str(alias_path),
    )
    assert res.gold_links == {"c_alias": ["canon:def_2_7"]}
    assert res.diagnostics.alias_usage.get("curated_allow") == 1


def test_no_number_only_fallback_kind_mismatch_stays_unresolved():
    nodes = [
        {
            "id": "paper:def_1_1",
            "arxiv_id": "paper",
            "type": "definition",
            "pdf_label_number": "1.1",
            "content": "Definition content.",
        }
    ]
    registry = build_target_registry(nodes)
    rows = [
        {
            "context_id": "c1",
            "target_arxiv_id": "paper",
            "target_match_status": "exact_target",
            "explicit_refs": [{"kind": "theorem", "number": "1.1"}],
        }
    ]
    res = build_gold_links(rows, registry)
    assert res.gold_links == {}
    assert res.diagnostics.dropped_by_reason.get("ref_not_found") == 1
