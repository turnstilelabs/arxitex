from __future__ import annotations

import json

from arxitex.tools.mentions.mapping.ref_artifact_mapper import (
    Policy,
    build_gold_links,
    build_target_registry,
    map_explicit_refs_to_artifacts,
    resolve_target_version,
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


def test_resolve_target_version_prefers_explicit_id():
    registry = _registry_with_two_versions()
    row = {"target_arxiv_id": "pmihes2012"}
    res = resolve_target_version(row, registry, policy=Policy())
    assert res.status == "mapped"
    assert res.version_id == "pmihes2012"


def test_resolve_target_version_unresolved_when_explicit_missing():
    registry = _registry_with_two_versions()
    row = {"target_arxiv_id": "missing"}
    res = resolve_target_version(row, registry, policy=Policy())
    assert res.status == "version_unresolved"


def test_map_explicit_ref_exact_kind_number():
    registry = _registry_with_two_versions()
    row = {
        "context_id": "c1",
        "target_arxiv_id": "pmihes2012",
        "target_match_status": "exact_target",
        "explicit_refs": [{"kind": "definition", "number": "2.6"}],
    }
    mapped = map_explicit_refs_to_artifacts(row, registry, policy=Policy())
    assert mapped.status == "mapped"
    assert [m.statement_id for m in mapped.matches] == ["pmihes2012:def_26"]
    assert mapped.matches[0].tier == "exact"


def test_exact_kind_number_duplicate_is_dropped_as_ambiguous():
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
    row = {
        "context_id": "c1",
        "target_arxiv_id": "perfectoid",
        "target_match_status": "exact_target",
        "explicit_refs": [{"kind": "theorem", "number": "5.2"}],
    }
    mapped = map_explicit_refs_to_artifacts(row, registry, policy=Policy())
    assert mapped.status == "ref_ambiguous"
    assert mapped.reason == "exact_multi_hit"


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
    res1 = build_gold_links(rows, registry, policy=Policy())
    res2 = build_gold_links(rows, registry, policy=Policy())
    assert res1.gold_links == res2.gold_links
    assert res1.gold_links == {"c_ok": ["pmihes2012:def_26"]}
    assert res1.diagnostics.mapped_rows == 1
    assert res1.diagnostics.dropped_rows == 1


def test_non_target_never_reaches_gold_links():
    registry = _registry_with_two_versions()
    row = {
        "context_id": "c_non_target",
        "target_arxiv_id": "pmihes2012",
        "target_match_status": "non_target",
        "explicit_refs": [{"kind": "definition", "number": "2.6"}],
    }
    res = build_gold_links([row], registry, policy=Policy())
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
            {
                "allow": [
                    {
                        "version_id": "1111.4914",
                        "kind": "definition",
                        "alt_number": "2.6",
                        "statement_id": "canon:def_2_7",
                    }
                ]
            }
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
        policy=Policy(alias_curated_path=str(alias_path)),
    )
    assert res.gold_links == {"c_alias": ["canon:def_2_7"]}
    assert res.diagnostics.alias_usage.get("curated_allow") == 1
