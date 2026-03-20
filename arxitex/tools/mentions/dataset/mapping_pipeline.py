"""Clean-core mention-context mapping + gold-link construction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger

from arxitex.tools.mentions.mapping.ref_artifact_mapper import (
    Policy as RefMappingPolicy,
)
from arxitex.tools.mentions.mapping.ref_artifact_mapper import build_gold_links
from arxitex.utils import read_jsonl, sha256_hash

_DEF_RE = re.compile(r"\bdefinition\b|\bdef\.?\b", flags=re.IGNORECASE)
_NON_DEF_RE = re.compile(
    r"\btheorem\b|\bthm\.?\b|\blemma\b|\blem\.?\b|\bproposition\b|\bprop\.?\b|\bcorollary\b|\bcor\.?\b",
    flags=re.IGNORECASE,
)


def _type_conflict(text: str, explicit_kind: str) -> bool:
    if not text or not explicit_kind:
        return False
    kind = explicit_kind.strip().lower()
    if kind == "definition":
        return bool(_NON_DEF_RE.search(text))
    if kind in {"theorem", "lemma", "proposition", "corollary"}:
        return bool(_DEF_RE.search(text))
    return False


def _normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def _tri_sentence_context(row: Dict) -> str:
    parts = [
        row.get("context_prev") or "",
        row.get("context_sentence") or "",
        row.get("context_next") or "",
    ]
    context = _normalize_space(" ".join(part for part in parts if part))
    if context:
        return context
    return _normalize_space(row.get("context_sentence") or "")


def _build_context_rows(
    mentions_rows: Iterable[Dict],
) -> Dict[str, Dict]:
    context_rows: Dict[str, Dict] = {}

    for row in mentions_rows:
        explicit_refs = row.get("explicit_refs") or []
        if not explicit_refs:
            continue
        if row.get("explicit_ref_source") == "paragraph":
            continue

        target_arxiv_id = row.get("target_arxiv_id")
        if not target_arxiv_id:
            continue

        context_text = _tri_sentence_context(row)
        if not context_text:
            continue

        first_kind = (explicit_refs[0].get("kind") or "").strip().lower()
        if first_kind and _type_conflict(context_text, first_kind):
            logger.debug(
                "Skipping context due to type mismatch (kind={}, arXiv={})",
                first_kind,
                row.get("arxiv_id"),
            )
            continue

        base_context_id = sha256_hash(
            "|".join(
                [
                    row.get("openalex_id") or "",
                    row.get("arxiv_id") or "",
                    row.get("context_sentence") or "",
                    row.get("cite_label") or "",
                ]
            )
        )

        ref_precision = row.get("reference_precision") or "explicit"
        for ref in explicit_refs:
            ref_kind = (ref.get("kind") or "").strip().lower()
            ref_number = (ref.get("number") or "").strip().lower()
            if not ref_kind or not ref_number:
                continue

            context_id = sha256_hash(f"{base_context_id}:{ref_kind}:{ref_number}")
            context_row = {
                "context_id": context_id,
                "context_text": context_text,
                "source_arxiv_id": row.get("arxiv_id"),
                "target_arxiv_id": target_arxiv_id,
                "explicit_refs": [
                    {
                        "kind": ref_kind,
                        "number": ref_number,
                    }
                ],
                "reference_precision": ref_precision,
                "target_match_status": row.get("target_match_status"),
            }
            context_rows[context_id] = context_row
    return context_rows


def _mapping_summary(mapping) -> Dict[str, Dict]:
    stats = mapping.diagnostics
    return {
        "total": stats.total_rows,
        "mapped": stats.mapped_rows,
        "dropped": stats.dropped_rows,
        "dropped_by_reason": stats.dropped_by_reason,
        "dropped_by_source": stats.dropped_by_source,
        "mapped_by_tier": stats.mapped_by_tier,
        "kept_by_target_status": stats.kept_by_target_status,
        "dropped_by_target_status": stats.dropped_by_target_status,
        "alias_usage": stats.alias_usage,
        "dropped_alias_reasons": stats.dropped_alias_reasons,
    }


def build_mentions_dataset(
    *,
    mentions_rows: Iterable[Dict],
    target_registry: Dict[str, object],
    contexts_out_path: Path,
    gold_links_out_path: Path,
    mapping_report_path: Optional[Path] = None,
    mapping_summary_path: Optional[Path] = None,
) -> int:
    contexts_out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_links_out_path.parent.mkdir(parents=True, exist_ok=True)

    context_rows = _build_context_rows(mentions_rows)

    policy = RefMappingPolicy()
    mapping = None
    gold_links: Dict[str, List[str]] = {}
    if context_rows:
        mapping = build_gold_links(
            context_rows.values(),
            target_registry,
            policy=policy,
            include_records=bool(mapping_report_path),
        )
        gold_links = {str(k): list(v) for k, v in mapping.gold_links.items()}

    kept_context_ids = set(gold_links.keys())
    with contexts_out_path.open("w", encoding="utf-8") as f:
        for context_id in sorted(kept_context_ids):
            row = context_rows.get(context_id)
            if not row:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gold_links_out_path.write_text(json.dumps(gold_links, indent=2))

    if mapping_report_path is not None:
        mapping_report_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_report_path.open("w", encoding="utf-8") as f:
            for rec in (mapping.records if mapping else []):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if mapping_summary_path is not None:
        summary = (
            _mapping_summary(mapping)
            if mapping
            else {"total": 0, "mapped": 0, "dropped": 0}
        )
        mapping_summary_path.write_text(json.dumps(summary, indent=2))

    mapped_pairs = sum(len(v) for v in gold_links.values())
    logger.info(
        "Wrote {} mapped context->statement links to {} / {}",
        mapped_pairs,
        contexts_out_path,
        gold_links_out_path,
    )
    return mapped_pairs


def write_target_splits(
    *,
    contexts_path: Path,
    gold_links_path: Path,
    out_dir: Path,
    val_target: str,
    test_target: str,
) -> Dict[str, int]:
    """Split mention contexts + gold links by target id."""

    out_dir.mkdir(parents=True, exist_ok=True)
    contexts = list(read_jsonl(str(contexts_path))) if contexts_path.exists() else []
    gold_links = (
        json.loads(gold_links_path.read_text(encoding="utf-8"))
        if gold_links_path.exists()
        else {}
    )

    splits = {"train": [], "val": [], "test": []}
    for row in contexts:
        target = row.get("target_arxiv_id")
        if target == val_target:
            splits["val"].append(row)
        elif target == test_target:
            splits["test"].append(row)
        else:
            splits["train"].append(row)

    counts: Dict[str, int] = {}
    for name, rows in splits.items():
        contexts_out = out_dir / f"{name}_mention_contexts.jsonl"
        with contexts_out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        context_ids = {row.get("context_id") for row in rows if row.get("context_id")}
        split_gold_links = {
            cid: vals for cid, vals in gold_links.items() if cid in context_ids
        }
        (out_dir / f"{name}_mention_gold_links.json").write_text(
            json.dumps(split_gold_links, indent=2),
            encoding="utf-8",
        )
        legacy_split = out_dir / f"{name}_mention_labels.json"
        if legacy_split.exists():
            legacy_split.unlink()
        counts[name] = len(rows)

    logger.info("Split context counts: {}", counts)
    return counts
