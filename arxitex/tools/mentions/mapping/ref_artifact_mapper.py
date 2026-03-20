"""Canonical explicit-ref -> statement-id mapping utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ALLOWED_TARGET_MATCH = {"exact_target", "same_work_alt_version"}
RULE_PRIORITY = {"override": 3, "deny": 2, "allow": 1}


@dataclass(frozen=True)
class Policy:
    """Strict policy with optional curated alias file."""

    alias_curated_path: Optional[str] = None


@dataclass
class MappingStats:
    total_rows: int = 0
    mapped_rows: int = 0
    dropped_rows: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    mapped_by_tier: Dict[str, int] = field(default_factory=dict)
    dropped_by_source: Dict[str, int] = field(default_factory=dict)
    kept_by_target_status: Dict[str, int] = field(default_factory=dict)
    dropped_by_target_status: Dict[str, int] = field(default_factory=dict)
    alias_usage: Dict[str, int] = field(default_factory=dict)
    dropped_alias_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class GoldLinksBuildResult:
    gold_links: Dict[str, List[str]]
    diagnostics: MappingStats
    records: List[Dict]


@dataclass
class VersionCandidate:
    version_id: str
    by_kind_number: Dict[Tuple[str, str], List[str]]
    node_ids: set[str]


TargetRegistry = Dict[str, VersionCandidate]


def _norm(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _source_id(row: Dict) -> str:
    return _norm(row.get("source_arxiv_id") or row.get("arxiv_id") or "unknown")


def _target_status(row: Dict) -> str:
    return _norm(row.get("target_match_status") or "unknown")


def _bump(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _target_drop_reason(row: Dict) -> Optional[str]:
    status = _target_status(row)
    if status == "non_target":
        return "non_target"
    if status == "unknown":
        return "unknown_target"
    if status and status not in ALLOWED_TARGET_MATCH:
        return "unknown_target"
    return None


def _numbers_for_node(raw: str, version_id: str) -> List[str]:
    if not raw:
        return []
    out = {raw}
    if ":" in raw:
        out.add(raw.split(":", 1)[1])
    else:
        out.add(f"{version_id}:{raw}")
    return sorted(out)


def _numbers_for_ref(number: str, version_id: str) -> List[str]:
    if not number:
        return []
    out = {number}
    if ":" not in number:
        out.add(f"{version_id}:{number}")
    return sorted(out)


def _make_record(
    *,
    row: Dict,
    context_id: str,
    target_status: str,
    status: str,
    drop_reason: Optional[str],
    version_status: Optional[str],
    version_id: Optional[str] = None,
    mapping_tier: Optional[str] = None,
    alias_sources: Optional[List[str]] = None,
    mapped_statement_ids: Optional[List[str]] = None,
) -> Dict:
    return {
        "context_id": context_id,
        "source_arxiv_id": row.get("source_arxiv_id") or row.get("arxiv_id"),
        "target_arxiv_id": row.get("target_arxiv_id"),
        "target_match_status": target_status or "unknown",
        "explicit_refs": row.get("explicit_refs") or [],
        "context_text": row.get("context_text") or "",
        "status": status,
        "drop_reason": drop_reason,
        "version_status": version_status,
        "version_id": version_id,
        "mapping_tier": mapping_tier,
        "alias_sources": alias_sources or [],
        "mapped_statement_ids": mapped_statement_ids or [],
    }


def _resolve_version(
    row: Dict, registry: TargetRegistry
) -> Tuple[str, Optional[str], str]:
    if not registry:
        return "version_unresolved", None, "empty_registry"

    explicit = _norm(
        row.get("resolved_target_version") or row.get("target_arxiv_id") or ""
    )
    if explicit:
        if explicit in registry:
            return "mapped", explicit, ""
        return "version_unresolved", None, "explicit_target_not_found"

    if len(registry) == 1:
        return "mapped", next(iter(registry.keys())), ""
    return "version_ambiguous", None, "missing_explicit_target_version"


def _match_exact(
    *,
    candidate: VersionCandidate,
    kind: str,
    number: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    for num in _numbers_for_ref(number, candidate.version_id):
        hits = candidate.by_kind_number.get((kind, num)) or []
        if len(hits) == 1:
            return {
                "statement_id": hits[0],
                "tier": "exact",
                "alias_source": None,
            }, None
        if len(hits) > 1:
            return None, "exact_multi_hit"
    return None, "ref_not_found"


def _load_curated_aliases(path: Optional[str]) -> Dict[Tuple[str, str, str], Dict]:
    """Canonical alias format only: list[dict]."""
    if not path:
        return {}
    alias_path = Path(path)
    if not alias_path.exists():
        return {}

    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}

    selected: Dict[Tuple[str, str, str], Dict] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        action = _norm(item.get("action") or "allow")
        if action not in {"allow", "override", "deny"}:
            continue
        key = (
            _norm(item.get("version_id") or "*") or "*",
            _norm(item.get("kind") or ""),
            _norm(item.get("alt_number") or ""),
        )
        if not key[1] or not key[2]:
            continue
        candidate = {
            "action": action,
            "statement_id": item.get("statement_id"),
            "reason": item.get("reason"),
        }
        prev = selected.get(key)
        if prev is None:
            selected[key] = candidate
            continue

        prev_pri = RULE_PRIORITY.get(prev["action"], 0)
        cand_pri = RULE_PRIORITY.get(action, 0)
        if cand_pri > prev_pri:
            selected[key] = candidate
        elif cand_pri == prev_pri and (prev.get("statement_id") or "") != (
            candidate.get("statement_id") or ""
        ):
            selected[key] = {
                "action": "deny",
                "statement_id": None,
                "reason": "curated_alias_conflict",
            }
    return selected


def _resolve_alias(
    *,
    version: VersionCandidate,
    kind: str,
    alt_number: str,
    target_status: str,
    aliases: Dict[Tuple[str, str, str], Dict],
) -> Tuple[Optional[Dict], Optional[str]]:
    if target_status != "same_work_alt_version":
        return None, None

    rule = aliases.get((version.version_id, kind, alt_number)) or aliases.get(
        ("*", kind, alt_number)
    )
    if rule is None:
        return None, "alias_unresolved"
    if rule.get("action") == "deny":
        return None, rule.get("reason") or "alias_denied"

    statement_id = rule.get("statement_id")
    if not statement_id:
        return None, rule.get("reason") or "alias_missing_statement_id"
    if statement_id not in version.node_ids:
        return None, "alias_statement_not_in_version"

    action = rule.get("action") or "allow"
    return {
        "statement_id": statement_id,
        "tier": "curated_alias",
        "alias_source": "curated_override" if action == "override" else "curated_allow",
    }, None


def _map_row_refs(
    *,
    row: Dict,
    candidate: VersionCandidate,
    aliases: Dict[Tuple[str, str, str], Dict],
) -> Tuple[str, List[Dict], str, List[str]]:
    refs = row.get("explicit_refs") or []
    if not refs:
        return "ref_unresolved", [], "missing_explicit_refs", []

    target_status = _target_status(row)
    matches: List[Dict] = []
    alias_sources: List[str] = []
    last_reason = "ref_not_found"

    for ref in refs:
        kind = _norm((ref.get("kind") or "").strip("."))
        number = _norm(ref.get("number") or "")
        if not kind or not number:
            continue

        exact_match, exact_reason = _match_exact(
            candidate=candidate, kind=kind, number=number
        )
        if exact_match is not None:
            matches.append(exact_match)
            continue
        if exact_reason == "exact_multi_hit":
            return "ref_ambiguous", [], "exact_multi_hit", []

        alias_match, alias_reason = _resolve_alias(
            version=candidate,
            kind=kind,
            alt_number=number,
            target_status=target_status,
            aliases=aliases,
        )
        if alias_match is not None:
            matches.append(alias_match)
            if alias_match.get("alias_source"):
                alias_sources.append(alias_match["alias_source"])
            continue
        if alias_reason:
            last_reason = alias_reason

    if not matches:
        return "ref_unresolved", [], last_reason, []
    unique_matches = list({m["statement_id"]: m for m in matches}.values())
    return "mapped", unique_matches, "", alias_sources


def build_target_registry(
    nodes: Iterable[Dict],
    *,
    default_version_id: str = "default",
) -> TargetRegistry:
    grouped: Dict[str, Dict[Tuple[str, str], List[str]]] = {}
    node_ids_by_version: Dict[str, set[str]] = {}

    for node in nodes:
        node_id = node.get("id")
        kind = _norm((node.get("type") or "").strip("."))
        version_id = (
            _norm(node.get("arxiv_id") or default_version_id) or default_version_id
        )
        if not node_id or not kind:
            continue
        node_ids_by_version.setdefault(version_id, set()).add(node_id)
        raw_number = _norm(node.get("pdf_label_number") or "")
        if not raw_number:
            continue
        by_kind_number = grouped.setdefault(version_id, {})
        for number in _numbers_for_node(raw_number, version_id):
            by_kind_number.setdefault((kind, number), []).append(node_id)

    registry: TargetRegistry = {}
    for version_id, by_kind_number_raw in grouped.items():
        by_kind_number = {
            key: sorted(set(ids)) for key, ids in by_kind_number_raw.items()
        }
        registry[version_id] = VersionCandidate(
            version_id=version_id,
            by_kind_number=by_kind_number,
            node_ids=node_ids_by_version.get(version_id, set()),
        )
    return registry


def build_gold_links(
    rows: Iterable[Dict],
    target_registry: TargetRegistry,
    *,
    policy: Policy,
    include_records: bool = False,
) -> GoldLinksBuildResult:
    gold_links: Dict[str, List[str]] = {}
    stats = MappingStats()
    records: List[Dict] = []
    aliases = _load_curated_aliases(policy.alias_curated_path)

    for row in rows:
        stats.total_rows += 1
        context_id = str(row.get("context_id") or "").strip()
        if not context_id:
            _bump(stats.dropped_by_reason, "missing_context_id")
            stats.dropped_rows += 1
            continue

        target_status = _target_status(row)
        target_drop = _target_drop_reason(row)
        if target_drop:
            _bump(stats.dropped_by_reason, target_drop)
            _bump(stats.dropped_by_target_status, target_status or "unknown")
            _bump(stats.dropped_by_source, _source_id(row))
            stats.dropped_rows += 1
            if include_records:
                records.append(
                    _make_record(
                        row=row,
                        context_id=context_id,
                        target_status=target_status,
                        status="dropped",
                        drop_reason=target_drop,
                        version_status="not_attempted",
                    )
                )
            continue
        _bump(stats.kept_by_target_status, target_status or "unknown")

        version_status, version_id, version_reason = _resolve_version(
            row, target_registry
        )
        if version_status != "mapped" or not version_id:
            reason = version_reason or version_status
            _bump(stats.dropped_by_reason, reason)
            _bump(stats.dropped_by_source, _source_id(row))
            stats.dropped_rows += 1
            if include_records:
                records.append(
                    _make_record(
                        row=row,
                        context_id=context_id,
                        target_status=target_status,
                        status="dropped",
                        drop_reason=reason,
                        version_status=version_status,
                        version_id=version_id,
                    )
                )
            continue

        candidate = target_registry.get(version_id)
        if candidate is None:
            reason = "version_not_in_registry"
            _bump(stats.dropped_by_reason, reason)
            _bump(stats.dropped_by_source, _source_id(row))
            stats.dropped_rows += 1
            if include_records:
                records.append(
                    _make_record(
                        row=row,
                        context_id=context_id,
                        target_status=target_status,
                        status="dropped",
                        drop_reason=reason,
                        version_status=version_status,
                        version_id=version_id,
                    )
                )
            continue

        map_status, matches, map_reason, alias_sources = _map_row_refs(
            row=row,
            candidate=candidate,
            aliases=aliases,
        )
        if map_status != "mapped":
            reason = map_reason or map_status
            _bump(stats.dropped_by_reason, reason)
            if reason.startswith("alias_"):
                _bump(stats.dropped_alias_reasons, reason)
            _bump(stats.dropped_by_source, _source_id(row))
            stats.dropped_rows += 1
            if include_records:
                records.append(
                    _make_record(
                        row=row,
                        context_id=context_id,
                        target_status=target_status,
                        status="dropped",
                        drop_reason=reason,
                        version_status=version_status,
                        version_id=version_id,
                    )
                )
            continue

        statement_ids = [m["statement_id"] for m in matches]
        if statement_ids:
            gold_links[context_id] = statement_ids
        stats.mapped_rows += 1
        for match in matches:
            _bump(stats.mapped_by_tier, match["tier"])
            if match.get("alias_source"):
                _bump(stats.alias_usage, match["alias_source"])

        if include_records:
            records.append(
                _make_record(
                    row=row,
                    context_id=context_id,
                    target_status=target_status,
                    status="mapped",
                    drop_reason=None,
                    version_status=version_status,
                    version_id=version_id,
                    mapping_tier=",".join(sorted({m["tier"] for m in matches})),
                    alias_sources=sorted({s for s in alias_sources if s}),
                    mapped_statement_ids=statement_ids,
                )
            )

    return GoldLinksBuildResult(
        gold_links=gold_links, diagnostics=stats, records=records
    )
