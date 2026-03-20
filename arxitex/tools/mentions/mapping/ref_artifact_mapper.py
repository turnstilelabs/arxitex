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
    """Only curated alias file path is configurable; strict mapping is fixed."""

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
    by_number: Dict[str, Dict[str, List[str]]]
    node_ids: set[str]


TargetRegistry = Dict[str, VersionCandidate]


def _norm(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _source_id(row: Dict) -> str:
    return _norm(row.get("source_arxiv_id") or row.get("arxiv_id") or "unknown")


def _target_status(row: Dict) -> str:
    return _norm(row.get("target_match_status") or "unknown")


def _ref_kind(ref: Dict) -> str:
    return _norm((ref.get("kind") or "").strip("."))


def _ref_number(ref: Dict) -> str:
    return _norm(ref.get("number") or "")


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


def build_target_registry(
    nodes: Iterable[Dict],
    *,
    default_version_id: str = "default",
) -> TargetRegistry:
    grouped: Dict[str, Dict] = {}
    for node in nodes:
        node_id = node.get("id")
        kind = _norm((node.get("type") or "").strip("."))
        version_id = (
            _norm(node.get("arxiv_id") or default_version_id) or default_version_id
        )
        if not node_id or not kind:
            continue

        bucket = grouped.setdefault(
            version_id,
            {
                "node_ids": set(),
                "by_kind_number": {},
            },
        )
        bucket["node_ids"].add(node_id)

        number_raw = _norm(node.get("pdf_label_number") or "")
        if not number_raw:
            continue
        by_kn = bucket["by_kind_number"]
        for number in _numbers_for_node(number_raw, version_id):
            by_kn.setdefault((kind, number), []).append(node_id)

    registry: TargetRegistry = {}
    for version_id, bucket in grouped.items():
        by_kind_number: Dict[Tuple[str, str], List[str]] = {}
        by_number: Dict[str, Dict[str, List[str]]] = {}
        for (kind, number), ids in bucket["by_kind_number"].items():
            uniq_ids = sorted(set(ids))
            by_kind_number[(kind, number)] = uniq_ids
            by_number.setdefault(number, {})[kind] = uniq_ids
        registry[version_id] = VersionCandidate(
            version_id=version_id,
            by_kind_number=by_kind_number,
            by_number=by_number,
            node_ids=set(bucket["node_ids"]),
        )
    return registry


def resolve_target_version(
    mention_row: Dict,
    target_registry: TargetRegistry,
    *,
    policy: Policy,
) -> Dict[str, Optional[str]]:
    _ = policy
    if not target_registry:
        return {
            "status": "version_unresolved",
            "version_id": None,
            "reason": "empty_registry",
        }

    explicit = _norm(
        mention_row.get("resolved_target_version")
        or mention_row.get("target_arxiv_id")
        or ""
    )
    if explicit:
        if explicit in target_registry:
            return {"status": "mapped", "version_id": explicit, "reason": None}
        return {
            "status": "version_unresolved",
            "version_id": None,
            "reason": "explicit_target_not_found",
        }

    if len(target_registry) == 1:
        only = next(iter(target_registry.values()))
        return {"status": "mapped", "version_id": only.version_id, "reason": None}

    return {
        "status": "version_ambiguous",
        "version_id": None,
        "reason": "missing_explicit_target_version",
    }


def _map_direct_ref(
    *,
    candidate: VersionCandidate,
    kind: str,
    number: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    for num in _numbers_for_ref(number, candidate.version_id):
        exact_hits = candidate.by_kind_number.get((kind, num)) or []
        if len(exact_hits) == 1:
            return {
                "statement_id": exact_hits[0],
                "tier": "exact",
                "alias_source": None,
            }, None
        if len(exact_hits) > 1:
            return None, "exact_multi_hit"

        same_kind = (candidate.by_number.get(num) or {}).get(kind) or []
        if len(same_kind) == 1:
            return {
                "statement_id": same_kind[0],
                "tier": "number_only",
                "alias_source": None,
            }, None
        if len(same_kind) > 1:
            return None, "number_only_multi_hit_same_kind"
    return None, None


def _iter_alias_entries(payload: object) -> Iterable[Tuple[str, Dict]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield "allow", item
        return
    if not isinstance(payload, dict):
        return
    for action in ("allow", "override", "deny"):
        entries = payload.get(action)
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict):
                yield action, item


def _load_curated_aliases(path: Optional[str]) -> Dict[Tuple[str, str, str], Dict]:
    if not path:
        return {}
    alias_path = Path(path)
    if not alias_path.exists():
        return {}

    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    selected: Dict[Tuple[str, str, str], Dict] = {}
    for action_hint, item in _iter_alias_entries(payload):
        action = _norm(item.get("action") or action_hint or "allow")
        if action not in {"allow", "override", "deny"}:
            continue
        key = (
            _norm(item.get("version_id") or "*") or "*",
            _norm(item.get("kind") or ""),
            _norm(item.get("alt_number") or item.get("number") or ""),
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
            continue
        if cand_pri == prev_pri and (prev.get("statement_id") or "") != (
            candidate.get("statement_id") or ""
        ):
            selected[key] = {
                "action": "deny",
                "statement_id": None,
                "reason": "curated_alias_conflict",
            }
    return selected


def _resolve_curated_alias(
    *,
    version: VersionCandidate,
    kind: str,
    alt_number: str,
    target_status: str,
    aliases: Dict[Tuple[str, str, str], Dict],
) -> Tuple[Optional[Dict], Optional[str]]:
    if target_status != "same_work_alt_version":
        return None, "alias_not_allowed_for_target_status"

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


def map_explicit_refs_to_artifacts(
    mention_row: Dict,
    target_registry: TargetRegistry,
    *,
    policy: Policy,
    version_resolution: Optional[Dict[str, Optional[str]]] = None,
    curated_rules_by_key: Optional[Dict[Tuple[str, str, str], Dict]] = None,
) -> Dict:
    version_resolution = version_resolution or resolve_target_version(
        mention_row, target_registry, policy=policy
    )
    version_status = version_resolution.get("status")
    version_id = version_resolution.get("version_id")
    if version_status != "mapped" or not version_id:
        return {
            "status": "ref_unresolved",
            "matches": [],
            "reason": "version_unresolved",
        }

    version = target_registry.get(version_id)
    if not version:
        return {
            "status": "ref_unresolved",
            "matches": [],
            "reason": "version_not_in_registry",
        }

    refs = mention_row.get("explicit_refs") or []
    if not refs:
        return {
            "status": "ref_unresolved",
            "matches": [],
            "reason": "missing_explicit_refs",
        }

    target_status = _target_status(mention_row)
    aliases = curated_rules_by_key or {}
    matches: List[Dict] = []
    alias_sources: List[str] = []
    last_reason = "no_ref_match"

    for ref in refs:
        kind = _ref_kind(ref)
        number = _ref_number(ref)
        if not kind or not number:
            continue

        direct_match, direct_reason = _map_direct_ref(
            candidate=version,
            kind=kind,
            number=number,
        )
        if direct_match is not None:
            matches.append(direct_match)
            continue
        if direct_reason and target_status != "same_work_alt_version":
            return {
                "status": "ref_ambiguous",
                "matches": [],
                "reason": direct_reason,
            }

        alias_match, alias_reason = _resolve_curated_alias(
            version=version,
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
        last_reason = direct_reason or alias_reason or "no_ref_match"

    if not matches:
        reason = "alias_unresolved" if last_reason.startswith("alias_") else last_reason
        return {"status": "ref_unresolved", "matches": [], "reason": reason}

    unique_matches = list({m["statement_id"]: m for m in matches}.values())
    return {
        "status": "mapped",
        "matches": unique_matches,
        "reason": None,
        "alias_sources": alias_sources,
    }


def build_gold_links(
    rows: Iterable[Dict],
    target_registry: TargetRegistry,
    *,
    policy: Policy,
) -> GoldLinksBuildResult:
    gold_links: Dict[str, List[str]] = {}
    stats = MappingStats()
    records: List[Dict] = []
    curated_aliases = _load_curated_aliases(policy.alias_curated_path)

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

        version = resolve_target_version(row, target_registry, policy=policy)
        if version.get("status") != "mapped" or not version.get("version_id"):
            reason = (
                version.get("reason") or version.get("status") or "version_unresolved"
            )
            _bump(stats.dropped_by_reason, reason)
            _bump(stats.dropped_by_source, _source_id(row))
            stats.dropped_rows += 1
            records.append(
                _make_record(
                    row=row,
                    context_id=context_id,
                    target_status=target_status,
                    status="dropped",
                    drop_reason=reason,
                    version_status=version.get("status"),
                    version_id=version.get("version_id"),
                )
            )
            continue

        mapped = map_explicit_refs_to_artifacts(
            row,
            target_registry,
            policy=policy,
            version_resolution=version,
            curated_rules_by_key=curated_aliases,
        )
        if mapped.get("status") != "mapped":
            reason = mapped.get("reason") or mapped.get("status") or "ref_unresolved"
            _bump(stats.dropped_by_reason, reason)
            if reason.startswith("alias_"):
                _bump(stats.dropped_alias_reasons, reason)
            _bump(stats.dropped_by_source, _source_id(row))
            stats.dropped_rows += 1
            records.append(
                _make_record(
                    row=row,
                    context_id=context_id,
                    target_status=target_status,
                    status="dropped",
                    drop_reason=reason,
                    version_status=version.get("status"),
                    version_id=version.get("version_id"),
                )
            )
            continue

        matches = mapped.get("matches") or []
        statement_ids = [m["statement_id"] for m in matches]
        if statement_ids:
            gold_links[context_id] = statement_ids
        stats.mapped_rows += 1

        for match in matches:
            _bump(stats.mapped_by_tier, match["tier"])
            if match.get("alias_source"):
                _bump(stats.alias_usage, match["alias_source"])

        records.append(
            _make_record(
                row=row,
                context_id=context_id,
                target_status=target_status,
                status="mapped",
                drop_reason=None,
                version_status=version.get("status"),
                version_id=version.get("version_id"),
                mapping_tier=",".join(sorted({m["tier"] for m in matches})),
                alias_sources=sorted(
                    {
                        s
                        for s in (mapped.get("alias_sources") or [])
                        if isinstance(s, str) and s
                    }
                ),
                mapped_statement_ids=statement_ids,
            )
        )

    return GoldLinksBuildResult(
        gold_links=gold_links, diagnostics=stats, records=records
    )
