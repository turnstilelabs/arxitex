from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _norm_kind(value: str) -> str:
    return (value or "").strip().lower().rstrip(".")


def _norm_number(value: str) -> str:
    raw = (value or "").strip().lower()
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    return raw


def _first_explicit_ref(query_row: Dict) -> Tuple[str, str]:
    refs = query_row.get("explicit_refs") or []
    if not refs:
        return ("", "")
    first = refs[0] or {}
    return (
        _norm_kind(first.get("kind") or ""),
        _norm_number(first.get("number") or ""),
    )


def _build_node_index(nodes: Iterable[Dict]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        out[node_id] = (
            _norm_kind(node.get("type") or ""),
            _norm_number(node.get("pdf_label_number") or ""),
        )
    return out


def _count_ambiguous_statement_keys(nodes: Iterable[Dict]) -> int:
    buckets: Dict[Tuple[str, str, str], int] = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        arxiv_id = (node.get("arxiv_id") or node_id.split(":", 1)[0] or "").strip()
        kind = _norm_kind(node.get("type") or "")
        number = _norm_number(node.get("pdf_label_number") or "")
        if not arxiv_id or not kind or not number:
            continue
        key = (arxiv_id, kind, number)
        buckets[key] = buckets.get(key, 0) + 1
    return sum(1 for count in buckets.values() if count > 1)


def audit_qrels_alignment(
    *,
    queries: List[Dict],
    qrels: Dict[str, List[str]],
    graph_nodes: Iterable[Dict],
) -> Dict:
    """Compute lightweight qrels integrity checks used by benchmark preflight."""
    q_by_id = {str(q.get("query_id")): q for q in queries if q.get("query_id")}
    node_index = _build_node_index(graph_nodes)

    mismatch_count = 0
    compared = 0
    missing_query = 0
    missing_artifact = 0
    for qid, rel_ids in qrels.items():
        query = q_by_id.get(str(qid))
        if not query:
            missing_query += 1
            continue
        ref_kind, ref_number = _first_explicit_ref(query)
        if not ref_kind or not ref_number:
            continue
        for aid in rel_ids or []:
            mapped = node_index.get(aid)
            if not mapped:
                missing_artifact += 1
                continue
            compared += 1
            kind, number = mapped
            if kind != ref_kind or number != ref_number:
                mismatch_count += 1

    query_count = len(q_by_id)
    coverage = (len(qrels) / query_count) if query_count else 0.0
    return {
        "query_count": query_count,
        "qrels_count": len(qrels),
        "qrels_coverage": round(coverage, 6),
        "compared_explicit_pairs": compared,
        "mismatch_count": mismatch_count,
        "missing_query_count": missing_query,
        "missing_artifact_count": missing_artifact,
        "ambiguous_statement_key_count": _count_ambiguous_statement_keys(graph_nodes),
    }
