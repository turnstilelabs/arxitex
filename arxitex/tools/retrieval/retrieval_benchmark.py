#!/usr/bin/env python3
"""Benchmark retrieval baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from loguru import logger

from arxitex.tools.mentions.mapping.ref_artifact_mapper import (
    Policy as RefMappingPolicy,
)
from arxitex.tools.mentions.mapping.ref_artifact_mapper import (
    build_gold_links,
    build_target_registry,
)
from arxitex.tools.retrieval.agentic.service import run_agentic_search
from arxitex.tools.retrieval.bm25_engine import BM25Engine
from arxitex.tools.retrieval.colgrep_engine import ColGrepEngine
from arxitex.tools.retrieval.dense_engine import DenseEngine
from arxitex.tools.retrieval.io import (
    build_artifacts,
    load_graph,
    load_qrels,
    load_queries,
)
from arxitex.tools.retrieval.logic_decomposition import extract_logic_decompositions
from arxitex.tools.retrieval.logic_rerank import LogicReranker, apply_logic_rerank
from arxitex.tools.retrieval.metrics import evaluate
from arxitex.tools.retrieval.msc2020 import MSCDictionary, MSCMatch
from arxitex.tools.retrieval.normalization import normalize_text
from arxitex.tools.retrieval.pylate_engine import PyLateEngine
from arxitex.tools.retrieval.qrels_audit import audit_qrels_alignment
from arxitex.tools.retrieval.structured import StructuredFields, extract_structured

EXPERIMENTS = {
    "e1": "bm25",
    "e2": "dense",
    "e3": "pylate",
    "e4": "hybrid-rrf",
    "e5": "agentic-colgrep",
}

RRF_K = 60
PREFLIGHT_MIN_QRELS_COVERAGE = 0.99
PREFLIGHT_MAX_MISMATCHES = 0
PREFLIGHT_MAX_AMBIGUOUS_KEYS = 3

TYPE_HINTS = {
    "definition": ["definition", "def.", "defn", "defn."],
    "lemma": ["lemma", "lem.", "lem"],
    "theorem": ["theorem", "thm.", "thm"],
    "proposition": ["proposition", "prop.", "prop"],
    "corollary": ["corollary", "cor.", "cor"],
    "remark": ["remark", "rem.", "rem"],
    "example": ["example", "ex.", "ex"],
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_default_param(cls, name: str, fallback: str = "") -> str:
    try:
        sig = inspect.signature(cls.__init__)
        param = sig.parameters.get(name)
        if param and param.default is not inspect._empty:
            return str(param.default)
    except Exception:
        pass
    return fallback


def _math_verify_available() -> bool:
    try:
        from arxitex.tools.retrieval import normalization

        return normalization.mv_parse is not None
    except Exception:
        return False


def _read_queries(path: str) -> List[Dict]:
    return load_queries(path)


def _prepare_queries(
    rows: List[Dict],
    *,
    explicit_only: bool,
    normalize_mode: str,
    single_ref_only: bool,
    structured_map: Dict[str, StructuredFields] | None = None,
    structured_math_verify: bool = False,
) -> List[Dict]:
    prepared: List[Dict] = []
    for row in rows:
        if explicit_only:
            if "reference_precision" in row:
                precision = (row.get("reference_precision") or "").lower()
                if precision and precision != "explicit":
                    continue
            else:
                # Backward-compatible explicit-only filter when precision isn't present.
                if not (row.get("explicit_refs") or []):
                    continue
        if single_ref_only:
            refs = row.get("explicit_refs") or []
            if len(refs) != 1:
                continue
        query_text = row.get("query_text") or row.get("text") or ""
        if not query_text:
            continue
        query_id = row.get("query_id") or _hash(query_text)
        query_style = row.get("query_style") or "unknown"
        explicit_refs_count = len(row.get("explicit_refs") or [])
        query_len = len(query_text.split())
        structured = (structured_map or {}).get(query_id)
        if structured:
            parts = []
            if structured.math_terms:
                parts.append("[TERMS] " + " ; ".join(structured.math_terms))
            if structured.math_exprs:
                expr_text = " ; ".join(structured.math_exprs)
                if structured_math_verify:
                    expr_text = (
                        normalize_text(expr_text, use_math_verify=True) or expr_text
                    )
                parts.append("[EXPRS] " + expr_text)
            if structured.domain_terms:
                parts.append("[DOMAIN] " + " ; ".join(structured.domain_terms))
            parts.append(query_text)
            query_text_for_norm = " ".join(parts)
        else:
            query_text_for_norm = query_text

        if normalize_mode == "none":
            norm = query_text_for_norm
        elif normalize_mode == "unicode":
            norm = (
                normalize_text(query_text_for_norm, use_math_verify=False)
                or query_text_for_norm
            )
        else:
            norm = (
                normalize_text(query_text_for_norm, use_math_verify=True)
                or query_text_for_norm
            )
        if not norm.strip():
            norm = query_text_for_norm
        prepared.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "normalized": norm,
                "relevant_ids": row.get("relevant_ids") or [],
                "query_style": query_style,
                "explicit_refs_count": explicit_refs_count,
                "query_len": query_len,
                "explicit_refs": row.get("explicit_refs") or [],
                "reference_precision": row.get("reference_precision"),
                "target_arxiv_id": row.get("target_arxiv_id"),
                "source_arxiv_id": row.get("source_arxiv_id") or row.get("arxiv_id"),
                "bib_entry": row.get("bib_entry"),
                "target_match_status": row.get("target_match_status"),
            }
        )
    return prepared


def _write_results(path: str, results: Dict[str, Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in results.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_mapping_context_rows(rows: List[Dict]) -> List[Dict]:
    """Adapt retrieval query rows to the canonical context-row mapping contract."""
    out: List[Dict] = []
    for row in rows:
        context_id = row.get("query_id")
        if not context_id:
            continue
        out.append(
            {
                "context_id": context_id,
                "context_text": row.get("query_text") or "",
                "explicit_refs": row.get("explicit_refs") or [],
                "target_arxiv_id": row.get("target_arxiv_id"),
                "source_arxiv_id": row.get("source_arxiv_id") or row.get("arxiv_id"),
                "target_match_status": row.get("target_match_status"),
            }
        )
    return out


def _resolve_artifact_ids(indices: List, id_lookup: Dict[int, str]) -> List[str]:
    resolved: List[str] = []
    for idx in indices:
        if isinstance(idx, int):
            art_id = id_lookup.get(idx)
        else:
            art_id = idx
        if art_id:
            resolved.append(art_id)
    return resolved


_STRUCTURED_STOP = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "remark",
    "definition",
    "proof",
    "section",
    "chapter",
    "book",
    "paper",
    "notes",
    "lecture notes",
    "scholze",
}


def _normalize_structured_token(text: str) -> str:
    if not text:
        return ""
    # strip control chars and collapse whitespace
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    cleaned = " ".join(cleaned.split()).lower()
    # remove outer punctuation
    cleaned = cleaned.strip(".,;:()[]{}\"'`")
    return cleaned


def _structured_tokens(items: Iterable[str]) -> set:
    out = set()
    for raw in items or []:
        norm = _normalize_structured_token(raw)
        if not norm:
            continue
        if len(norm) < 3 or norm in _STRUCTURED_STOP:
            continue
        out.add(norm)
        # simplified variant for math-y strings
        simple = "".join(ch for ch in norm if ch.isalnum())
        if simple and simple != norm and len(simple) >= 3:
            out.add(simple)
    return out


def _structured_sets(fields: StructuredFields | None) -> Dict[str, set]:
    if not fields:
        return {"terms": set(), "exprs": set(), "domain": set()}
    return {
        "terms": _structured_tokens(fields.math_terms or []),
        "exprs": _structured_tokens(fields.math_exprs or []),
        "domain": _structured_tokens(fields.domain_terms or []),
    }


def _build_structured_idf(
    structured_artifacts: Dict[str, StructuredFields],
) -> Dict[str, float]:
    import math

    df: Dict[str, int] = {}
    n_docs = 0
    for fields in structured_artifacts.values():
        n_docs += 1
        aset = _structured_sets(fields)
        seen = set().union(aset["terms"], aset["exprs"], aset["domain"])
        for token in seen:
            df[token] = df.get(token, 0) + 1
    if n_docs == 0:
        return {}
    return {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _overlap_score(
    q: Dict[str, set], a: Dict[str, set], idf: Dict[str, float]
) -> float:
    def weight(tokens: set) -> float:
        return sum(idf.get(t, 1.0) for t in tokens)

    return (
        weight(q["terms"] & a["terms"])
        + 2.0 * weight(q["exprs"] & a["exprs"])
        + weight(q["domain"] & a["domain"])
    )


def _apply_structured_filter_boost(
    results: Dict[str, Dict],
    *,
    queries: List[Dict],
    structured_queries: Dict[str, StructuredFields],
    structured_artifacts: Dict[str, StructuredFields],
    id_lookup: Dict[int, str],
    do_filter: bool,
    min_overlap: int,
    boost: float,
) -> None:
    idf = _build_structured_idf(structured_artifacts)
    for qid, row in results.items():
        q_fields = structured_queries.get(qid)
        if not q_fields:
            continue
        qsets = _structured_sets(q_fields)
        indices = row.get("indices", [])
        scores = row.get("scores", [])
        if not indices:
            continue
        if not scores or len(scores) != len(indices):
            scores = [0.0] * len(indices)
        kept = []
        for idx, score in zip(indices, scores):
            art_id = id_lookup.get(idx) if isinstance(idx, int) else idx
            a_fields = structured_artifacts.get(art_id)
            asets = _structured_sets(a_fields)
            ov = _overlap_score(qsets, asets, idf)
            if boost:
                score = score + boost * ov
            if (not do_filter) or ov >= min_overlap:
                kept.append((idx, score))
        if kept:
            row["indices"] = [i for i, _ in kept]
            row["scores"] = [s for _, s in kept]
        else:
            # fallback: keep original (avoid empty results)
            row["indices"] = indices
            row["scores"] = scores


def _artifact_base_id(artifact_id: str) -> str:
    return artifact_id.replace("#proof", "")


def _infer_type_hint(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for t, needles in TYPE_HINTS.items():
        for n in needles:
            if n in lower:
                return t
    return None


def _rerank_by_type_hint(
    indices: List,
    scores: List,
    query_text: str,
    id_lookup: Dict[int, str],
    node_types: Dict[str, str],
) -> Tuple[List, List]:
    hint = _infer_type_hint(query_text)
    if not hint or not indices:
        return indices, scores
    paired = list(zip(indices, scores))
    matches = []
    non_matches = []
    for idx, score in paired:
        art_id = id_lookup.get(idx) if isinstance(idx, int) else idx
        if not art_id:
            non_matches.append((idx, score))
            continue
        base_id = _artifact_base_id(art_id)
        node_type = (node_types.get(base_id) or "").lower()
        if node_type == hint:
            matches.append((idx, score))
        else:
            non_matches.append((idx, score))
    if not matches:
        return indices, scores
    reordered = matches + non_matches
    return [i for i, _ in reordered], [s for _, s in reordered]


def _first_rel_rank(retrieved: List[str], rel_set: Iterable[str]) -> int | None:
    rel = set(rel_set)
    if not rel:
        return None
    for i, doc_id in enumerate(retrieved):
        if doc_id in rel:
            return i + 1
    return None


def _hit_at_k(retrieved: List[str], rel_set: Iterable[str], k: int) -> float:
    rel = set(rel_set)
    if not rel:
        return 0.0
    return 1.0 if any(doc_id in rel for doc_id in retrieved[:k]) else 0.0


def _bucket_query_len(n: int) -> str:
    if n <= 6:
        return "0-6"
    if n <= 12:
        return "7-12"
    if n <= 20:
        return "13-20"
    return "21+"


def _bucket_refs(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    return "2+"


def _rrf_fuse(
    *,
    result_sets: List[Dict[str, Dict]],
    queries: List[Dict],
    id_lookup: Dict[int, str],
    idx_lookup: Dict[str, int],
    k: int,
    rrf_k: int = RRF_K,
) -> Dict[str, Dict]:
    fused: Dict[str, Dict] = {}
    for row in queries:
        qid = row["query_id"]
        scores: Dict[str, float] = {}
        for results in result_sets:
            base = results.get(qid) or {}
            indices = base.get("indices") or []
            for rank, idx in enumerate(indices):
                art_id = id_lookup.get(idx) if isinstance(idx, int) else idx
                if not art_id:
                    continue
                scores[art_id] = scores.get(art_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        fused_ids = [art_id for art_id, _ in ranked]
        fused_indices = [idx_lookup.get(art_id) for art_id in fused_ids]
        fused_scores = [score for _, score in ranked]
        fused[qid] = {
            "query_id": qid,
            "query_text": row["query_text"],
            "indices": [idx for idx in fused_indices if idx is not None],
            "scores": fused_scores,
        }
    return fused


def _compute_per_query_metrics(
    *,
    exp: str,
    results: Dict[str, Dict],
    queries: List[Dict],
    qrels: Dict[str, List[str]],
    k: int,
    use_expanded: bool = False,
) -> List[Dict]:
    rows: List[Dict] = []
    for q in queries:
        qid = q["query_id"]
        rel_ids = qrels.get(qid) or []
        rel_set = set(rel_ids)
        res = results.get(qid) or {}
        if use_expanded:
            retrieved = res.get("expanded_ids") or res.get("artifact_ids") or []
        else:
            retrieved = res.get("artifact_ids") or []
        evaluated = bool(rel_set)
        hit = _hit_at_k(retrieved, rel_set, k) if evaluated else None
        first_rank = _first_rel_rank(retrieved, rel_set) if evaluated else None
        rows.append(
            {
                "query_id": qid,
                "method": exp,
                "query_style": q.get("query_style") or "unknown",
                "query_len": q.get("query_len") or 0,
                "explicit_refs_count": q.get("explicit_refs_count") or 0,
                "qrels_count": len(rel_set),
                "hit@10": hit,
                "first_rel_rank": first_rank,
                "evaluated": evaluated,
            }
        )
    return rows


def _write_per_query_metrics(
    out_dir: str, exp: str, rows: List[Dict]
) -> Dict[str, str]:
    jsonl_path = os.path.join(out_dir, f"per_query_metrics_{exp}.jsonl")
    csv_path = os.path.join(out_dir, f"per_query_metrics_{exp}.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        cols = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return {"jsonl": jsonl_path, "csv": csv_path}


def _stratify_metrics(rows: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    def agg(key_fn):
        buckets: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for row in rows:
            if not row.get("evaluated"):
                continue
            key = key_fn(row)
            buckets.setdefault(key, []).append(float(row.get("hit@10") or 0.0))
            counts[key] = counts.get(key, 0) + 1
        out = {}
        for key, vals in buckets.items():
            out[key] = {
                "count": counts.get(key, 0),
                "hit@10": (sum(vals) / len(vals)) if vals else 0.0,
            }
        return out

    return {
        "by_query_style": agg(lambda r: r.get("query_style") or "unknown"),
        "by_query_len": agg(lambda r: _bucket_query_len(int(r.get("query_len") or 0))),
        "by_explicit_refs": agg(
            lambda r: _bucket_refs(int(r.get("explicit_refs_count") or 0))
        ),
    }


def run_bm25(
    texts: List[str], queries: List[Dict], k: int, *, k1: float | None, b: float | None
) -> Dict[str, Dict]:
    engine = BM25Engine(k1=k1, b=b)
    engine.build(texts)
    results = {}
    for row in queries:
        idxs, scores = engine.search(row["normalized"], k=k)
        results[row["query_id"]] = {
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "indices": idxs,
            "scores": scores,
        }
    return results


def run_dense(
    texts: List[str], ids: List[str], queries: List[Dict], k: int, cache_dir: str
) -> Dict[str, Dict]:
    engine = DenseEngine(cache_dir=cache_dir)
    engine.build(texts, ids)
    results = {}
    for row in queries:
        idxs, scores = engine.search(row["normalized"], k=k)
        results[row["query_id"]] = {
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "indices": idxs,
            "scores": scores,
        }
    return results


def run_pylate(
    texts: List[str],
    ids: List[str],
    queries: List[Dict],
    k: int,
    index_dir: str,
    model_name: str | None = None,
) -> Tuple[Dict[str, Dict], float]:
    if model_name:
        engine = PyLateEngine(model_name=model_name, index_dir=index_dir)
    else:
        engine = PyLateEngine(index_dir=index_dir)
    logger.info("PyLate input ids count: {}", len(ids))
    engine.build(texts, ids)
    if hasattr(engine, "_ids") and engine._ids is not None:
        logger.info("PyLate ids count: {}", len(engine._ids))
    latency = None
    results = {}
    for i, row in enumerate(queries):
        if i == 0:
            latency = engine.latency_probe(row["normalized"], k=k)
        idxs, scores = engine.search(row["normalized"], k=k)
        if not idxs and row["query_text"] and row["query_text"] != row["normalized"]:
            idxs, scores = engine.search(row["query_text"], k=k)
        if i == 0:
            logger.info("PyLate first-query results: {} hits", len(idxs))
        results[row["query_id"]] = {
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "indices": idxs,
            "scores": scores,
        }
    return results, latency or 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval baselines for SGA artifacts."
    )
    parser.add_argument(
        "--graph", default="public/data/sga4-5.json", help="Graph JSON path."
    )
    parser.add_argument(
        "--queries",
        default="data/citation_dataset/queries.jsonl",
        help="Queries JSONL path.",
    )
    parser.add_argument(
        "--qrels",
        "--gold-links",
        dest="qrels",
        default="data/citation_dataset/qrels.json",
        help="Optional gold-links JSON/JSONL path (qrels format).",
    )
    parser.add_argument(
        "--mapping-curated-aliases",
        default="",
        help="Optional curated alias JSON used by auto gold-link mapping.",
    )
    parser.add_argument("--out-dir", default="data/retrieval", help="Output directory.")
    parser.add_argument("--experiment", default="all", help="e1,e2,e3,e4,e5,all")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-K results per query."
    )
    parser.add_argument(
        "--pylate-model", default=None, help="PyLate model name or local path."
    )
    parser.add_argument(
        "--pylate-index-dir",
        default="",
        help="Directory to store PyLate index files (default: out-dir).",
    )
    parser.add_argument(
        "--index-mode",
        default="content",
        help="Index text variant: content, content+prereqs, content+semantic, content+all",
    )
    parser.add_argument(
        "--normalize-mode",
        default="auto",
        help="Normalization: auto (unicode + math_verify if available), unicode (no math_verify), none",
    )
    parser.add_argument(
        "--no-type-prefix",
        action="store_false",
        dest="include_type_prefix",
        help="Disable [TYPE] prefix in indexed text.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=0,
        help="Limit to first N queries after filtering (0 = no limit).",
    )
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Disable filtering to explicit-only queries (default: filter when reference_precision is present).",
    )
    parser.add_argument(
        "--no-proofs",
        action="store_false",
        dest="include_proofs",
        help="Disable indexing proofs as artifacts.",
    )
    parser.add_argument(
        "--single-ref-only",
        action="store_true",
        help="Only keep queries with exactly one explicit reference.",
    )
    parser.add_argument(
        "--single-source-only-metrics",
        action="store_true",
        help="Compute metrics only on queries whose gold links resolve to exactly one artifact.",
    )
    parser.add_argument(
        "--bm25-k1", type=float, default=None, help="BM25 k1 parameter override."
    )
    parser.add_argument(
        "--bm25-b", type=float, default=None, help="BM25 b parameter override."
    )
    parser.add_argument(
        "--rrf-include-e2",
        action="store_true",
        help="Include dense results in e4 RRF fusion (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Enable structured extraction for queries/artifacts (LLM).",
    )
    parser.add_argument(
        "--structured-prepend",
        action="store_true",
        help="Prepend structured fields to query/artifact text.",
    )
    parser.add_argument(
        "--structured-model",
        default="gpt-5-mini-2025-08-07",
        help="LLM model for structured extraction.",
    )
    parser.add_argument(
        "--structured-cache",
        default="",
        help="Cache directory for structured extraction (default: <out-dir>/structured_cache).",
    )
    parser.add_argument(
        "--structured-math-verify",
        action="store_true",
        help="Apply math_verify to extracted math expressions only.",
    )
    parser.add_argument(
        "--structured-concurrency",
        type=int,
        default=4,
        help="Parallelism for structured extraction (default: 4).",
    )
    parser.add_argument(
        "--structured-filter",
        action="store_true",
        help="Filter results to those overlapping structured fields.",
    )
    parser.add_argument(
        "--structured-min-overlap",
        type=int,
        default=1,
        help="Minimum overlap count for structured filtering (default: 1).",
    )
    parser.add_argument(
        "--structured-boost",
        type=float,
        default=0.0,
        help="Additive score boost per overlap count (default: 0.0).",
    )
    parser.add_argument(
        "--type-rerank",
        action="store_true",
        help="Re-rank results by query type hints (definition/lemma/theorem/etc).",
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Directory for log files (default: <out-dir>/logs).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run identifier for logs/metadata (default: UTC timestamp).",
    )
    parser.add_argument(
        "--logic-rerank",
        action="store_true",
        help="Enable logic-aware reranking on top of retrieval outputs.",
    )
    parser.add_argument(
        "--logic-model",
        default="gpt-5-mini-2025-08-07",
        help="LLM model for logic decomposition + pairwise entailment labels.",
    )
    parser.add_argument(
        "--logic-cache",
        default="",
        help="Cache directory for logic decomposition (default: <out-dir>/logic_cache).",
    )
    parser.add_argument(
        "--logic-concurrency",
        type=int,
        default=4,
        help="Parallelism for logic decomposition and hypothesis labeling.",
    )
    parser.add_argument(
        "--logic-top-n",
        type=int,
        default=20,
        help="Apply logic reranking only to top N candidates per query (default: 20).",
    )
    parser.add_argument(
        "--logic-msc-csv",
        default="data/msc2020/msc2020.csv",
        help="Path to MSC2020 CSV file used for symbolic context triage.",
    )
    parser.add_argument(
        "--logic-no-debruijn",
        action="store_false",
        dest="logic_use_debruijn",
        help="Disable De Bruijn-style fallback normalization in goal scoring.",
    )
    parser.add_argument(
        "--agentic-model",
        default="gpt-5-mini-2025-08-07",
        help="LLM model for agentic ColGREP retrieval.",
    )
    parser.add_argument(
        "--agentic-max-steps",
        type=int,
        default=3,
        help="Max agentic refinement steps (default: 3).",
    )
    parser.add_argument(
        "--agentic-top-k",
        type=int,
        default=10,
        help="Top-K ColGREP candidates per step (default: 10).",
    )
    parser.add_argument(
        "--colgrep-bin",
        default="colgrep",
        help="ColGREP binary path (default: colgrep).",
    )
    parser.add_argument(
        "--colgrep-index-dir",
        default="",
        help="ColGREP index directory (optional).",
    )
    parser.add_argument(
        "--colgrep-chunks-dir",
        default="",
        help="ColGREP chunk directory (defaults to --colgrep-index-dir).",
    )
    parser.add_argument(
        "--preflight-report",
        default="",
        help="Optional path to write gold-links preflight JSON (default: <out-dir>/qrels_preflight.json).",
    )
    parser.set_defaults(include_proofs=True, include_type_prefix=True)
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    run_id = args.run_id or _default_run_id()
    log_dir = args.log_dir or os.path.join(args.out_dir, "logs")
    _ensure_dir(log_dir)

    graph = load_graph(args.graph)
    raw_queries = _read_queries(args.queries)

    structured_cache_dir = (
        Path(args.structured_cache)
        if args.structured_cache
        else Path(args.out_dir) / "structured_cache"
    )
    structured_artifacts: Dict[str, StructuredFields] = {}
    structured_queries: Dict[str, StructuredFields] = {}
    if args.structured:
        # Artifact structured extraction
        artifact_texts = []
        artifact_ids = []
        for node_id, node in graph.nodes.items():
            content = node.get("content") or ""
            if content.strip():
                artifact_ids.append(node_id)
                artifact_texts.append(content)
            proof = node.get("proof") or ""
            if args.include_proofs and proof.strip():
                artifact_ids.append(f"{node_id}#proof")
                artifact_texts.append(proof)
        structured_artifacts = extract_structured(
            texts=artifact_texts,
            ids=artifact_ids,
            model=args.structured_model,
            cache_path=structured_cache_dir / "artifacts.jsonl",
            concurrency=args.structured_concurrency,
        )

        # Query structured extraction
        query_ids = [
            r.get("query_id") or _hash(r.get("query_text") or "") for r in raw_queries
        ]
        query_texts = [r.get("query_text") or "" for r in raw_queries]
        structured_queries = extract_structured(
            texts=query_texts,
            ids=query_ids,
            model=args.structured_model,
            cache_path=structured_cache_dir / "queries.jsonl",
            concurrency=args.structured_concurrency,
        )

    artifacts = build_artifacts(
        graph,
        include_proofs=args.include_proofs,
        index_mode=args.index_mode,
        include_type_prefix=args.include_type_prefix,
        normalize_mode=args.normalize_mode,
        structured_map=(
            {k: v.model_dump() for k, v in structured_artifacts.items()}
            if args.structured and args.structured_prepend
            else None
        ),
        structured_math_verify=args.structured_math_verify,
    )
    texts = [a.index_text for a in artifacts]
    ids = [a.artifact_id for a in artifacts]
    id_lookup = {idx: art_id for idx, art_id in enumerate(ids)}
    idx_lookup = {art_id: idx for idx, art_id in enumerate(ids)}
    node_types = {node_id: node.get("type") for node_id, node in graph.nodes.items()}

    has_precision = any("reference_precision" in row for row in raw_queries)
    explicit_only = has_precision and not args.all_queries
    queries = _prepare_queries(
        raw_queries,
        explicit_only=explicit_only,
        normalize_mode=args.normalize_mode,
        single_ref_only=args.single_ref_only,
        structured_map=(
            structured_queries if args.structured and args.structured_prepend else None
        ),
        structured_math_verify=args.structured_math_verify,
    )
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    if has_precision:
        logger.info(
            "Query filter: {}",
            "explicit-only" if explicit_only else "all (explicit + implicit)",
        )
    if not queries:
        logger.error("No queries loaded from {}", args.queries)
        return 1

    logic_artifacts = {}
    logic_queries = {}
    query_msc: Dict[str, MSCMatch] = {}
    artifact_msc: Dict[str, MSCMatch] = {}
    logic_reranker = None
    if args.logic_rerank:
        logic_cache_dir = (
            Path(args.logic_cache)
            if args.logic_cache
            else Path(args.out_dir) / "logic_cache"
        )
        logic_artifacts = extract_logic_decompositions(
            texts=[a.text for a in artifacts],
            ids=ids,
            model=args.logic_model,
            cache_path=logic_cache_dir / "artifacts.jsonl",
            concurrency=args.logic_concurrency,
        )
        query_ids_for_logic = [q["query_id"] for q in queries]
        query_texts_for_logic = [q["query_text"] for q in queries]
        logic_queries = extract_logic_decompositions(
            texts=query_texts_for_logic,
            ids=query_ids_for_logic,
            model=args.logic_model,
            cache_path=logic_cache_dir / "queries.jsonl",
            concurrency=args.logic_concurrency,
        )

        msc_dict = None
        if os.path.exists(args.logic_msc_csv):
            msc_dict = MSCDictionary.from_csv(args.logic_msc_csv)
        else:
            logger.warning(
                "MSC2020 CSV not found at {}. Context score will be 0.0.",
                args.logic_msc_csv,
            )

        if msc_dict is not None:
            query_msc = {
                qid: msc_dict.match_context(
                    (logic_queries.get(qid) and logic_queries[qid].context) or ""
                )
                for qid in query_ids_for_logic
            }
            artifact_msc = {
                aid: msc_dict.match_context(
                    (logic_artifacts.get(aid) and logic_artifacts[aid].context) or ""
                )
                for aid in ids
            }

        logic_reranker = LogicReranker(
            model=args.logic_model,
            top_n=args.logic_top_n,
            concurrency=args.logic_concurrency,
            use_debruijn=args.logic_use_debruijn,
        )

    qrels = load_qrels(args.qrels) if args.qrels else {}
    qrel_mapping_diagnostics = None
    qrel_mapping_report_path = None
    if not qrels:
        # Build gold links (qrels format) from graph + explicit refs via shared resolver.
        registry = build_target_registry(
            graph.nodes.values(),
            default_version_id=(Path(args.graph).stem or "default"),
        )
        qrel_result = build_gold_links(
            _to_mapping_context_rows(queries),
            registry,
            policy=RefMappingPolicy(
                require_kind_match=True,
                allow_number_only_fallback=True,
                strict_target_match=True,
                drop_unknown_target=True,
                alias_curated_path=(args.mapping_curated_aliases or None),
            ),
        )
        qrels = qrel_result.gold_links
        qrel_mapping_diagnostics = {
            "total_rows": qrel_result.diagnostics.total_rows,
            "mapped_rows": qrel_result.diagnostics.mapped_rows,
            "dropped_rows": qrel_result.diagnostics.dropped_rows,
            "dropped_by_reason": qrel_result.diagnostics.dropped_by_reason,
            "mapped_by_tier": qrel_result.diagnostics.mapped_by_tier,
            "dropped_by_source": qrel_result.diagnostics.dropped_by_source,
            "kept_by_target_status": qrel_result.diagnostics.kept_by_target_status,
            "dropped_by_target_status": qrel_result.diagnostics.dropped_by_target_status,
            "alias_usage": qrel_result.diagnostics.alias_usage,
            "dropped_alias_reasons": qrel_result.diagnostics.dropped_alias_reasons,
        }
        qrel_mapping_report_path = os.path.join(
            args.out_dir, "qrels_mapping_report.jsonl"
        )
        with open(qrel_mapping_report_path, "w", encoding="utf-8") as mf:
            for rec in qrel_result.records:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if qrels:
            logger.info(
                "Built gold links from graph + explicit_refs ({} queries)", len(qrels)
            )
    if qrels:
        query_ids = {q["query_id"] for q in queries}
        qrels = {qid: rel for qid, rel in qrels.items() if qid in query_ids}
        if args.single_source_only_metrics:
            qrels = {qid: rel for qid, rel in qrels.items() if len(rel) == 1}

    qrels_preflight = audit_qrels_alignment(
        queries=queries,
        qrels=qrels,
        graph_nodes=graph.nodes.values(),
    )
    # Always persist audit diagnostics; optionally fail fast on threshold violations.
    qrels_preflight_path = args.preflight_report or os.path.join(
        args.out_dir, "qrels_preflight.json"
    )
    with open(qrels_preflight_path, "w", encoding="utf-8") as pf:
        json.dump(qrels_preflight, pf, ensure_ascii=False, indent=2)
    failures: List[str] = []
    coverage = float(qrels_preflight.get("qrels_coverage", 0.0))
    mismatch_count = int(qrels_preflight.get("mismatch_count", 0))
    ambiguous_keys = int(qrels_preflight.get("ambiguous_statement_key_count", 0))
    if coverage < PREFLIGHT_MIN_QRELS_COVERAGE:
        failures.append(f"qrels_coverage={coverage} < {PREFLIGHT_MIN_QRELS_COVERAGE}")
    if mismatch_count > PREFLIGHT_MAX_MISMATCHES:
        failures.append(f"mismatch_count={mismatch_count} > {PREFLIGHT_MAX_MISMATCHES}")
    if ambiguous_keys > PREFLIGHT_MAX_AMBIGUOUS_KEYS:
        failures.append(
            "ambiguous_statement_key_count="
            f"{ambiguous_keys} > {PREFLIGHT_MAX_AMBIGUOUS_KEYS}"
        )
    if failures:
        logger.error("Preflight failed: {}", "; ".join(failures))
        return 2

    pylate_index_dir = args.pylate_index_dir or args.out_dir

    metadata = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph_path": args.graph,
        "queries_path": args.queries,
        "qrels_path": args.qrels or None,
        "index_mode": args.index_mode,
        "normalize_mode": args.normalize_mode,
        "include_type_prefix": args.include_type_prefix,
        "include_proofs": args.include_proofs,
        "experiments": (
            args.experiment.lower()
            if args.experiment != "all"
            else ["e1", "e2", "e3", "e4", "e5"]
        ),
        "top_k": args.top_k,
        "query_counts": {
            "raw": len(raw_queries),
            "filtered": len(queries),
        },
        "qrels_coverage": len(qrels) if qrels else 0,
        "qrels_mapping": qrel_mapping_diagnostics,
        "qrels_mapping_report_path": qrel_mapping_report_path,
        "qrels_preflight": qrels_preflight,
        "qrels_preflight_path": qrels_preflight_path,
        "artifact_count": len(artifacts),
        "environment": {
            "python": sys.version.split()[0],
            "math_verify_available": _math_verify_available(),
        },
        "models": {
            "dense_model": _get_default_param(
                DenseEngine, "model", "text-embedding-3-small"
            ),
            "pylate_model": args.pylate_model
            or _get_default_param(
                PyLateEngine, "model_name", "lightonai/Reason-ModernColBERT"
            ),
            "pylate_index_dir": pylate_index_dir,
        },
        "logic_rerank": {
            "enabled": bool(args.logic_rerank),
            "model": args.logic_model,
            "top_n": args.logic_top_n,
            "concurrency": args.logic_concurrency,
            "msc_csv": args.logic_msc_csv,
            "use_debruijn": bool(args.logic_use_debruijn),
        },
        "agentic_colgrep": {
            "model": args.agentic_model,
            "max_steps": args.agentic_max_steps,
            "top_k": args.agentic_top_k,
            "colgrep_bin": args.colgrep_bin,
            "colgrep_index_dir": args.colgrep_index_dir or None,
            "colgrep_chunks_dir": args.colgrep_chunks_dir or None,
        },
    }
    with open(
        os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote run metadata to {}/run_metadata.json", args.out_dir)

    experiments = [args.experiment.lower()]
    if args.experiment.lower() == "all":
        experiments = ["e1", "e2", "e3", "e4", "e5"]

    metrics_summary = {}
    runtimes = {}
    per_query_paths: Dict[str, Dict[str, str]] = {}
    stratified_metrics: Dict[str, Dict] = {}
    results_cache: Dict[str, Dict[str, Dict]] = {}

    for exp in experiments:
        label = EXPERIMENTS.get(exp)
        if not label:
            logger.warning("Unknown experiment: {}", exp)
            continue

        log_path = os.path.join(log_dir, f"{run_id}_{exp}.log")
        sink_id = logger.add(
            log_path, level="INFO", enqueue=True, backtrace=False, diagnose=False
        )
        logger.info("Logging to {}", log_path)

        logger.info("Running {} ({})", exp, label)
        results = {}
        latency = None
        start = time.perf_counter()

        if exp == "e1":
            results = run_bm25(
                texts, queries, args.top_k, k1=args.bm25_k1, b=args.bm25_b
            )
            results_cache["e1"] = results
        elif exp == "e2":
            results = run_dense(texts, ids, queries, args.top_k, cache_dir=args.out_dir)
            results_cache["e2"] = results
        elif exp == "e3":
            results, latency = run_pylate(
                texts,
                ids,
                queries,
                args.top_k,
                index_dir=pylate_index_dir,
                model_name=args.pylate_model,
            )
            results_cache["e3"] = results
        elif exp == "e4":
            base_results: List[Dict[str, Dict]] = []
            if "e1" in results_cache:
                base_results.append(results_cache["e1"])
            else:
                base_results.append(
                    run_bm25(texts, queries, args.top_k, k1=args.bm25_k1, b=args.bm25_b)
                )

            if "e3" in results_cache:
                base_results.append(results_cache["e3"])
            else:
                e3_results, _ = run_pylate(
                    texts,
                    ids,
                    queries,
                    args.top_k,
                    index_dir=pylate_index_dir,
                    model_name=args.pylate_model,
                )
                base_results.append(e3_results)

            if args.rrf_include_e2:
                if "e2" in results_cache:
                    base_results.append(results_cache["e2"])
                else:
                    base_results.append(
                        run_dense(
                            texts, ids, queries, args.top_k, cache_dir=args.out_dir
                        )
                    )

            results = _rrf_fuse(
                result_sets=base_results,
                queries=queries,
                id_lookup=id_lookup,
                idx_lookup=idx_lookup,
                k=args.top_k,
            )
            results_cache["e4"] = results
        elif exp == "e5":
            chunks_dir = args.colgrep_chunks_dir or args.colgrep_index_dir
            if not chunks_dir:
                logger.error(
                    "--colgrep-chunks-dir or --colgrep-index-dir is required for e5"
                )
                logger.remove(sink_id)
                continue

            engine = ColGrepEngine(
                chunks_dir=chunks_dir,
                index_dir=args.colgrep_index_dir or None,
                colgrep_bin=args.colgrep_bin,
            )
            try:
                engine.build()
            except Exception as exc:
                logger.warning("ColGREP build failed: {}", exc)

            results = {}

            def _search_fn(q: str, k: int, *, _engine=engine):
                return [c.__dict__ for c in _engine.search_candidates(q, k=k)]

            for row in queries:
                mention = row.get("query_text") or ""
                agent_result = run_agentic_search(
                    mention=mention,
                    search_fn=_search_fn,
                    model=args.agentic_model,
                    max_steps=args.agentic_max_steps,
                    top_k=args.agentic_top_k,
                )
                ordered_ids = agent_result.candidates
                scored_pairs = list(zip(ordered_ids, agent_result.scores))
                filtered_ids = []
                indices = []
                scores = []
                for aid, score in scored_pairs:
                    if aid in idx_lookup:
                        filtered_ids.append(aid)
                        indices.append(idx_lookup[aid])
                        scores.append(score)
                results[row["query_id"]] = {
                    "query_id": row["query_id"],
                    "query_text": mention,
                    "indices": indices,
                    "scores": scores,
                    "artifact_ids": filtered_ids,
                    "selected_id": agent_result.selected_id,
                    "agent_trace": [t.__dict__ for t in agent_result.trace],
                }
            results_cache["e5"] = results
        else:
            logger.warning("Skipping unsupported experiment: {}", exp)
            logger.remove(sink_id)
            continue

        if (
            args.structured
            and (args.structured_filter or args.structured_boost)
            and exp in {"e1", "e2", "e3", "e4"}
        ):
            _apply_structured_filter_boost(
                results,
                queries=queries,
                structured_queries=structured_queries,
                structured_artifacts=structured_artifacts,
                id_lookup=id_lookup,
                do_filter=args.structured_filter,
                min_overlap=args.structured_min_overlap,
                boost=args.structured_boost,
            )

        if args.type_rerank and exp in {"e1", "e2", "e3", "e4"}:
            for row in results.values():
                indices = row.get("indices", [])
                scores = row.get("scores", [])
                if not indices:
                    continue
                new_indices, new_scores = _rerank_by_type_hint(
                    indices,
                    scores,
                    row.get("query_text") or "",
                    id_lookup,
                    node_types,
                )
                row["indices"] = new_indices
                row["scores"] = new_scores

        if args.logic_rerank and exp in {"e1", "e2", "e3", "e4"}:
            apply_logic_rerank(
                results=results,
                query_ids=[q["query_id"] for q in queries],
                id_lookup=id_lookup,
                query_logic=logic_queries,
                artifact_logic=logic_artifacts,
                query_msc=query_msc,
                artifact_msc=artifact_msc,
                reranker=logic_reranker,
            )

        # Map indices to ids
        for row in results.values():
            indices = row.get("indices", [])
            row["artifact_ids"] = _resolve_artifact_ids(indices, id_lookup)

        out_path = os.path.join(args.out_dir, f"{exp}_results.jsonl")
        _write_results(out_path, results)

        metrics = {}
        if qrels:
            target = {qid: row.get("artifact_ids", []) for qid, row in results.items()}
            metrics = evaluate(target, qrels, k=args.top_k)
        if latency is not None:
            metrics["pylate_latency_sec"] = latency

        metrics_summary[exp] = metrics
        runtimes[exp] = time.perf_counter() - start
        logger.info("{} metrics: {}", exp, metrics)
        logger.info("{} runtime_sec: {:.3f}", exp, runtimes[exp])
        logger.remove(sink_id)

        if qrels:
            per_query_rows = _compute_per_query_metrics(
                exp=exp,
                results=results,
                queries=queries,
                qrels=qrels,
                k=args.top_k,
                use_expanded=False,
            )
            per_query_paths[exp] = _write_per_query_metrics(
                args.out_dir, exp, per_query_rows
            )
            stratified_metrics[exp] = _stratify_metrics(per_query_rows)

    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics_summary,
            },
            f,
            indent=2,
        )
    logger.info("Wrote metrics summary to {}", summary_path)

    if stratified_metrics:
        strat_path = os.path.join(args.out_dir, "stratified_metrics.json")
        with open(strat_path, "w", encoding="utf-8") as f:
            json.dump(stratified_metrics, f, indent=2)
        logger.info("Wrote stratified metrics to {}", strat_path)
    else:
        strat_path = None

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_summary_path": summary_path,
        "metrics": metrics_summary,
        "runtimes_sec": runtimes,
        "artifact_count": len(artifacts),
        "per_query_metrics": per_query_paths,
        "stratified_metrics_path": strat_path,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to {}/summary.json", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
