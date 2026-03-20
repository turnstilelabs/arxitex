#!/usr/bin/env python3
"""Evaluate bi-encoder retrieval on mention dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependencies. Install sentence-transformers and torch."
    ) from exc

from arxitex.tools.retrieval.metrics import evaluate


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate bi-encoder retrieval.")
    parser.add_argument("--model", required=True, help="Model path or name.")
    parser.add_argument("--queries", required=True, help="Queries jsonl path.")
    parser.add_argument("--qrels", required=True, help="Qrels json path.")
    parser.add_argument("--statements", required=True, help="Statements jsonl path.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default=None, help="Metrics output path.")
    args = parser.parse_args()

    queries = _load_jsonl(Path(args.queries))
    statements = _load_jsonl(Path(args.statements))
    qrels = json.loads(Path(args.qrels).read_text(encoding="utf-8"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)

    stmt_ids = [s["statement_id"] for s in statements]
    stmt_texts = [s["statement_text"] for s in statements]
    stmt_arxiv = [s.get("arxiv_id") for s in statements]
    stmt_emb = model.encode(stmt_texts, convert_to_numpy=True, show_progress_bar=True)
    stmt_emb = stmt_emb / np.linalg.norm(stmt_emb, axis=1, keepdims=True)

    by_arxiv: Dict[str, List[int]] = {}
    for idx, aid in enumerate(stmt_arxiv):
        if not aid:
            continue
        by_arxiv.setdefault(aid, []).append(idx)

    results: Dict[str, List[str]] = {}
    for q in queries:
        qid = q["query_id"]
        text = q["query_text"]
        q_emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        candidate_idx = None
        target_arxiv = q.get("target_arxiv_id")
        if target_arxiv:
            candidate_idx = by_arxiv.get(target_arxiv, [])
        if candidate_idx:
            scores = np.dot(stmt_emb[candidate_idx], q_emb[0])
            top_local = np.argsort(-scores)[: args.k]
            results[qid] = [stmt_ids[candidate_idx[i]] for i in top_local]
        else:
            scores = np.dot(stmt_emb, q_emb[0])
            top_idx = np.argsort(-scores)[: args.k]
            results[qid] = [stmt_ids[i] for i in top_idx]

    metrics = evaluate(results, qrels, k=args.k)
    logger.info("Metrics: {}", metrics)
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
