"""Ranking metrics for retrieval experiments."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


def ndcg_at_k(retrieved: List[str], relevant: Iterable[str], k: int = 10) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in rel_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def recall_at_k(retrieved: List[str], relevant: Iterable[str], k: int = 10) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in rel_set)
    return hits / len(rel_set)


def hit_rate_at_k(retrieved: List[str], relevant: Iterable[str], k: int = 10) -> float:
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    for doc_id in retrieved[:k]:
        if doc_id in rel_set:
            return 1.0
    return 0.0


def evaluate(
    results: Dict[str, List[str]], qrels: Dict[str, List[str]], k: int = 10
) -> Dict[str, float]:
    ndcgs = []
    recalls = []
    hits = []
    for qid, retrieved in results.items():
        relevant = qrels.get(qid)
        if not relevant:
            continue
        ndcgs.append(ndcg_at_k(retrieved, relevant, k=k))
        recalls.append(recall_at_k(retrieved, relevant, k=k))
        hits.append(hit_rate_at_k(retrieved, relevant, k=k))
    if not ndcgs:
        return {"nDCG@%d" % k: 0.0, "Recall@%d" % k: 0.0, "Hit@%d" % k: 0.0}
    return {
        "nDCG@%d" % k: sum(ndcgs) / len(ndcgs),
        "Recall@%d" % k: sum(recalls) / len(recalls),
        "Hit@%d" % k: sum(hits) / len(hits),
    }
