#!/usr/bin/env python3
"""Train a bi-encoder for mention -> statement retrieval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

try:
    import numpy as np
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Missing dependencies. Install sentence-transformers and torch."
    ) from exc

from arxitex.tools.retrieval.metrics import evaluate


def _load_statement_pool(path: Path) -> Dict[str, List[Tuple[str, str]]]:
    pool: Dict[str, List[Tuple[str, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arxiv_id = row.get("arxiv_id")
        sid = row.get("statement_id")
        text = row.get("statement_text")
        if not arxiv_id or not sid or not text:
            continue
        pool.setdefault(arxiv_id, []).append((sid, text))
    return pool


def _load_pairs(
    path: Path,
    *,
    negatives_per_query: int = 0,
    statement_pool: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    seed: int = 13,
) -> List[InputExample]:
    rng = random.Random(seed)
    examples: List[InputExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        q = row.get("query_text") or ""
        s = row.get("target_statement_text") or ""
        arxiv_id = row.get("target_arxiv_id")
        target_id = row.get("target_statement_id")
        if not q or not s:
            continue
        texts = [q, s]
        if negatives_per_query > 0 and statement_pool and arxiv_id and target_id:
            candidates = [
                (sid, text)
                for sid, text in statement_pool.get(arxiv_id, [])
                if sid != target_id and text
            ]
            if candidates:
                rng.shuffle(candidates)
                for _, neg_text in candidates[:negatives_per_query]:
                    texts.append(neg_text)
        examples.append(InputExample(texts=texts))
    return examples


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _estimate_train_loss(
    model: SentenceTransformer,
    loss_fn: losses.MultipleNegativesRankingLoss,
    train_loader: DataLoader,
    max_batches: int,
) -> float:
    if max_batches <= 0:
        return 0.0
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch in train_loader:
            sentence_features, labels = model.smart_batching_collate(batch)
            loss_value = loss_fn(sentence_features, labels)
            total += float(loss_value.item())
            seen += 1
            if seen >= max_batches:
                break
    model.train()
    return total / max(seen, 1)


def _evaluate_model(
    model: SentenceTransformer,
    queries: List[Dict],
    qrels: Dict[str, List[str]],
    statements: List[Dict],
    k: int,
) -> Dict[str, float]:
    stmt_ids = [s["statement_id"] for s in statements]
    stmt_texts = [s["statement_text"] for s in statements]
    stmt_arxiv = [s.get("arxiv_id") for s in statements]
    stmt_emb = model.encode(stmt_texts, convert_to_numpy=True, show_progress_bar=False)
    stmt_emb = stmt_emb / (np.linalg.norm(stmt_emb, axis=1, keepdims=True) + 1e-9)

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
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        candidate_idx = None
        target_arxiv = q.get("target_arxiv_id")
        if target_arxiv:
            candidate_idx = by_arxiv.get(target_arxiv, [])
        if candidate_idx:
            scores = np.dot(stmt_emb[candidate_idx], q_emb[0])
            top_local = np.argsort(-scores)[:k]
            results[qid] = [stmt_ids[candidate_idx[i]] for i in top_local]
        else:
            scores = np.dot(stmt_emb, q_emb[0])
            top_idx = np.argsort(-scores)[:k]
            results[qid] = [stmt_ids[i] for i in top_idx]

    return evaluate(results, qrels, k=k)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train bi-encoder on mention pairs.")
    parser.add_argument("--train", required=True, help="Train jsonl path.")
    parser.add_argument("--val", required=False, help="Val jsonl path.")
    parser.add_argument(
        "--model",
        default="intfloat/e5-small-v2",
        help="Base sentence-transformers model.",
    )
    parser.add_argument("--out-dir", default="data/models/biencoder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--train-loss-batches",
        type=int,
        default=10,
        help="Number of train batches to estimate loss per epoch (0 disables).",
    )
    parser.add_argument("--eval-queries", default="", help="Eval queries jsonl path.")
    parser.add_argument("--eval-qrels", default="", help="Eval qrels json path.")
    parser.add_argument("--eval-statements", default="", help="Statements jsonl path.")
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument(
        "--eval-log",
        default="",
        help="Optional JSONL log path for per-epoch eval metrics.",
    )
    parser.add_argument(
        "--statements",
        default="",
        help="Statements jsonl path (required for same-paper negatives).",
    )
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=0,
        help="Number of same-paper negatives to add per query (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    train_path = Path(args.train)
    statement_pool = None
    if args.negatives_per_query > 0:
        if not args.statements:
            raise RuntimeError("--statements is required for same-paper negatives.")
        statement_pool = _load_statement_pool(Path(args.statements))
        logger.info(
            "Loaded statement pool for {} papers",
            len(statement_pool),
        )

    train_examples = _load_pairs(
        train_path,
        negatives_per_query=args.negatives_per_query,
        statement_pool=statement_pool,
        seed=args.seed,
    )
    if not train_examples:
        raise RuntimeError("No training pairs found.")
    logger.info("Loaded {} train pairs", len(train_examples))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    loss_loader = DataLoader(
        train_examples,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=lambda batch: batch,
    )

    eval_queries: Optional[List[Dict]] = None
    eval_qrels: Optional[Dict[str, List[str]]] = None
    eval_statements: Optional[List[Dict]] = None
    if args.eval_queries and args.eval_qrels and args.eval_statements:
        eval_queries = _load_jsonl(Path(args.eval_queries))
        eval_qrels = json.loads(Path(args.eval_qrels).read_text(encoding="utf-8"))
        eval_statements = _load_jsonl(Path(args.eval_statements))
        logger.info(
            "Loaded eval set: {} queries, {} statements",
            len(eval_queries),
            len(eval_statements),
        )

    for epoch in range(1, args.epochs + 1):
        warmup = max(10, int(len(train_loader) * 0.1))
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            warmup_steps=warmup,
            optimizer_params={"lr": args.lr},
            output_path=args.out_dir,
        )

        train_loss_value = None
        if args.train_loss_batches > 0:
            train_loss_value = _estimate_train_loss(
                model,
                train_loss,
                loss_loader,
                max_batches=args.train_loss_batches,
            )
            logger.info(
                "Epoch {} train loss (avg over {} batches): {}",
                epoch,
                args.train_loss_batches,
                train_loss_value,
            )

        if eval_queries is not None and eval_qrels is not None and eval_statements:
            metrics = _evaluate_model(
                model,
                eval_queries,
                eval_qrels,
                eval_statements,
                k=args.eval_k,
            )
            logger.info("Epoch {} eval metrics: {}", epoch, metrics)
            if args.eval_log:
                with Path(args.eval_log).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "train_loss": train_loss_value,
                                "metrics": metrics,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    logger.info("Saved bi-encoder to {}", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
