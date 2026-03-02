"""BM25 lexical baseline using bm25s."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from loguru import logger

from .tokenizer import tokenize_latex


class BM25Engine:
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] | None = None,
        *,
        k1: float | None = None,
        b: float | None = None,
    ):
        self.tokenizer = tokenizer or tokenize_latex
        try:
            import bm25s
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("bm25s is required for BM25Engine") from exc
        self._bm25 = bm25s.BM25(method="lucene")
        if k1 is not None and hasattr(self._bm25, "k1"):
            self._bm25.k1 = k1
        if b is not None and hasattr(self._bm25, "b"):
            self._bm25.b = b
        self._built = False

    def build(self, texts: Sequence[str]) -> None:
        try:
            self._bm25.index(texts, tokenizer=self.tokenizer)
        except TypeError:
            tokenized = [self.tokenizer(t) for t in texts]
            self._bm25.index(tokenized)
        self._built = True
        logger.info("BM25 index built for {} documents", len(texts))

    def search(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        if not self._built:
            raise RuntimeError("BM25 index not built")

        tokenized = self.tokenizer(query)
        candidates = [
            ("search", {"query": query, "k": k}),
            ("search", {"query": tokenized, "k": k}),
            ("retrieve", {"query": query, "k": k}),
            ("retrieve", {"query": tokenized, "k": k}),
            (
                "retrieve",
                {
                    "query_tokens": [tokenized],
                    "k": k,
                    "return_as": "tuple",
                    "show_progress": False,
                },
            ),
            ("get_top_n", {"query": query, "n": k}),
            ("get_top_n", {"query": tokenized, "n": k}),
        ]

        for method, kwargs in candidates:
            if not hasattr(self._bm25, method):
                continue
            try:
                result = getattr(self._bm25, method)(**kwargs)
            except TypeError:
                try:
                    result = getattr(self._bm25, method)(*kwargs.values())
                except Exception:
                    continue
            indices, scores = _normalize_bm25_result(result, k)
            if indices:
                return indices, scores
        raise RuntimeError("Unable to query bm25s index with available methods")


def _normalize_bm25_result(result, k: int) -> Tuple[List[int], List[float]]:
    if result is None:
        return [], []

    if hasattr(result, "documents") and hasattr(result, "scores"):
        indices = result.documents
        scores = result.scores
        # bm25s returns shape (n_queries, k)
        if (
            hasattr(indices, "__len__")
            and len(indices)
            and hasattr(indices[0], "__len__")
        ):
            indices = indices[0]
        if hasattr(scores, "__len__") and len(scores) and hasattr(scores[0], "__len__"):
            scores = scores[0]
        idx_list = [int(x) for x in list(indices)[:k]]
        score_list = [float(x) for x in list(scores)[:k]]
        return idx_list, score_list

    if isinstance(result, tuple) and len(result) == 2:
        scores, indices = result
        if scores and isinstance(scores[0], (list, tuple)):
            scores = scores[0]
        if indices and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        idx_list = [int(x) for x in list(indices)[:k]]
        score_list = [float(x) for x in list(scores)[:k]]
        return idx_list, score_list

    if isinstance(result, dict):
        indices = result.get("doc_ids") or result.get("indices") or []
        scores = result.get("scores") or []
        idx_list = [int(x) for x in list(indices)[:k]]
        score_list = [float(x) for x in list(scores)[:k]]
        return idx_list, score_list

    if isinstance(result, list):
        return list(result)[:k], [1.0] * min(k, len(result))

    return [], []
