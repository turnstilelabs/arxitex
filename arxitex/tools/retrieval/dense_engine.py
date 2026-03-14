"""Dense embedding baseline using OpenAI embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from loguru import logger


class DenseEngine:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: str | None = None,
        max_chars: int = 4000,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_chars = max_chars
        self.embeddings: np.ndarray | None = None
        self.ids: List[str] = []
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai is required for DenseEngine") from exc
        self._client = OpenAI()

    def build(self, texts: Sequence[str], ids: Sequence[str]) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            emb_path = self.cache_dir / f"embeddings_{self._safe_model_name()}.npy"
            id_path = self.cache_dir / f"embeddings_{self._safe_model_name()}.json"
            if emb_path.exists() and id_path.exists():
                with id_path.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, dict):
                    cached_ids = cached.get("ids") or []
                    cached_hashes = cached.get("text_hashes") or []
                else:
                    cached_ids = cached
                    cached_hashes = []
                current_hashes = self._hash_texts(texts) if cached_hashes else []
                if cached_ids == list(ids) and (
                    not cached_hashes or cached_hashes == current_hashes
                ):
                    self.embeddings = np.load(emb_path)
                    self.ids = list(ids)
                    logger.info("Loaded cached embeddings from {}", emb_path)
                    return

        vectors = self._embed_texts(texts)
        self.embeddings = np.array(vectors, dtype=np.float32)
        self.ids = list(ids)
        if self.cache_dir:
            emb_path = self.cache_dir / f"embeddings_{self._safe_model_name()}.npy"
            id_path = self.cache_dir / f"embeddings_{self._safe_model_name()}.json"
            np.save(emb_path, self.embeddings)
            with id_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ids": self.ids,
                        "text_hashes": self._hash_texts(texts),
                    },
                    f,
                )
            logger.info("Cached embeddings to {}", emb_path)

    def search(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        if self.embeddings is None:
            raise RuntimeError("Dense embeddings not built")
        query_vec = self._embed_texts([query])[0]
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        docs = self.embeddings
        docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
        scores = docs_norm @ q
        top_idx = np.argsort(-scores)[:k]
        return top_idx.tolist(), scores[top_idx].tolist()

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        batch_size = 64
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = []
            truncated = 0
            for t in texts[i : i + batch_size]:
                if self.max_chars and len(t) > self.max_chars:
                    batch.append(t[: self.max_chars])
                    truncated += 1
                else:
                    batch.append(t)
            resp = self._client.embeddings.create(model=self.model, input=batch)
            for item in resp.data:
                vectors.append(item.embedding)
            if truncated:
                logger.info(
                    "Truncated {} / {} texts to {} chars",
                    truncated,
                    len(batch),
                    self.max_chars,
                )
            logger.info(
                "Embedded {} / {} texts", min(i + batch_size, len(texts)), len(texts)
            )
        return vectors

    def _safe_model_name(self) -> str:
        return self.model.replace("/", "-")

    def _hash_texts(self, texts: Sequence[str]) -> List[str]:
        hashes: List[str] = []
        for t in texts:
            if self.max_chars and len(t) > self.max_chars:
                t = t[: self.max_chars]
            digest = hashlib.sha256(t.encode("utf-8")).hexdigest()
            hashes.append(digest)
        return hashes
