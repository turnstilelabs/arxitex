"""Late interaction retrieval using PyLate PLAID + Reason-ModernColBERT."""

from __future__ import annotations

import os
import shutil
import time
from typing import List, Optional, Sequence, Tuple

from loguru import logger


class PyLateEngine:
    def __init__(
        self,
        model_name: str = "lightonai/Reason-ModernColBERT",
        index_dir: str = "data/pylate_index",
        index_type: str = "plaid",
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.index_type = index_type
        self.index = None
        self._built = False
        self._model = None
        self._texts: Optional[Sequence[str]] = None
        self._ids: Optional[Sequence[str]] = None

    def build(self, texts: Sequence[str], ids: Sequence[str]) -> None:
        pylate = _import_pylate()
        models, indexes = _get_pylate_modules(pylate)
        model = _build_model(models, self.model_name)
        self._model = model
        index_cls = _get_index_class(indexes, prefer=self.index_type)
        self._texts = texts
        self._ids = ids
        logger.info("PyLate index type: {}", self.index_type)
        logger.info(
            "Encoding {} documents with PyLate model {}", len(texts), self.model_name
        )

        index_folder = os.path.join(self.index_dir, "pylate_index")
        if os.path.exists(index_folder):
            shutil.rmtree(index_folder)
        try:
            self.index = index_cls(
                index_folder=index_folder, index_name="colbert", override=True
            )
        except TypeError:
            self.index = index_cls(index_folder=index_folder)

        documents_embeddings = model.encode(
            sentences=list(texts),
            batch_size=8,
            is_query=False,
        )
        try:
            emb_len = len(documents_embeddings)
        except Exception:
            emb_len = None
        logger.info("Encoded embeddings count: {}", emb_len)
        if not documents_embeddings:
            raise RuntimeError("PyLate encode returned no embeddings")
        if hasattr(self.index, "add_documents"):
            self.index.add_documents(
                documents_ids=list(ids), documents_embeddings=documents_embeddings
            )
        else:
            raise RuntimeError("PLAID index does not expose add_documents")

        self._built = True
        logger.info("PyLate index built for {} documents", len(texts))

    def search(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        if not self._built:
            raise RuntimeError("PyLate index not built")

        if self._model is None:
            raise RuntimeError("PyLate model not initialized")
        if not query or not query.strip():
            return [], []
        k = max(1, int(k))
        queries_embeddings = self._model.encode(
            sentences=[query],
            batch_size=1,
            is_query=True,
        )
        if self.index_type == "plaid":
            result = _plaid_search_safe(self.index, queries_embeddings, k, self._ids)
        else:
            if not callable(self.index):
                raise RuntimeError("PyLate index does not expose __call__")
            result = self.index(queries_embeddings=queries_embeddings, k=k)

        return _normalize_result(result, k)

    def latency_probe(self, query: str, k: int = 10) -> float:
        if not query or not query.strip():
            return 0.0
        k = max(1, int(k))
        start = time.perf_counter()
        _ = self.search(query, k=k)
        return time.perf_counter() - start


def _import_pylate():
    try:
        import pylate
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pylate is required for PyLateEngine") from exc
    return pylate


def _get_pylate_modules(pylate):
    try:
        from pylate import indexes, models
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Unable to import pylate.models or pylate.indexes") from exc
    return models, indexes


def _build_model(models, model_name: str):
    for cls_name in (
        "ColBERT",
        "ModernColBERT",
        "ReasonModernColBERT",
        "ReasonModernColbert",
    ):
        cls = getattr(models, cls_name, None)
        if cls is None:
            continue
        try:
            return cls(model_name_or_path=model_name, device="cpu")
        except Exception:
            try:
                return cls(model_name_or_path=model_name)
            except Exception:
                pass
    raise RuntimeError("Unable to construct PyLate model; check pylate API")


def _get_plaid_class(indexes):
    cls = getattr(indexes, "PLAID", None)
    if cls is not None:
        return cls
    raise RuntimeError("Unable to locate PLAID class in pylate.indexes")


def _get_voyager_class(indexes):
    for name in ("Voyager", "VOYAGER", "VoyagerIndex"):
        cls = getattr(indexes, name, None)
        if cls is not None:
            return cls
    return None


def _get_index_class(indexes, prefer: str = "plaid"):
    prefer = (prefer or "plaid").lower()
    if prefer == "voyager":
        cls = _get_voyager_class(indexes)
        if cls is not None:
            return cls
    return _get_plaid_class(indexes)


def _plaid_search_safe(index, queries_embeddings, k: int, ids: Sequence[str] | None):
    try:
        from pylate.rank.rank import reshape_embeddings
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Unable to import pylate rank utilities") from exc

    if not hasattr(index, "searcher") or index.searcher is None:
        raise RuntimeError("PLAID searcher is not initialized")

    plaid_ids_to_documents_ids = index._load_plaid_ids_to_documents_ids()
    has_mapping = len(plaid_ids_to_documents_ids) > 0
    documents = []
    distances = []
    queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
    for query_embeddings in queries_embeddings:
        result = index.searcher.search(query_embeddings, k=k)
        query_docs = []
        for r in result[0]:
            doc_id = None
            if has_mapping:
                try:
                    doc_id = plaid_ids_to_documents_ids.get(r)
                except Exception:
                    doc_id = None
                if doc_id is None:
                    try:
                        doc_id = plaid_ids_to_documents_ids.get(int(r))
                    except Exception:
                        doc_id = None
                if doc_id is None:
                    try:
                        doc_id = plaid_ids_to_documents_ids.get(str(r))
                    except Exception:
                        doc_id = None
            if doc_id is None and ids is not None:
                try:
                    idx = int(r)
                    if 0 <= idx < len(ids):
                        doc_id = ids[idx]
                except Exception:
                    doc_id = None
            if doc_id is not None:
                query_docs.append(doc_id)
        documents.append(query_docs)
        distances.append(result[2])
    plaid_ids_to_documents_ids.close()
    results = [
        [
            {"id": doc_id, "score": score}
            for doc_id, score in zip(query_documents, query_distances)
        ]
        for query_documents, query_distances in zip(documents, distances)
    ]
    return results


def _normalize_result(result, k: int) -> Tuple[List[int], List[float]]:
    if result is None:
        return [], []
    if isinstance(result, list) and result and isinstance(result[0], list):
        # pylate PLAID returns list of list of dicts
        rows = result[0]
        ids = [row.get("id") for row in rows if row.get("id") is not None]
        scores = [row.get("score") for row in rows if row.get("score") is not None]
        return ids[:k], scores[:k]
    if isinstance(result, tuple) and len(result) == 2:
        scores, indices = result
        return list(indices)[:k], list(scores)[:k]
    if isinstance(result, dict):
        indices = result.get("doc_ids") or result.get("indices") or []
        scores = result.get("scores") or []
        return list(indices)[:k], list(scores)[:k]
    if isinstance(result, list):
        return list(result)[:k], [1.0] * min(k, len(result))
    return [], []
