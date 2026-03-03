"""Shared scoring helpers for citation matching."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Optional


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_title(s: str) -> str:
    """Normalize titles for fuzzy matching."""

    t = unicodedata.normalize("NFKD", s or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"\\(emph|textit|textbf|itshape|bfseries)\b", " ", t)
    t = re.sub(r"[{}]", " ", t)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return _norm_ws(t)


def normalize_author(s: str) -> str:
    """Normalize an author string."""

    t = (s or "").strip()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    if not t:
        return ""
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
        if len(parts) >= 2:
            t = " ".join(parts[1:] + [parts[0]])
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return _norm_ws(t)


def title_similarity(a: str, b: str) -> float:
    """Return a normalized title similarity score in [0, 1]."""

    na = normalize_title(a)
    nb = normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(a=na, b=nb).ratio()


def author_overlap(
    wanted: Iterable[str],
    candidate: Iterable[str],
    *,
    use_last_name: bool = True,
) -> float:
    """Compute author overlap ratio.

    When use_last_name is True, compare last-name sets (robust to formatting).
    When False, compare full normalized author strings.
    """

    def last_name(a: str) -> str:
        na = normalize_author(a)
        toks = [t for t in na.split(" ") if t]
        return toks[-1] if toks else ""

    if use_last_name:
        wanted_norm = {last_name(a) for a in wanted if last_name(a)}
        cand_norm = {last_name(a) for a in candidate if last_name(a)}
    else:
        wanted_norm = {normalize_author(a) for a in wanted if normalize_author(a)}
        cand_norm = {normalize_author(a) for a in candidate if normalize_author(a)}

    if not wanted_norm or not cand_norm:
        return 0.0
    return len(wanted_norm.intersection(cand_norm)) / max(1, len(wanted_norm))


def best_match_index(
    candidates: list[dict],
    *,
    title: Optional[str],
    authors: Optional[Iterable[str]],
    title_key: str = "title",
    authors_key: str = "authors",
    count_key: str = "cited_by_count",
    min_title_similarity: float = 0.0,
    min_author_overlap: float = 0.0,
    require_author_overlap: bool = False,
    use_last_name: bool = False,
) -> Optional[int]:
    """Return index of best candidate by title similarity, author overlap, count."""

    wanted_title = title or ""
    wanted_authors = list(authors or [])
    best_idx: Optional[int] = None
    best_score: tuple[float, float, float] | None = None

    for idx, cand in enumerate(candidates or []):
        cand_title = cand.get(title_key) or cand.get("display_name") or ""
        ts = title_similarity(wanted_title, cand_title) if wanted_title else 0.0
        if wanted_title and ts < min_title_similarity:
            continue

        cand_authors = cand.get(authors_key) or []
        ao = author_overlap(wanted_authors, cand_authors, use_last_name=use_last_name)
        if require_author_overlap and ao <= 0.0:
            continue
        if wanted_authors and ao < min_author_overlap:
            continue

        count = cand.get(count_key)
        try:
            count_score = float(count) if count is not None else -1.0
        except Exception:
            count_score = -1.0

        # Primary: title similarity; tie-break with author overlap and then count.
        score = (ts, ao, count_score)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    return best_idx
