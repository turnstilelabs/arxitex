from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Optional

from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema
from arxitex.tools.citations_openalex import strip_arxiv_version


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_title(s: str) -> str:
    """Normalize titles for fuzzy matching.

    Goal: be robust to case, punctuation, and minor TeX/formatting noise.
    """

    # First, decompose Unicode accents so we can strip combining marks,
    # turning e.g. "Erd\u00f6s" into "Erdos" rather than "Erd os".
    t = unicodedata.normalize("NFKD", s or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    # Drop common TeX commands and braces.
    t = re.sub(r"\\(emph|textit|textbf|itshape|bfseries)\b", " ", t)
    t = re.sub(r"[{}]", " ", t)
    # Remove math env markers that can appear in bib titles.
    t = re.sub(r"\$[^$]*\$", " ", t)
    # Normalize punctuation to spaces.
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = _norm_ws(t)
    return t


def normalize_author(s: str) -> str:
    """Normalize an author string.

    Handles "Last, First" -> "first last" and drops punctuation.
    """

    t = (s or "").strip()
    # Strip accents from author names as well so that "Erd\u00f6s" -> "Erdos".
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


ARXIV_ID_IN_TEXT_RE = re.compile(
    r"(?:arxiv\s*[:\s]*|arxiv:|\babs/)([\d\.]{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+\.[a-z\-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def try_extract_arxiv_id_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = ARXIV_ID_IN_TEXT_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _strip_tex_commands(s: str) -> str:
    """Best-effort removal of TeX commands for heuristics."""

    t = s or ""

    # Normalize common TeX accent commands to their base characters before
    # we strip generic macros. This avoids turning "Erd\H{o}s" into
    # "Erd s"; instead we want "Erdos".
    accent_patterns = [
        r"\\\"\s*\{?([A-Za-z])\}?",  # \"o or \"{o} or \" o
        r"\\'\s*\{?([A-Za-z])\}?",  # \'a or \'{a} or \' a
        r"\\`\s*\{?([A-Za-z])\}?",  # \`a or \`{a} or \` a
        r"\\\^\s*\{?([A-Za-z])\}?",  # \^o variants
        r"\\~\s*\{?([A-Za-z])\}?",  # \~n variants
        r"\\H\s*\{?([A-Za-z])\}?",  # \H{o} or \H o
        r"\\c\s*\{?([A-Za-z])\}?",  # \c{c} or \c c
        r"\\k\s*\{?([A-Za-z])\}?",  # \k{a} or \k a
    ]
    for pat in accent_patterns:
        t = re.sub(pat, r"\1", t)
    # Replace remaining simple macros with a space.
    t = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", t)
    t = re.sub(r"[{}]", " ", t)
    # Strip accents introduced by direct Unicode characters.
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = _norm_ws(t)
    return t


def extract_title_and_authors(full_reference: str) -> tuple[Optional[str], list[str]]:
    """Heuristically extract (title, authors[]) from a bibliography-like string.

    This is intentionally best-effort. If we fail to find a plausible title, we
    return (None, []). Authors may also be empty.
    """

    if not full_reference:
        return None, []

    ref = _norm_ws(full_reference)

    # 1) Quoted title.
    for qre in (
        re.compile(r"“([^”]{6,})”"),
        re.compile(r'"([^"]{6,})"'),
        re.compile(r"'([^']{6,})'"),
    ):
        m = qre.search(ref)
        if m:
            title = _norm_ws(m.group(1))
            authors = _extract_authors_prefix(ref[: m.start()])
            return title, authors

    # 2) TeX emphasis patterns (\emph{Title}).
    m = re.search(r"\\emph\{([^}]{6,})\}", ref)
    if m:
        title = _norm_ws(m.group(1))
        authors = _extract_authors_prefix(ref[: m.start()])
        return title, authors

    # 3) Split by commas and choose a plausible segment.
    clean = _strip_tex_commands(ref)
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    if len(parts) < 2:
        return None, []

    def is_noise(seg: str) -> bool:
        s = seg.lower()
        if "arxiv" in s or "doi" in s or "http" in s:
            return True
        if re.search(r"\b(19|20)\d{2}\b", s):
            # year-like
            return True
        if re.search(r"\bvol\b|\bno\b|\bpp\b|pages?\b|journal\b|proc\b", s):
            return True
        return False

    candidates: list[str] = []
    for seg in parts:
        if is_noise(seg):
            continue
        # title-ish: longer than a typical author token and has letters.
        if len(seg) >= 10 and re.search(r"[a-zA-Z]", seg):
            candidates.append(seg)
    if not candidates:
        return None, []

    # Prefer the longest remaining segment.
    title = max(candidates, key=len)

    # Authors: heuristic prefix before the title segment in the original cleaned string.
    joined = ", ".join(parts)
    i = joined.lower().find(title.lower())
    authors = []
    if i > 0:
        authors = _extract_authors_prefix(joined[:i])

    return _norm_ws(title), authors


def _extract_authors_prefix(prefix: str) -> list[str]:
    """Extract list of authors from the prefix part of a bib entry."""

    p = _strip_tex_commands(prefix)
    p = re.sub(r"\bet\s+al\b\.?", " ", p, flags=re.IGNORECASE)
    p = _norm_ws(p)
    if not p:
        return []

    # Try splitting by ' and ' first, then commas.
    if " and " in p.lower():
        raw = [
            a.strip() for a in re.split(r"\band\b", p, flags=re.IGNORECASE) if a.strip()
        ]
    else:
        raw = [a.strip() for a in p.split(",") if a.strip()]

    # Filter non-author-like tokens.
    out: list[str] = []
    for a in raw:
        # Avoid "Some paper" or random phrases.
        if len(a) < 3:
            continue
        if re.search(r"\b(arxiv|doi|http|vol|no|pp|pages?)\b", a, flags=re.IGNORECASE):
            continue
        # Must contain at least one letter.
        if not re.search(r"[A-Za-z]", a):
            continue
        out.append(_norm_ws(a))
        if len(out) >= 6:
            break
    return out


def _author_overlap(wanted: list[str], got: list[str]) -> float:
    """Compute author overlap in a robust way.

    We intentionally compare by **last name** (last token) because bibliographies
    often abbreviate given names (e.g., "J. Doe"), while arXiv returns full
    names.
    """

    def last_name(a: str) -> str:
        na = normalize_author(a)
        toks = [t for t in na.split(" ") if t]
        return toks[-1] if toks else ""

    w = {last_name(a) for a in (wanted or []) if last_name(a)}
    g = {last_name(a) for a in (got or []) if last_name(a)}
    if not w or not g:
        return 0.0
    return len(w.intersection(g)) / max(1, len(w))


def _title_score(wanted: str, got: str) -> float:
    a = normalize_title(wanted)
    b = normalize_title(got)
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


@dataclass
class MatchResult:
    matched_arxiv_id: Optional[str]
    match_method: str  # direct_regex|search|none
    extracted_title: Optional[str]
    extracted_authors: list[str]
    matched_title: Optional[str]
    matched_authors: list[str]
    title_score: Optional[float]
    author_overlap: Optional[float]
    arxiv_query: Optional[str]


def _cache_key(title: str, authors: list[str]) -> str:
    payload = {
        "title": normalize_title(title),
        "authors": [
            normalize_author(a) for a in (authors or []) if normalize_author(a)
        ],
    }
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _load_cache(conn, *, cache_key: str) -> Optional[dict]:
    row = conn.execute(
        """
        SELECT matched_arxiv_id, matched_title, matched_authors_json,
               title_score, author_overlap, arxiv_query, last_fetched_at_utc
        FROM external_reference_arxiv_search_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if not row:
        return None
    return dict(row)


def _upsert_cache(conn, *, cache_key: str, data: dict) -> None:
    conn.execute(
        """
        INSERT INTO external_reference_arxiv_search_cache (
            cache_key, matched_arxiv_id, matched_title, matched_authors_json,
            title_score, author_overlap, arxiv_query, last_fetched_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            matched_arxiv_id=excluded.matched_arxiv_id,
            matched_title=excluded.matched_title,
            matched_authors_json=excluded.matched_authors_json,
            title_score=excluded.title_score,
            author_overlap=excluded.author_overlap,
            arxiv_query=excluded.arxiv_query,
            last_fetched_at_utc=excluded.last_fetched_at_utc
        """,
        (
            cache_key,
            data.get("matched_arxiv_id"),
            data.get("matched_title"),
            data.get("matched_authors_json"),
            data.get("title_score"),
            data.get("author_overlap"),
            data.get("arxiv_query"),
            data.get("last_fetched_at_utc") or _utc_now_iso(),
        ),
    )


def match_external_reference_to_arxiv(
    *,
    api: ArxivAPI,
    full_reference: str,
    extracted_title: Optional[str] = None,
    extracted_authors: Optional[list[str]] = None,
    title_threshold_with_authors: float = 0.92,
    title_threshold_no_authors: float = 0.96,
    refresh_cache: bool = False,
    db_path_for_cache: Optional[str] = None,
    refresh_days: int = 30,
) -> MatchResult:
    """Best-effort match of a bibliography reference to an arXiv paper.

    - Uses a robust arXiv-id regex as a fast path.
    - Otherwise extracts title/authors heuristically and searches arXiv API.
    - Optional SQLite-backed cache to skip repeat queries.
    """

    # Fast path: sometimes the resolver missed an arXiv id; re-check.
    direct = try_extract_arxiv_id_from_text(full_reference)
    if direct:
        return MatchResult(
            matched_arxiv_id=strip_arxiv_version(direct),
            match_method="direct_regex",
            extracted_title=None,
            extracted_authors=[],
            matched_title=None,
            matched_authors=[],
            title_score=None,
            author_overlap=None,
            arxiv_query=None,
        )

    title = extracted_title
    authors = extracted_authors or []
    if not title:
        title, authors2 = extract_title_and_authors(full_reference)
        authors = authors or authors2

    if not title:
        return MatchResult(
            matched_arxiv_id=None,
            match_method="none",
            extracted_title=None,
            extracted_authors=[],
            matched_title=None,
            matched_authors=[],
            title_score=None,
            author_overlap=None,
            arxiv_query=None,
        )

    # Cache
    cache_key = None
    cached = None
    if db_path_for_cache:
        ensure_schema(db_path_for_cache)
        cache_key = _cache_key(title, authors)
        conn = connect(db_path_for_cache)
        try:
            cached = _load_cache(conn, cache_key=cache_key)
        finally:
            conn.close()

        if cached and not refresh_cache:
            # Respect refresh_days.
            try:
                dt = datetime.fromisoformat(
                    str(cached.get("last_fetched_at_utc") or "").replace("Z", "+00:00")
                )
            except Exception:
                dt = None
            if dt is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_days)
                if dt >= cutoff:
                    # Return cached decision (including cached misses).
                    ma = cached.get("matched_arxiv_id")
                    mt = cached.get("matched_title")
                    try:
                        mas = json.loads(cached.get("matched_authors_json") or "[]")
                    except Exception:
                        mas = []
                    return MatchResult(
                        matched_arxiv_id=ma,
                        match_method="search" if ma else "none",
                        extracted_title=title,
                        extracted_authors=authors,
                        matched_title=mt,
                        matched_authors=mas,
                        title_score=(
                            float(cached["title_score"])
                            if cached.get("title_score") is not None
                            else None
                        ),
                        author_overlap=(
                            float(cached["author_overlap"])
                            if cached.get("author_overlap") is not None
                            else None
                        ),
                        arxiv_query=str(cached.get("arxiv_query") or ""),
                    )

    # Build query
    # Quoted title improves precision; arXiv API query syntax supports ti:"...".
    # We add one author last name when available as a weak filter.
    query = f'ti:"{title}"'
    if authors:
        # take last token of first author
        n = normalize_author(authors[0]).split(" ")
        if n:
            query += f" AND au:{n[-1]}"

    xml = api.fetch_papers(query, start=0, batch_size=10)
    cnt, total, entries = api.parse_response(xml)
    if cnt == 0 or not entries:
        res = MatchResult(
            matched_arxiv_id=None,
            match_method="none",
            extracted_title=title,
            extracted_authors=authors,
            matched_title=None,
            matched_authors=[],
            title_score=None,
            author_overlap=None,
            arxiv_query=query,
        )
        if db_path_for_cache and cache_key:
            _write_cache_record(
                db_path_for_cache,
                cache_key,
                res,
                query,
            )
        return res

    best: Optional[MatchResult] = None
    best_score = -1.0

    for e in entries:
        paper = api.entry_to_paper(e)
        if not paper:
            continue
        pt = paper.get("title") or ""
        pa = paper.get("authors") or []
        ts = _title_score(title, pt)
        ao = _author_overlap(authors, pa)

        # Prefer title score primarily.
        score = ts + 0.1 * ao
        if score > best_score:
            best_score = score
            best = MatchResult(
                matched_arxiv_id=strip_arxiv_version(str(paper.get("arxiv_id") or "")),
                match_method="search",
                extracted_title=title,
                extracted_authors=authors,
                matched_title=pt,
                matched_authors=pa,
                title_score=ts,
                author_overlap=ao,
                arxiv_query=query,
            )

    if best is None:
        best = MatchResult(
            matched_arxiv_id=None,
            match_method="none",
            extracted_title=title,
            extracted_authors=authors,
            matched_title=None,
            matched_authors=[],
            title_score=None,
            author_overlap=None,
            arxiv_query=query,
        )

    # Thresholding
    if best.matched_arxiv_id and best.title_score is not None:
        if authors:
            ok = best.title_score >= title_threshold_with_authors and (
                (best.author_overlap or 0.0) >= 0.10
            )
        else:
            ok = best.title_score >= title_threshold_no_authors
        if not ok:
            best = MatchResult(
                matched_arxiv_id=None,
                match_method="none",
                extracted_title=title,
                extracted_authors=authors,
                matched_title=best.matched_title,
                matched_authors=best.matched_authors,
                title_score=best.title_score,
                author_overlap=best.author_overlap,
                arxiv_query=query,
            )

    if db_path_for_cache and cache_key:
        _write_cache_record(db_path_for_cache, cache_key, best, query)

    return best


def _write_cache_record(db_path: str, cache_key: str, res: MatchResult, query: str):
    try:
        conn = connect(db_path)
        try:
            with conn:
                _upsert_cache(
                    conn,
                    cache_key=cache_key,
                    data={
                        "matched_arxiv_id": res.matched_arxiv_id,
                        "matched_title": res.matched_title,
                        "matched_authors_json": json.dumps(
                            res.matched_authors or [], ensure_ascii=False
                        ),
                        "title_score": res.title_score,
                        "author_overlap": res.author_overlap,
                        "arxiv_query": query,
                        "last_fetched_at_utc": _utc_now_iso(),
                    },
                )
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover - best-effort cache
        logger.debug(f"Failed to write arXiv cache: {e}")
