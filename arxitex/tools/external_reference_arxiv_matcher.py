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


@dataclass
class TitleCandidate:
    title: str
    method: str
    score: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_title(s: str) -> str:
    """Normalize titles for fuzzy matching."""

    # Decompose accents, strip combining marks.
    t = unicodedata.normalize("NFKD", s or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    # Drop common TeX commands and braces.
    t = re.sub(r"\\(emph|textit|textbf|itshape|bfseries)\b", " ", t)
    t = re.sub(r"[{}]", " ", t)
    # Remove inline math.
    t = re.sub(r"\$[^$]*\$", " ", t)
    # Normalize punctuation to spaces.
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


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _is_doi_url(text: str) -> bool:
    return bool(re.search(r"https?://(?:dx\.)?doi\.org/", text, flags=re.IGNORECASE))


def is_url_like_reference(text: str) -> bool:
    """True for dataset/software/webpage references (not papers)."""

    t = (text or "").strip()
    if not t:
        return False
    if _is_doi_url(t):
        return False
    if "\\href{" in t:
        return False
    return bool(_URL_RE.search(t))


def _extract_braced_group(s: str, start: int) -> tuple[Optional[str], int]:
    """Extract a {...} group starting at s[start] == '{'."""

    if start < 0 or start >= len(s) or s[start] != "{":
        return None, start
    depth = 0
    i = start
    buf: list[str] = []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf), i + 1
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    return None, start


def _extract_href_title(ref: str) -> Optional[str]:
    r"""Extract title argument of \href{url}{title}, supporting nested braces."""

    i = ref.find("\\href")
    if i == -1:
        return None
    j = ref.find("{", i)
    if j == -1:
        return None
    _url, k = _extract_braced_group(ref, j)
    if _url is None:
        return None
    k = ref.find("{", k)
    if k == -1:
        return None
    title, _ = _extract_braced_group(ref, k)
    return title


def _strip_tex_commands(s: str) -> str:
    """Best-effort TeX cleanup for candidate extraction."""

    t = s or ""

    # TeX accents -> base letters (allow whitespace).
    accent_patterns = [
        r"\\\"\s*\{?([A-Za-z])\}?",
        r"\\'\s*\{?([A-Za-z])\}?",
        r"\\`\s*\{?([A-Za-z])\}?",
        r"\\\^\s*\{?([A-Za-z])\}?",
        r"\\~\s*\{?([A-Za-z])\}?",
        r"\\H\s*\{?([A-Za-z])\}?",
        r"\\c\s*\{?([A-Za-z])\}?",
        r"\\k\s*\{?([A-Za-z])\}?",
    ]
    for pat in accent_patterns:
        t = re.sub(pat, r"\1", t)

    # Replace common structural macros.
    t = re.sub(r"\\newblock\b", " ", t)
    # Drop other macros but keep their brace content when possible (rough).
    t = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", t)
    t = t.replace("{", " ").replace("}", " ")

    # Strip accents introduced by unicode.
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return _norm_ws(t)


def _extract_authors_prefix(prefix: str) -> list[str]:
    p = _strip_tex_commands(prefix)
    p = re.sub(r"\bet\s+al\b\.?", " ", p, flags=re.IGNORECASE)
    p = _norm_ws(p)
    if not p:
        return []
    if " and " in p.lower():
        raw = [
            a.strip() for a in re.split(r"\band\b", p, flags=re.IGNORECASE) if a.strip()
        ]
    else:
        raw = [a.strip() for a in p.split(",") if a.strip()]
    out: list[str] = []
    for a in raw:
        if len(a) < 3:
            continue
        if re.search(r"\b(arxiv|doi|http|vol|no|pp|pages?)\b", a, flags=re.IGNORECASE):
            continue
        if not re.search(r"[A-Za-z]", a):
            continue
        out.append(_norm_ws(a))
        if len(out) >= 6:
            break
    return out


def _strip_outer_quotes(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        return t[1:-1].strip()
    return t


def _strip_trailing_metadata(s: str) -> str:
    """Remove obvious trailing metadata from a candidate title span."""

    t = _norm_ws(s)
    # Cut at common journal/publisher boundaries.
    for pat in (
        r"\\textit\b",
        r"\\emph\b",
        r"\bInternational\b",
        r"\bJournal\b",
        r"\bJ\.\b",
        r"\bProc\b",
    ):
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m and m.start() > 0:
            t = t[: m.start()].strip(" ,;:")
    # Cut at year
    m = re.search(r"\b(19|20)\d{2}\b", t)
    if m and m.start() > 0:
        t = t[: m.start()].strip(" ,;:")
    return t


def _looks_like_author_segment(seg: str) -> bool:
    """Heuristic: decide if `seg` is another author token (e.g. 'S.~Norine')."""

    s = _strip_tex_commands(seg)
    low = s.lower()
    if not s:
        return False
    # short-ish, contains a name-like token
    if len(s) > 40:
        return False
    if re.search(r"\b[A-Z]\.?\b", seg):
        return True
    if "~" in seg:
        return True
    # "Lastname" only
    if re.fullmatch(r"[A-Za-z\-]{3,}", s):
        return True
    # Avoid mistaking a title fragment for an author.
    if any(w in low.split() for w in {"of", "and", "in", "on", "for", "with"}):
        return False
    return False


def extract_title_and_authors(full_reference: str) -> tuple[Optional[str], list[str]]:
    """Backward-compatible helper.

    Historically we returned a single best-effort title and an authors list.
    With the new candidate-based approach, we return the top-ranked candidate
    (if any).
    """

    cands, authors = generate_title_candidates(full_reference, limit=1)
    if not cands:
        return None, []
    return cands[0].title, authors


PUBLISHER_WORDS = {
    "springer",
    "cambridge",
    "oxford",
    "wiley",
    "elsevier",
    "birkhauser",
    "birkhäuser",
    "ams",
}

JOURNAL_WORDS = {
    "journal",
    "notices",
    "transactions",
    "annals",
    "proceedings",
    "series",
    "vol",
    "no",
    "pp",
    "research",
}


def _candidate_quality_score(title: str) -> float:
    t = _strip_tex_commands(title)
    t = _norm_ws(t)
    if not t:
        return -1e9
    low = t.lower().strip()

    # reject date/year-only
    if low in PUBLISHER_WORDS:
        return -1e9
    if re.fullmatch(r"(19|20)\d{2}\.?", low):
        return -1e9
    if re.fullmatch(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(19|20)\d{2}\.?",
        low,
    ):
        return -1e9

    words = [w for w in re.split(r"\s+", low) if w]
    n = len(words)
    if n < 3:
        return -1e9

    score = min(12.0, float(n))
    if any(w in JOURNAL_WORDS for w in words):
        score -= 8.0
    if any(w in PUBLISHER_WORDS for w in words):
        score -= 10.0
    if re.search(r"\b(19|20)\d{2}\b", low):
        score -= 3.0
    if any(w in {"of", "and", "in", "on", "for", "with", "via"} for w in words):
        score += 2.0
    return score


def generate_title_candidates(
    full_reference: str,
    *,
    limit: int = 4,
) -> tuple[list[TitleCandidate], list[str]]:
    """Generate up to `limit` plausible titles and a best-effort authors list."""

    ref = _norm_ws(full_reference or "")
    if not ref:
        return [], []
    if is_url_like_reference(ref):
        return [], []

    # Prefer authors block up to first comma, but support "A, B, Title" where
    # the second comma still belongs to the author list.
    authors: list[str] = []
    first_comma = ref.find(",")
    second_comma = ref.find(",", first_comma + 1) if first_comma != -1 else -1
    author_comma_end = first_comma
    if first_comma != -1:
        if second_comma != -1:
            between = ref[first_comma + 1 : second_comma]
            if _looks_like_author_segment(between):
                author_comma_end = second_comma
        authors = _extract_authors_prefix(ref[:author_comma_end])

    raw: list[tuple[str, str]] = []

    # 0) \textit{Journal} pattern: "Author, Title. \textit{Journal} ..."
    if "\\textit{" in ref:
        j = ref.find("\\textit{")
        if j > 0:
            head = ref[:j]
            last_dot = head.rfind(".")
            first_comma2 = head.find(",")
            if first_comma2 != -1 and last_dot != -1 and last_dot > first_comma2:
                raw.append((head[first_comma2 + 1 : last_dot], "textit_dot"))

    # 0b) Year-dot pattern: "..., 2005. Title. Journal ..." -> take after year-dot
    m_year_dot = re.search(r"\b(19|20)\d{2}\s*\.\s*", ref)
    if m_year_dot:
        after_year = ref[m_year_dot.end() :]
        dot = after_year.find(".")
        if dot != -1:
            raw.append((after_year[:dot], "after_year_dot"))

    # 1) href
    href_t = _extract_href_title(ref)
    if href_t:
        raw.append((href_t, "href"))

    # 2) quoted (common bib style)
    for qre in (
        re.compile(r"“([^”]{6,})”"),
        re.compile(r'"([^"]{6,})"'),
        re.compile(r"'([^']{6,})'"),
    ):
        m = qre.search(ref)
        if m:
            raw.append((_strip_outer_quotes(m.group(1)), "quoted"))
            break

    # 3) emph
    m = re.search(r"\\emph\{([^}]{6,})\}", ref)
    if m:
        raw.append((m.group(1), "emph"))

    # 4) Title spans based on comma structure.
    if first_comma != -1:
        # Candidate: between first and second comma (useful for "Author, Title, Subtitle, ...")
        if second_comma != -1:
            mid = ref[first_comma + 1 : second_comma]
            raw.append((mid, "between_comma1_comma2"))

        # Candidate: after author block (comma) to first period
        start = (author_comma_end + 1) if author_comma_end != -1 else (first_comma + 1)
        after = ref[start:]

        # If we have a year-dot inside this span, cut to after it.
        m_year_prefix = re.match(r"\s*(19|20)\d{2}\s*\.\s*", after)
        if m_year_prefix:
            after = after[m_year_prefix.end() :]

        dot = after.find(".")
        if dot != -1:
            raw.append((_strip_trailing_metadata(after[:dot]), "after_authors_dot"))

        m_year = re.search(r"\b(19|20)\d{2}\b", after)
        if m_year:
            raw.append(
                (
                    _strip_trailing_metadata(after[: m_year.start()]),
                    "after_authors_year",
                )
            )

    # De-dupe and score
    seen: set[str] = set()
    out: list[TitleCandidate] = []
    for cand, method in raw:
        cleaned = _strip_tex_commands(cand)
        cleaned = _norm_ws(cleaned).strip(" ,;:")
        if not cleaned:
            continue
        cleaned = _strip_outer_quotes(cleaned)
        key = normalize_title(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        sc = _candidate_quality_score(cleaned)
        if sc < -1e8:
            continue
        out.append(TitleCandidate(title=cleaned, method=method, score=sc))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[: max(0, int(limit))], authors


def _author_last_name(authors: list[str]) -> Optional[str]:
    if not authors:
        return None
    n = normalize_author(authors[0]).split(" ")
    n = [t for t in n if t]
    return n[-1] if n else None


def _author_overlap(wanted: list[str], got: list[str]) -> float:
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
    return dict(row) if row else None


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


def _candidate_query(title: str, authors: list[str]) -> str:
    q = f'ti:"{title}"'
    last = _author_last_name(authors)
    if last:
        q += f" AND au:{last}"
    return q


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
    max_candidates: int = 4,
) -> MatchResult:
    """Try up to `max_candidates` candidate titles and keep the best arXiv match."""

    # direct id
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

    # candidates
    if extracted_title:
        candidates = [
            TitleCandidate(title=extracted_title, method="provided", score=0.0)
        ]
        authors = extracted_authors or []
    else:
        candidates, authors = generate_title_candidates(
            full_reference, limit=max_candidates
        )
        authors = extracted_authors or authors

    if not candidates:
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

    # Evaluate each candidate, using the existing cache.
    best: Optional[MatchResult] = None
    best_score = -1.0

    for cand in candidates[:max_candidates]:
        title = cand.title
        query = _candidate_query(title, authors)

        # Cache lookup per candidate.
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
                try:
                    dt = datetime.fromisoformat(
                        str(cached.get("last_fetched_at_utc") or "").replace(
                            "Z", "+00:00"
                        )
                    )
                except Exception:
                    dt = None
                if dt is not None:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_days)
                    if dt >= cutoff:
                        ma = cached.get("matched_arxiv_id")
                        mt = cached.get("matched_title")
                        try:
                            mas = json.loads(cached.get("matched_authors_json") or "[]")
                        except Exception:
                            mas = []
                        ts = (
                            float(cached["title_score"])
                            if cached.get("title_score") is not None
                            else None
                        )
                        ao = (
                            float(cached["author_overlap"])
                            if cached.get("author_overlap") is not None
                            else None
                        )

                        if ma and ts is not None:
                            score = ts + 0.1 * (ao or 0.0)
                            if score > best_score:
                                best_score = score
                                best = MatchResult(
                                    matched_arxiv_id=ma,
                                    match_method="search",
                                    extracted_title=title,
                                    extracted_authors=authors,
                                    matched_title=mt,
                                    matched_authors=mas,
                                    title_score=ts,
                                    author_overlap=ao,
                                    arxiv_query=str(cached.get("arxiv_query") or query),
                                )
                        continue

        xml = api.fetch_papers(query, start=0, batch_size=10)
        cnt, _total, entries = api.parse_response(xml)
        if cnt == 0 or not entries:
            # cache miss decision
            if db_path_for_cache and cache_key:
                _write_cache_record(
                    db_path_for_cache,
                    cache_key,
                    MatchResult(
                        matched_arxiv_id=None,
                        match_method="none",
                        extracted_title=title,
                        extracted_authors=authors,
                        matched_title=None,
                        matched_authors=[],
                        title_score=None,
                        author_overlap=None,
                        arxiv_query=query,
                    ),
                    query,
                )
            continue

        # pick best entry for this candidate
        local_best: Optional[MatchResult] = None
        local_best_score = -1.0

        for e in entries:
            paper = api.entry_to_paper(e)
            if not paper:
                continue
            pt = paper.get("title") or ""
            pa = paper.get("authors") or []
            ts = _title_score(title, pt)
            ao = _author_overlap(authors, pa)
            score = ts + 0.1 * ao
            if score > local_best_score:
                local_best_score = score
                local_best = MatchResult(
                    matched_arxiv_id=strip_arxiv_version(
                        str(paper.get("arxiv_id") or "")
                    ),
                    match_method="search",
                    extracted_title=title,
                    extracted_authors=authors,
                    matched_title=pt,
                    matched_authors=pa,
                    title_score=ts,
                    author_overlap=ao,
                    arxiv_query=query,
                )

        if local_best is None:
            continue

        # Threshold
        if local_best.title_score is not None:
            if authors:
                ok = (
                    local_best.title_score >= title_threshold_with_authors
                    and (local_best.author_overlap or 0.0) >= 0.10
                )
            else:
                ok = local_best.title_score >= title_threshold_no_authors
            if not ok:
                # Cache rejection (still store metadata)
                local_best = MatchResult(
                    matched_arxiv_id=None,
                    match_method="none",
                    extracted_title=title,
                    extracted_authors=authors,
                    matched_title=local_best.matched_title,
                    matched_authors=local_best.matched_authors,
                    title_score=local_best.title_score,
                    author_overlap=local_best.author_overlap,
                    arxiv_query=query,
                )

        # Write cache for this candidate.
        if db_path_for_cache and cache_key:
            _write_cache_record(db_path_for_cache, cache_key, local_best, query)

        if local_best.matched_arxiv_id and local_best.title_score is not None:
            score = local_best.title_score + 0.1 * (local_best.author_overlap or 0.0)
            if score > best_score:
                best_score = score
                best = local_best

    if best is None:
        # If nothing matched, return a "none" result but keep the best candidate
        # title for observability.
        top = candidates[0]
        return MatchResult(
            matched_arxiv_id=None,
            match_method="none",
            extracted_title=top.title,
            extracted_authors=authors,
            matched_title=None,
            matched_authors=[],
            title_score=None,
            author_overlap=None,
            arxiv_query=_candidate_query(top.title, authors),
        )
    return best


def _write_cache_record(
    db_path: str, cache_key: str, res: MatchResult, query: str
) -> None:
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
    except Exception as e:  # pragma: no cover
        logger.debug(f"Failed to write arXiv cache: {e}")
