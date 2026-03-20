"""Resolve mentions pipeline targets via arXiv + OpenAlex."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from arxitex.arxiv_api import ArxivAPI
from arxitex.arxiv_utils import parse_arxiv_id
from arxitex.tools.matching.scoring import best_match_index
from arxitex.tools.mentions.acquisition.openalex_citations import (
    OPENALEX_BASE,
    OpenAlexClient,
)
from arxitex.tools.mentions.extraction.mention_utils import title_similarity

DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SURVEY_RE = re.compile(
    r"\b(survey|current developments|cdm|lecture notes|applications?)\b",
    re.IGNORECASE,
)
PMIHES_RE = re.compile(r"\b(ih[eé]s|publ\.\s*math|hautes [eé]tudes)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TargetWorkProfile:
    title: str
    authors: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    year: Optional[int] = None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _extract_doi(text: str) -> Optional[str]:
    m = DOI_RE.search(text or "")
    if not m:
        return None
    return m.group(0).rstrip(".,);]").lower()


def _extract_year(text: str) -> Optional[int]:
    m = YEAR_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def classify_bib_entry(entry_text: str, profile: TargetWorkProfile) -> str:
    """Classify whether a bibliography entry points to the target work."""
    text = (entry_text or "").strip()
    if not text:
        return "unknown"

    doi = _extract_doi(text)
    if profile.doi and doi:
        if doi == profile.doi.lower():
            return "exact_target"
        return "non_target"

    sim = title_similarity(text, profile.title or "")
    text_norm = _norm(text)
    title_norm = _norm(profile.title or "")
    title_prefix_match = bool(title_norm and title_norm in text_norm)
    has_survey_cue = bool(SURVEY_RE.search(text))
    has_pmihes_cue = bool(PMIHES_RE.search(text))
    entry_year = _extract_year(text)
    year_delta = (
        abs(entry_year - profile.year)
        if entry_year is not None and profile.year is not None
        else None
    )

    if title_prefix_match and has_survey_cue:
        return "non_target"

    # Bibliography strings are short/noisy; title phrase containment is a strong cue.
    if title_prefix_match:
        if has_pmihes_cue:
            return "same_work_alt_version"
        if year_delta is not None and year_delta <= 3:
            return "same_work_alt_version"
        if sim >= 0.85:
            return "same_work_alt_version"
        return "unknown"

    if sim >= 0.92 and not has_survey_cue:
        return "same_work_alt_version"

    return "non_target"


class OpenAlexTargetResolver:
    """Resolve a target paper from arXiv metadata and OpenAlex search results.

    Responsibilities:
    - Derive stable dataset target ids from arXiv ids.
    - Fetch title/authors from arXiv when not provided.
    - Select the best OpenAlex work id using title similarity and author overlap.
    """

    def __init__(
        self,
        *,
        cache_dir: str,
        mailto: Optional[str] = None,
        api_key: Optional[str] = None,
        per_page: int = 25,
    ) -> None:
        self.cache_dir = cache_dir
        self.mailto = mailto
        self.api_key = api_key
        self.per_page = per_page
        self._openalex_client = OpenAlexClient(
            cache_dir=cache_dir,
            rate_limit=0.0,
            mailto=mailto,
            api_key=api_key,
            per_page=per_page,
        )

    def derive_target_id(self, arxiv_id: str) -> str:
        if not arxiv_id:
            raise ValueError("arXiv id is required to derive target id")
        safe = arxiv_id.replace("/", "_").strip()
        return f"arxiv_{safe}"

    def fetch_arxiv_metadata(self, arxiv_id: str, api: ArxivAPI) -> TargetWorkProfile:
        params = {"id_list": parse_arxiv_id(arxiv_id)}
        resp = api.session.get(api.base_url, params=params, timeout=30)
        resp.raise_for_status()
        count, _total, entries = api.parse_response(resp.text)
        if count <= 0 or not entries:
            raise RuntimeError(f"No arXiv entry found for {arxiv_id}")
        paper = api.entry_to_paper(entries[0])
        if not paper:
            raise RuntimeError(f"Unable to parse arXiv entry for {arxiv_id}")
        return TargetWorkProfile(
            title=paper.get("title") or "",
            authors=list(paper.get("authors") or []),
        )

    @classmethod
    def select_openalex_work_id(
        cls,
        results: list[dict[str, Any]],
        title: Optional[str],
        authors: Optional[Iterable[str]],
    ) -> Optional[str]:
        candidates: list[dict[str, Any]] = []
        for r in results or []:
            r_authors = [
                (au.get("author") or {}).get("display_name")
                for au in (r.get("authorships") or [])
                if isinstance(au, dict)
            ]
            candidates.append(
                {
                    "id": r.get("id"),
                    "title": r.get("title") or r.get("display_name") or "",
                    "authors": r_authors,
                    "cited_by_count": r.get("cited_by_count") or -1,
                }
            )

        min_title = 0.92 if title else 0.0
        min_author = 0.10 if authors else 0.0
        require_author = bool(authors)

        idx = best_match_index(
            candidates,
            title=title,
            authors=authors,
            title_key="title",
            authors_key="authors",
            count_key="cited_by_count",
            min_title_similarity=min_title,
            min_author_overlap=min_author,
            require_author_overlap=require_author,
            use_last_name=False,
        )
        if idx is None:
            return None
        return candidates[idx].get("id")

    def resolve_openalex_work_id(
        self,
        *,
        title: str,
        authors: Optional[Iterable[str]],
    ) -> Optional[str]:
        if not title:
            return None

        from urllib.parse import urlencode

        params = {"search": title, "per-page": self.per_page}
        url = f"{OPENALEX_BASE}/works?{urlencode(params)}"
        data = self._openalex_client.fetch_json(url)

        results = data.get("results") or []
        return self.select_openalex_work_id(results, title, authors)

    def fetch_openalex_work(self, work_id_or_url: str) -> Optional[dict[str, Any]]:
        """Fetch one OpenAlex work JSON via shared OpenAlex client/caching."""
        if not work_id_or_url:
            return None
        wid = str(work_id_or_url).strip()
        if "/works/" in wid:
            wid = wid.rsplit("/", 1)[-1]
        if not wid:
            return None

        url = f"{OPENALEX_BASE}/works/{wid}"
        return self._openalex_client.fetch_json(url)
