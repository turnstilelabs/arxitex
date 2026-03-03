"""Resolve citation dataset targets via arXiv + OpenAlex."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests

from arxitex.arxiv_api import ArxivAPI
from arxitex.arxiv_utils import parse_arxiv_id
from arxitex.tools.citations.utils import ensure_dir, sha256_hash
from arxitex.tools.matching.scoring import best_match_index


@dataclass
class TargetMetadata:
    title: str
    authors: list[str]


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

    def derive_target_id(self, arxiv_id: str) -> str:
        if not arxiv_id:
            raise ValueError("arXiv id is required to derive target id")
        safe = arxiv_id.replace("/", "_").strip()
        return f"arxiv_{safe}"

    def fetch_arxiv_metadata(self, arxiv_id: str, api: ArxivAPI) -> TargetMetadata:
        params = {"id_list": parse_arxiv_id(arxiv_id)}
        resp = api.session.get(api.base_url, params=params, timeout=30)
        resp.raise_for_status()
        count, _total, entries = api.parse_response(resp.text)
        if count <= 0 or not entries:
            raise RuntimeError(f"No arXiv entry found for {arxiv_id}")
        paper = api.entry_to_paper(entries[0])
        if not paper:
            raise RuntimeError(f"Unable to parse arXiv entry for {arxiv_id}")
        return TargetMetadata(
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

        params = {"search": title, "per-page": self.per_page}
        if self.mailto:
            params["mailto"] = self.mailto

        url = "https://api.openalex.org/works"
        cache_key = url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        ensure_dir(self.cache_dir)
        cache_path = os.path.join(self.cache_dir, f"{sha256_hash(cache_key)}.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        results = data.get("results") or []
        return self.select_openalex_work_id(results, title, authors)
