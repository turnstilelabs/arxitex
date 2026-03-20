"""Stage 1: fetch OpenAlex citing works for a target and write *_works.jsonl."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from arxitex.arxiv_api import ArxivAPI
from arxitex.arxiv_utils import is_arxiv_url, try_parse_arxiv_id
from arxitex.tools.matching.arxiv_matcher import match_external_reference_to_arxiv
from arxitex.utils import append_jsonl, ensure_dir, sha256_hash, write_json

OPENALEX_BASE = "https://api.openalex.org"
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0


def normalize_openalex_work_id(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    if "?" in s:
        s = s.split("?", 1)[0]
    if "#" in s:
        s = s.split("#", 1)[0]
    s = s.rstrip("/")
    m = re.search(
        r"(?:https?://)?(?:www\.)?openalex\.org/(?:works/)?(w\d+)$",
        s,
        re.I,
    )
    if m:
        return f"https://openalex.org/{m.group(1).upper()}"
    if s.lower().startswith("http"):
        return s
    if s[0].lower() == "w":
        return f"https://openalex.org/{s.upper()}"
    return s


class OpenAlexClient:
    def __init__(
        self,
        *,
        cache_dir: str,
        rate_limit: float,
        mailto: Optional[str],
        api_key: Optional[str],
        per_page: int,
    ) -> None:
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self.mailto = mailto
        self.api_key = api_key
        self.per_page = per_page

    def fetch_json(self, url: str) -> Dict[str, Any]:
        ensure_dir(self.cache_dir)
        cache_path = os.path.join(self.cache_dir, sha256_hash(url) + ".json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Self-heal on partial/corrupt cache writes.
                os.remove(cache_path)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.mailto:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}mailto={requests.utils.quote(self.mailto)}"

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                if self.rate_limit > 0:
                    time.sleep(self.rate_limit)
                return data

            if resp.status_code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"OpenAlex error {resp.status_code}: {resp.text[:200]}"
                )

            last_error = resp
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = BACKOFF_BASE_SECONDS * (2**attempt)
            else:
                delay = BACKOFF_BASE_SECONDS * (2**attempt)
            time.sleep(delay)

        if last_error is not None:
            raise RuntimeError(
                f"OpenAlex error {last_error.status_code}: {last_error.text[:200]}"
            )

        raise RuntimeError("OpenAlex error: request failed after retries.")

    def iter_citing_works(self, target_id: str) -> Iterable[Dict[str, Any]]:
        cursor = "*"
        select = (
            "id,display_name,title,type,publication_year,doi,authorships,cited_by_count,"
            "referenced_works_count,ids,indexed_in,primary_location,best_oa_location,locations"
        )
        while True:
            url = (
                f"{OPENALEX_BASE}/works?"
                f"filter=cites:{target_id}"
                f"&per-page={self.per_page}"
                f"&cursor={requests.utils.quote(cursor)}"
                f"&select={select}"
            )
            data = self.fetch_json(url)
            results = data.get("results") or []
            for w in results:
                yield w

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break


class OpenAlexWorkParser:
    @staticmethod
    def iter_locations(work: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        for key in ["primary_location", "best_oa_location"]:
            loc = work.get(key)
            if isinstance(loc, dict):
                yield loc
        for loc in work.get("locations") or []:
            if isinstance(loc, dict):
                yield loc

    @classmethod
    def location_is_arxiv(cls, loc: Dict[str, Any]) -> bool:
        for field in ["landing_page_url", "pdf_url", "source_url"]:
            v = loc.get(field)
            if isinstance(v, str) and is_arxiv_url(v):
                return True

        raw = loc.get("raw_source_name")
        if isinstance(raw, str) and "arxiv" in raw.lower():
            return True

        src = loc.get("source") or {}
        if isinstance(src, dict):
            name = src.get("display_name") or src.get("host_organization_name")
            if isinstance(name, str) and "arxiv" in name.lower():
                return True
            sid = src.get("id")
            if isinstance(sid, str) and "arxiv" in sid.lower():
                return True

        return False

    @classmethod
    def extract_arxiv_id(cls, work: Dict[str, Any]) -> Optional[str]:
        ids = work.get("ids") or {}
        v = ids.get("arxiv")
        if isinstance(v, str):
            parsed = try_parse_arxiv_id(v)
            if parsed:
                return parsed

        for loc in cls.iter_locations(work):
            for field in ["landing_page_url", "pdf_url", "source_url"]:
                v = loc.get(field)
                if isinstance(v, str) and is_arxiv_url(v):
                    parsed = try_parse_arxiv_id(v)
                    if parsed:
                        return parsed

        return None

    @classmethod
    def collect_source_urls(cls, work: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        for loc in cls.iter_locations(work):
            for field in ["landing_page_url", "pdf_url", "source_url"]:
                v = loc.get(field)
                if isinstance(v, str) and v not in urls:
                    urls.append(v)
        return urls

    @classmethod
    def work_to_record(cls, work: Dict[str, Any], target_id: str) -> Dict[str, Any]:
        arxiv_id = cls.extract_arxiv_id(work)
        source_urls = cls.collect_source_urls(work)
        indexed_in = work.get("indexed_in") or []
        has_arxiv_loc = any(
            cls.location_is_arxiv(loc) for loc in cls.iter_locations(work)
        )
        arxiv_available = bool(arxiv_id) or ("arxiv" in indexed_in) or has_arxiv_loc

        authors = []
        for a in work.get("authorships") or []:
            author = a.get("author") or {}
            name = author.get("display_name") or ""
            orcid = author.get("orcid")
            if name:
                authors.append({"name": name, "orcid": orcid})

        return {
            "openalex_id": work.get("id"),
            "title": work.get("display_name") or work.get("title") or "",
            "type": work.get("type"),
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi"),
            "cited_by_count": work.get("cited_by_count"),
            "referenced_works_count": work.get("referenced_works_count"),
            "authors": authors,
            "indexed_in": indexed_in,
            "arxiv_id": arxiv_id,
            "arxiv_available": arxiv_available,
            "source_urls": source_urls,
            "target_work_ids": [target_id],
        }


def resolve_arxiv_via_fallback(
    work: Dict[str, Any],
    cache_db: str,
    refresh_days: int,
):
    title = work.get("display_name") or work.get("title") or ""
    if not title:
        return None

    authors = []
    for a in work.get("authorships") or []:
        author = a.get("author") or {}
        name = author.get("display_name") or ""
        if name:
            authors.append(name)

    api = ArxivAPI()
    res = match_external_reference_to_arxiv(
        api=api,
        full_reference=title,
        extracted_title=title,
        extracted_authors=authors or None,
        db_path_for_cache=cache_db,
        refresh_days=refresh_days,
    )
    return res if res and res.matched_arxiv_id else None


class OpenAlexCitingWorksStage:
    def __init__(
        self,
        *,
        target_ids: list[str],
        target_id: str,
        out_dir: str,
        cache_dir: str,
        mailto: Optional[str],
        api_key: Optional[str],
        per_page: int,
        max_works: int,
        rate_limit: float,
        fallback_arxiv: bool,
        fallback_cache_db: str,
        fallback_refresh_days: int,
    ) -> None:
        self.target_ids = target_ids
        self.target_id = target_id
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        self.mailto = mailto
        self.api_key = api_key
        self.per_page = per_page
        self.max_works = max_works
        self.rate_limit = rate_limit
        self.fallback_arxiv = fallback_arxiv
        self.fallback_cache_db = fallback_cache_db
        self.fallback_refresh_days = fallback_refresh_days
        self.client = OpenAlexClient(
            cache_dir=cache_dir,
            rate_limit=rate_limit,
            mailto=mailto,
            api_key=api_key,
            per_page=per_page,
        )

    def run(self) -> int:
        ensure_dir(self.out_dir)
        ensure_dir(self.cache_dir)

        target_ids_path = os.path.join(
            self.out_dir, f"{self.target_id}_target_ids.json"
        )
        write_json(target_ids_path, self.target_ids)

        works_path = os.path.join(self.out_dir, f"{self.target_id}_works.jsonl")
        if os.path.exists(works_path):
            os.remove(works_path)

        fetched_total = 0
        records_by_id: Dict[str, Dict[str, Any]] = {}
        anon_records: List[Dict[str, Any]] = []
        for target_work_id in self.target_ids:
            for w in self.client.iter_citing_works(target_work_id):
                rec = OpenAlexWorkParser.work_to_record(w, target_work_id)
                if self.fallback_arxiv and not rec.get("arxiv_available"):
                    match = resolve_arxiv_via_fallback(
                        w,
                        self.fallback_cache_db,
                        self.fallback_refresh_days,
                    )
                    if match:
                        rec["arxiv_available"] = True
                        rec["arxiv_id"] = match.matched_arxiv_id
                        rec["arxiv_fallback_match"] = match.matched_arxiv_id
                        rec["arxiv_fallback_method"] = match.match_method
                        rec["arxiv_fallback_title_score"] = match.title_score
                        rec["arxiv_fallback_author_overlap"] = match.author_overlap
                        rec["arxiv_fallback_query"] = match.arxiv_query
                        rec["arxiv_fallback_title"] = match.matched_title
                        rec["arxiv_fallback_ids"] = {"arxiv": match.matched_arxiv_id}
                work_id = rec.get("openalex_id")
                if isinstance(work_id, str) and work_id:
                    existing = records_by_id.get(work_id)
                    if existing:
                        existing_ids = existing.get("target_work_ids") or []
                        if target_work_id not in existing_ids:
                            existing_ids.append(target_work_id)
                            existing["target_work_ids"] = existing_ids
                    else:
                        records_by_id[work_id] = rec
                else:
                    anon_records.append(rec)
                fetched_total += 1
                unique_total = len(records_by_id) + len(anon_records)
                if self.max_works and unique_total >= self.max_works:
                    break
            unique_total = len(records_by_id) + len(anon_records)
            if self.max_works and unique_total >= self.max_works:
                break

        for work_id in sorted(records_by_id.keys()):
            append_jsonl(works_path, records_by_id[work_id])
        for rec in anon_records:
            append_jsonl(works_path, rec)

        unique_total = len(records_by_id) + len(anon_records)
        print(
            f"Wrote {unique_total} citing works to {works_path} (fetched {fetched_total})"
        )
        print(f"Target IDs: {', '.join(self.target_ids)}")
        return 0
