#!/usr/bin/env python3
"""Stage 1: build list of works that cite a target work via OpenAlex.

Optional fallback:
- If --fallback-arxiv is enabled and a citing work has no arXiv signal,
  the script uses the shared arXiv matcher utility to query the arXiv API
  by title/author and mark the work as arXiv-available when matched.

Outputs:
- data/{target}_target_ids.json
- data/{target}_works.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from arxitex.arxiv_api import ArxivAPI
from arxitex.tools.citations.arxiv_matcher import match_external_reference_to_arxiv
from arxitex.tools.citations.dataset.utils import append_jsonl, ensure_dir, sha256_hash

OPENALEX_BASE = "https://api.openalex.org"
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", re.I)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fetch_json(
    url: str,
    cache_dir: str,
    rate_limit: float,
    mailto: Optional[str],
    api_key: Optional[str],
) -> Dict[str, Any]:
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, sha256_hash(url) + ".json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    params = None
    if mailto:
        # OpenAlex supports mailto for polite usage.
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}mailto={requests.utils.quote(mailto)}"

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            if rate_limit > 0:
                time.sleep(rate_limit)
            return data

        if resp.status_code not in {429, 500, 502, 503, 504}:
            raise RuntimeError(f"OpenAlex error {resp.status_code}: {resp.text[:200]}")

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


def normalize_openalex_work_id(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    if "?" in s:
        s = s.split("?", 1)[0]
    if "#" in s:
        s = s.split("#", 1)[0]
    s = s.rstrip("/")
    # Accept full URL or short work id (e.g. W123...). Normalize to https URL.
    m = re.search(r"(?:openalex\\.org/)?(?:works/)?(w\\d+)$", s, re.I)
    if m:
        return f"https://openalex.org/{m.group(1).upper()}"
    if s.lower().startswith("http"):
        return s
    if s[0].lower() == "w":
        return f"https://openalex.org/{s.upper()}"
    return s


def is_arxiv_url(url: str) -> bool:
    u = url.lower()
    return (
        "arxiv.org" in u
        or "export.arxiv.org" in u
        or "ar5iv.org" in u
        or "ar5iv.labs" in u
    )


def iter_all_locations(work: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Include primary and best OA locations if present.
    for key in ["primary_location", "best_oa_location"]:
        loc = work.get(key)
        if isinstance(loc, dict):
            yield loc
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            yield loc


def location_is_arxiv(loc: Dict[str, Any]) -> bool:
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


def extract_arxiv_id(work: Dict[str, Any]) -> Optional[str]:
    ids = work.get("ids") or {}
    v = ids.get("arxiv")
    if isinstance(v, str):
        m = ARXIV_ID_RE.search(v)
        if m:
            return m.group(1)

    for loc in iter_all_locations(work):
        for field in ["landing_page_url", "pdf_url", "source_url"]:
            v = loc.get(field)
            if isinstance(v, str) and is_arxiv_url(v):
                m = ARXIV_ID_RE.search(v)
                if m:
                    return m.group(1)

    return None


def collect_source_urls(work: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for loc in iter_all_locations(work):
        for field in ["landing_page_url", "pdf_url", "source_url"]:
            v = loc.get(field)
            if isinstance(v, str) and v not in urls:
                urls.append(v)
    return urls


def work_to_record(work: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    arxiv_id = extract_arxiv_id(work)
    source_urls = collect_source_urls(work)
    indexed_in = work.get("indexed_in") or []
    has_arxiv_loc = any(location_is_arxiv(loc) for loc in iter_all_locations(work))
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
    """Return MatchResult when arXiv fallback succeeds, else None."""
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


def iter_citing_works(
    target_id: str,
    cache_dir: str,
    rate_limit: float,
    mailto: Optional[str],
    api_key: Optional[str],
    per_page: int,
) -> Iterable[Dict[str, Any]]:
    cursor = "*"
    select = "id,display_name,title,type,publication_year,doi,authorships,cited_by_count,referenced_works_count,ids,indexed_in,primary_location,best_oa_location,locations"
    while True:
        url = (
            f"{OPENALEX_BASE}/works?"
            f"filter=cites:{target_id}"
            f"&per-page={per_page}"
            f"&cursor={requests.utils.quote(cursor)}"
            f"&select={select}"
        )
        data = fetch_json(url, cache_dir, rate_limit, mailto, api_key)
        results = data.get("results") or []
        for w in results:
            yield w

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build list of OpenAlex works citing a target work."
    )
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument(
        "--target-id", default="target", help="Output file prefix (target id)."
    )
    parser.add_argument(
        "--cache-dir", default="data/cache", help="Cache directory for API responses."
    )
    parser.add_argument(
        "--mailto", default=None, help="Contact email for OpenAlex polite usage."
    )
    parser.add_argument("--api-key", default=None, help="OpenAlex API key (optional).")
    parser.add_argument(
        "--target-ids",
        nargs="*",
        default=None,
        help="OpenAlex Work IDs to use as citation targets.",
    )
    parser.add_argument(
        "--target-ids-file",
        default=None,
        help="Path to a JSON file containing a list of OpenAlex Work IDs.",
    )
    parser.add_argument(
        "--seed-ids",
        nargs="*",
        default=None,
        help="Deprecated. Use --target-ids instead.",
    )
    parser.add_argument(
        "--seed-ids-file",
        default=None,
        help="Deprecated. Use --target-ids-file instead.",
    )
    parser.add_argument(
        "--per-page", type=int, default=200, help="OpenAlex per-page size (max 200)."
    )
    parser.add_argument(
        "--max-works",
        type=int,
        default=0,
        help="Stop after N citing works (0 = no limit).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds to sleep between OpenAlex requests.",
    )
    parser.add_argument(
        "--fallback-arxiv",
        action="store_true",
        help="Try to find an arXiv version via the arXiv matcher when not marked arXiv.",
    )
    parser.add_argument(
        "--fallback-cache-db",
        default=None,
        help="SQLite cache DB for arXiv fallback (default: {cache_dir}/arxiv_fallback_cache.db).",
    )
    parser.add_argument(
        "--fallback-refresh-days",
        type=int,
        default=30,
        help="Refresh cached arXiv matches older than N days.",
    )
    args = parser.parse_args(argv)

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)
    if args.fallback_cache_db is None:
        args.fallback_cache_db = os.path.join(args.cache_dir, "arxiv_fallback_cache.db")

    target_ids: List[str] = []
    target_id = args.target_id

    if args.seed_ids or args.seed_ids_file:
        print(
            "Warning: --seed-ids/--seed-ids-file are deprecated. Use --target-ids/--target-ids-file.",
            file=sys.stderr,
        )

    ids_file = args.target_ids_file or args.seed_ids_file
    if ids_file:
        with open(ids_file, "r", encoding="utf-8") as f:
            target_ids = json.load(f)
        target_ids = [
            normalize_openalex_work_id(s) for s in target_ids if isinstance(s, str)
        ]
    elif args.target_ids or args.seed_ids:
        raw_ids = args.target_ids or args.seed_ids or []
        target_ids = [normalize_openalex_work_id(s) for s in raw_ids]
    else:
        print(
            "No target IDs provided. Pass --target-ids/--target-ids-file.",
            file=sys.stderr,
        )
        return 2

    target_ids_path = os.path.join(args.out_dir, f"{target_id}_target_ids.json")
    _write_json(target_ids_path, target_ids)

    works_path = os.path.join(args.out_dir, f"{target_id}_works.jsonl")
    if os.path.exists(works_path):
        os.remove(works_path)

    fetched_total = 0
    records_by_id: Dict[str, Dict[str, Any]] = {}
    anon_records: List[Dict[str, Any]] = []
    for target_work_id in target_ids:
        for w in iter_citing_works(
            target_work_id,
            args.cache_dir,
            args.rate_limit,
            args.mailto,
            args.api_key,
            args.per_page,
        ):
            rec = work_to_record(w, target_work_id)
            if args.fallback_arxiv and not rec.get("arxiv_available"):
                match = resolve_arxiv_via_fallback(
                    w,
                    args.fallback_cache_db,
                    args.fallback_refresh_days,
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
            if args.max_works and fetched_total >= args.max_works:
                break
        if args.max_works and fetched_total >= args.max_works:
            break

    for work_id in sorted(records_by_id.keys()):
        append_jsonl(works_path, records_by_id[work_id])
    for rec in anon_records:
        append_jsonl(works_path, rec)

    unique_total = len(records_by_id) + len(anon_records)
    print(
        f"Wrote {unique_total} citing works to {works_path} (fetched {fetched_total})"
    )
    print(f"Target IDs: {', '.join(target_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
