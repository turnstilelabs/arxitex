from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx
from loguru import logger

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema

ARXIV_VERSION_RE = re.compile(r"^(?P<base>.+)v(?P<vnum>\d+)$")


def strip_arxiv_version(arxiv_id: str) -> str:
    """Normalize an arXiv id to the base work id (strip trailing vN).

    Examples
    --------
    - 2501.01234v3 -> 2501.01234
    - math.AG/0601001v2 -> math.AG/0601001
    - 2501.01234 -> 2501.01234
    """

    arxiv_id = (arxiv_id or "").strip()
    m = ARXIV_VERSION_RE.match(arxiv_id)
    return m.group("base") if m else arxiv_id


@dataclass
class CitationRecord:
    paper_id: str  # base arxiv id
    source: str  # openalex
    source_work_id: Optional[str]
    citation_count: Optional[int]
    last_fetched_at_utc: str
    raw_json: Optional[str] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_paper_citation(db_path: str, rec: CitationRecord) -> None:
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        with conn:
            # Ensure FK target exists
            conn.execute(
                "INSERT OR IGNORE INTO papers (paper_id) VALUES (?)", (rec.paper_id,)
            )
            conn.execute(
                """
                INSERT INTO paper_citations (
                    paper_id, source, source_work_id, citation_count, last_fetched_at_utc, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    source=excluded.source,
                    source_work_id=excluded.source_work_id,
                    citation_count=excluded.citation_count,
                    last_fetched_at_utc=excluded.last_fetched_at_utc,
                    raw_json=excluded.raw_json
                """,
                (
                    rec.paper_id,
                    rec.source,
                    rec.source_work_id,
                    rec.citation_count,
                    rec.last_fetched_at_utc,
                    rec.raw_json,
                ),
            )
    finally:
        conn.close()


def get_existing_citation_timestamp(db_path: str, paper_id: str) -> Optional[str]:
    """Returns last_fetched_at_utc if present."""
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT last_fetched_at_utc FROM paper_citations WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _load_existing_citation_timestamps(db_path: str) -> dict[str, str]:
    """Load all existing citation timestamps into memory.

    This avoids doing 200k individual SQLite lookups (and connections), which would
    make long backfills appear "stuck".
    """

    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT paper_id, last_fetched_at_utc FROM paper_citations"
        ).fetchall()
        return {r[0]: r[1] for r in rows if r and r[0] and r[1]}
    finally:
        conn.close()


async def fetch_openalex_citation(
    client: httpx.AsyncClient,
    *,
    base_arxiv_id: str,
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    mailto: Optional[str] = None,
) -> CitationRecord:
    """Fetch total citation count from OpenAlex.

    OpenAlex does not support filtering by `ids.arxiv` directly.
    We therefore do a best-effort match via `search` using paper metadata.
    """

    # OpenAlex does NOT support filtering by ids.arxiv directly.
    # We therefore use a best-effort metadata match via `search`.
    # Strategy:
    # - search by title (best signal)
    # - pick the best candidate by title similarity, then check author overlap
    # - use that candidate's `cited_by_count`

    q = title or base_arxiv_id
    params = {"search": q, "per-page": 25}
    if mailto:
        params["mailto"] = mailto

    url = "https://api.openalex.org/works"
    resp = await client.get(url, params=params)

    # Basic rate limit handling
    if resp.status_code == 429:
        raise httpx.HTTPStatusError("rate_limited", request=resp.request, response=resp)
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()
    results = data.get("results") or []

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    wanted_title = norm(title or "")
    wanted_authors = {norm(a) for a in (authors or []) if a}

    best = None
    best_score = -1.0
    for r in results:
        rt = norm(r.get("title") or "")
        if wanted_title:
            title_score = SequenceMatcher(a=wanted_title, b=rt).ratio()
        else:
            title_score = 0.0

        # authors: OpenAlex uses authorships[].author.display_name
        rauths = {
            norm((au.get("author") or {}).get("display_name") or "")
            for au in (r.get("authorships") or [])
            if isinstance(au, dict)
        }
        rauths = {a for a in rauths if a}
        author_overlap = 0.0
        if wanted_authors and rauths:
            author_overlap = len(wanted_authors.intersection(rauths)) / max(
                1, len(wanted_authors)
            )

        # final score: title dominates; author overlap breaks ties
        score = title_score * 0.9 + author_overlap * 0.1

        if score > best_score:
            best_score = score
            best = r

    # If we have a title, require a minimal similarity to avoid garbage matches.
    if title and (best is None or best_score < 0.6):
        return CitationRecord(
            paper_id=base_arxiv_id,
            source="openalex",
            source_work_id=None,
            citation_count=None,
            last_fetched_at_utc=_utc_now_iso(),
            raw_json=None,
        )

    return CitationRecord(
        paper_id=base_arxiv_id,
        source="openalex",
        source_work_id=best.get("id"),
        citation_count=best.get("cited_by_count"),
        last_fetched_at_utc=_utc_now_iso(),
        raw_json=json.dumps(best),
    )


async def backfill_citations_openalex(
    *,
    db_path: str,
    arxiv_ids: list[str],
    paper_metadata_by_id: Optional[dict[str, dict]] = None,
    workers: int = 8,
    qps: float = 1.0,
    refresh_days: int = 30,
    mailto: Optional[str] = None,
    max_papers: Optional[int] = None,
) -> dict:
    """Backfill citations for given arXiv IDs.

    - Normalizes to base ids
    - Skips recently fetched entries (refresh_days)
    """

    ensure_schema(db_path)

    base_ids = [strip_arxiv_version(a) for a in arxiv_ids if a]
    # stable unique
    seen = set()
    uniq = []
    for b in base_ids:
        if b not in seen:
            seen.add(b)
            uniq.append(b)

    if max_papers is not None:
        uniq = uniq[:max_papers]

    cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_days)

    # filter stale/missing
    existing_ts = _load_existing_citation_timestamps(db_path)
    to_fetch: list[str] = []
    for bid in uniq:
        ts = existing_ts.get(bid)
        if not ts:
            to_fetch.append(bid)
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            to_fetch.append(bid)
            continue
        if dt < cutoff:
            to_fetch.append(bid)

    logger.info(
        f"OpenAlex citation backfill: unique={len(uniq)} to_fetch={len(to_fetch)} refresh_days={refresh_days}"
    )

    workers = max(1, workers)
    qps = max(0.05, float(qps))
    stats = {"attempted": 0, "success": 0, "missing": 0, "failed": 0}

    queue: asyncio.Queue[str] = asyncio.Queue()
    for bid in to_fetch:
        queue.put_nowait(bid)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:

        # Global throttling across all workers.
        throttle_lock = asyncio.Lock()
        last_request_at = 0.0

        async def throttle():
            nonlocal last_request_at
            min_interval = 1.0 / qps
            async with throttle_lock:
                now = asyncio.get_event_loop().time()
                wait = (last_request_at + min_interval) - now
                if wait > 0:
                    await asyncio.sleep(wait)
                last_request_at = asyncio.get_event_loop().time()

        async def worker_loop():
            while True:
                try:
                    bid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                stats["attempted"] += 1

                meta = (paper_metadata_by_id or {}).get(bid) or {}
                title = meta.get("title")
                authors = (
                    meta.get("authors")
                    if isinstance(meta.get("authors"), list)
                    else None
                )

                delay = 1.0
                for attempt in range(5):
                    try:
                        await throttle()
                        rec = await fetch_openalex_citation(
                            client,
                            base_arxiv_id=bid,
                            title=title,
                            authors=authors,
                            mailto=mailto,
                        )
                        upsert_paper_citation(db_path, rec)
                        if rec.citation_count is None:
                            stats["missing"] += 1
                        else:
                            stats["success"] += 1
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response is not None and e.response.status_code == 429:
                            # Back off more aggressively on rate limit.
                            await asyncio.sleep(max(delay, 5.0))
                            delay = min(delay * 2, 120)
                            continue
                        stats["failed"] += 1
                        logger.debug(f"OpenAlex fetch failed for {bid}: {e}")
                        break
                    except Exception as e:
                        if attempt < 4:
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 30)
                            continue
                        stats["failed"] += 1
                        logger.debug(f"OpenAlex fetch failed for {bid}: {e}")
                        break

                queue.task_done()

                # Periodic progress log (avoid spam)
                if stats["attempted"] % 500 == 0:
                    logger.info(
                        "OpenAlex progress: attempted={a} success={s} missing={m} failed={f} remaining={r}".format(
                            a=stats["attempted"],
                            s=stats["success"],
                            m=stats["missing"],
                            f=stats["failed"],
                            r=queue.qsize(),
                        )
                    )

        await asyncio.gather(*[worker_loop() for _ in range(workers)])

    return {"unique": len(uniq), "to_fetch": len(to_fetch), **stats}
