from __future__ import annotations

import asyncio
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
ARXIV_MODERN_RE = re.compile(r"^\d{4}\.\d{4,5}$")


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
                    paper_id, source, source_work_id, citation_count, last_fetched_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    source=excluded.source,
                    source_work_id=excluded.source_work_id,
                    citation_count=excluded.citation_count,
                    last_fetched_at_utc=excluded.last_fetched_at_utc
                """,
                (
                    rec.paper_id,
                    rec.source,
                    rec.source_work_id,
                    rec.citation_count,
                    rec.last_fetched_at_utc,
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
    # - filter to high-confidence matches (title similarity + author overlap)
    # - return the max cited_by_count among those matches

    def _mk_empty() -> CitationRecord:
        return CitationRecord(
            paper_id=base_arxiv_id,
            source="openalex",
            source_work_id=None,
            citation_count=None,
            last_fetched_at_utc=_utc_now_iso(),
        )

    # NOTE: We intentionally prefer search-based matching even when the arXiv DOI
    # exists. OpenAlex often has both an arXiv preprint Work (sometimes 0 cites)
    # and a published journal Work (with the citations). We want the best-cited
    # high-confidence match.

    # Search-based matching (title + authors)
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

    def norm_author(s: str) -> str:
        """Normalize author names to improve matching.

        OpenAlex display_name can appear as "First Last" or "Last, First".
        We normalize by:
        - lowercasing
        - removing punctuation
        - if a comma is present, swapping order
        """

        t = (s or "").strip().lower()
        if not t:
            return ""
        if "," in t:
            parts = [p.strip() for p in t.split(",") if p.strip()]
            if len(parts) >= 2:
                t = " ".join(parts[1:] + [parts[0]])
        # Drop punctuation and collapse whitespace
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    wanted_title = norm(title or "")
    wanted_authors = {norm_author(a) for a in (authors or []) if a}
    wanted_authors = {a for a in wanted_authors if a}

    # Build a set of high-confidence candidates and pick the max cited_by_count.
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for r in results:
        rt = norm(r.get("title") or "")
        if wanted_title:
            title_score = SequenceMatcher(a=wanted_title, b=rt).ratio()
        else:
            title_score = 0.0

        # authors: OpenAlex uses authorships[].author.display_name
        rauths = {
            norm_author((au.get("author") or {}).get("display_name") or "")
            for au in (r.get("authorships") or [])
            if isinstance(au, dict)
        }
        rauths = {a for a in rauths if a}
        author_overlap = 0.0
        if wanted_authors and rauths:
            author_overlap = len(wanted_authors.intersection(rauths)) / max(
                1, len(wanted_authors)
            )

        candidates.append((title_score, author_overlap, r))

    if not candidates:
        return _mk_empty()

    # Filter: strict title similarity + at least one shared author when we have
    # author metadata.
    HIGH_TITLE = 0.92
    MIN_AUTHOR_OVERLAP = 0.10
    filtered: list[dict[str, Any]] = []
    for title_score, author_overlap, r in candidates:
        if title and title_score < HIGH_TITLE:
            continue
        if wanted_authors and author_overlap <= 0.0:
            continue
        # If we have authors, prefer some overlap; keep a small floor to avoid
        # rejecting cases where OpenAlex author strings differ slightly.
        if wanted_authors and author_overlap < MIN_AUTHOR_OVERLAP:
            continue
        filtered.append(r)

    if title and not filtered:
        return _mk_empty()

    if not filtered:
        # No title provided; fall back to best of unfiltered.
        filtered = [r for _, _, r in candidates]

    def cited(w: dict[str, Any]) -> int:
        try:
            v = w.get("cited_by_count")
            return int(v) if v is not None else -1
        except Exception:
            return -1

    best = max(filtered, key=cited)

    return CitationRecord(
        paper_id=base_arxiv_id,
        source="openalex",
        source_work_id=best.get("id"),
        citation_count=best.get("cited_by_count"),
        last_fetched_at_utc=_utc_now_iso(),
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

    cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_days)

    # filter stale/missing
    existing_ts = _load_existing_citation_timestamps(db_path)
    to_fetch: list[str] = []
    for bid in uniq:
        need_fetch = False
        ts = existing_ts.get(bid)
        if not ts:
            need_fetch = True
        else:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                need_fetch = True
            else:
                if dt < cutoff:
                    need_fetch = True

        if need_fetch:
            to_fetch.append(bid)
            if max_papers is not None and len(to_fetch) >= max_papers:
                break

    logger.info(
        f"OpenAlex citation backfill: unique={len(uniq)} to_fetch={len(to_fetch)} refresh_days={refresh_days}"
    )

    # Track how many papers changed from citation_count=0 -> >0 in this run.
    existing_counts: dict[str, int | None] = {}
    try:
        conn0 = connect(db_path)
        rows0 = conn0.execute(
            "SELECT paper_id, citation_count FROM paper_citations"
        ).fetchall()
        existing_counts = {
            (r[0] if r else None): (r[1] if r else None) for r in rows0 if r and r[0]
        }
    finally:
        try:
            conn0.close()  # type: ignore[name-defined]
        except Exception:
            pass

    upgrades_0_to_pos = 0
    upgrade_examples: list[tuple[str, int]] = []

    # Aggregate HTTP error visibility (to avoid "silent" stalls).
    http_400 = 0
    http_429 = 0
    http_other = 0
    ex_400: list[str] = []
    ex_429: list[str] = []
    ex_other: list[str] = []

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
            # Track scalar metrics as nonlocal so we can update them from
            # within this worker loop. List-typed collections are mutated
            # in-place and do not need nonlocal declarations.
            nonlocal upgrades_0_to_pos, http_400, http_429, http_other
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

                        prev = existing_counts.get(bid)
                        upsert_paper_citation(db_path, rec)
                        if rec.citation_count is None:
                            stats["missing"] += 1
                        else:
                            stats["success"] += 1

                        # Log upgrades from 0 -> positive citations.
                        try:
                            if prev == 0 and (rec.citation_count or 0) > 0:
                                upgrades_0_to_pos += 1
                                if len(upgrade_examples) < 10:
                                    upgrade_examples.append(
                                        (bid, int(rec.citation_count))
                                    )
                        except Exception:
                            pass
                        break
                    except httpx.HTTPStatusError as e:
                        status = (
                            e.response.status_code if e.response is not None else None
                        )
                        if status == 429:
                            http_429 += 1
                            if len(ex_429) < 10:
                                ex_429.append(bid)
                            # Back off more aggressively on rate limit.
                            await asyncio.sleep(max(delay, 5.0))
                            delay = min(delay * 2, 120)
                            continue

                        if status == 400:
                            http_400 += 1
                            if len(ex_400) < 10:
                                ex_400.append(bid)
                            # 400 is usually a bad query/title encoding. Don't retry.
                            stats["failed"] += 1
                            logger.warning(
                                f"OpenAlex 400 for {bid} (bad request). Example query/title likely contains TeX/symbols."
                            )
                            break

                        http_other += 1
                        if len(ex_other) < 10:
                            ex_other.append(bid)
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

                # Periodic progress log (avoid spam). Log more frequently at the
                # start of long runs to avoid "stuck" perception.
                log_every = 100 if stats["attempted"] < 2000 else 500
                if stats["attempted"] % log_every == 0:
                    logger.info(
                        "OpenAlex progress: attempted={a} success={s} missing={m} failed={f} remaining={r}".format(
                            a=stats["attempted"],
                            s=stats["success"],
                            m=stats["missing"],
                            f=stats["failed"],
                            r=queue.qsize(),
                        )
                    )

                    if http_400 or http_429 or http_other:
                        msg = f"HTTP errors so far: 400={http_400}, 429={http_429}, other={http_other}"
                        if ex_400:
                            msg += f" | eg 400: {', '.join(ex_400[:5])}"
                        if ex_429:
                            msg += f" | eg 429: {', '.join(ex_429[:5])}"
                        if ex_other:
                            msg += f" | eg other: {', '.join(ex_other[:5])}"
                        logger.info(msg)
                    if upgrades_0_to_pos:
                        ex = ", ".join([f"{pid}->{c}" for pid, c in upgrade_examples])
                        logger.info(
                            f"Upgrades 0->>0 so far: {upgrades_0_to_pos} (examples: {ex})"
                        )

        await asyncio.gather(*[worker_loop() for _ in range(workers)])

    return {
        "unique": len(uniq),
        "to_fetch": len(to_fetch),
        "upgrades_0_to_pos": upgrades_0_to_pos,
        "upgrade_examples": upgrade_examples,
        "http_400": http_400,
        "http_429": http_429,
        "http_other": http_other,
        **stats,
    }
