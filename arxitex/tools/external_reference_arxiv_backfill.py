from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema
from arxitex.tools.external_reference_arxiv_matcher import (
    MatchResult,
    extract_title_and_authors,
    generate_title_candidates,
    match_external_reference_to_arxiv,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BackfillStats:
    considered: int = 0
    skipped_fresh: int = 0
    direct_regex: int = 0
    matched: int = 0
    no_title: int = 0
    none: int = 0
    failed: int = 0


def _load_existing_match_timestamps(db_path: str) -> dict[tuple[str, str], str]:
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT paper_id, external_artifact_id, last_matched_at_utc
            FROM external_reference_arxiv_matches
            """
        ).fetchall()
        out: dict[tuple[str, str], str] = {}
        for r in rows:
            if not r:
                continue
            out[(str(r[0]), str(r[1]))] = str(r[2])
        return out
    finally:
        conn.close()


def _iter_external_reference_artifacts(
    conn,
    *,
    only_paper_ids: Optional[list[str]] = None,
    max_refs: Optional[int] = None,
) -> list[dict]:
    sql = (
        "SELECT paper_id, artifact_id, content_tex "
        "FROM artifacts "
        "WHERE artifact_type = 'external_reference'"
    )
    params: list = []
    if only_paper_ids:
        placeholders = ",".join(["?"] * len(only_paper_ids))
        sql += f" AND paper_id IN ({placeholders})"
        params.extend(list(only_paper_ids))
    sql += " ORDER BY paper_id, artifact_id"
    if max_refs is not None:
        sql += " LIMIT ?"
        params.append(int(max_refs))
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def _load_successfully_processed_arxiv_ids(db_path: str) -> list[str]:
    """Return arxiv_id list from processed_papers where status LIKE 'success%'.

    We treat processed_papers as the authoritative indicator that the pipeline
    successfully handled a paper (graph built, or a success variant).
    """

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT arxiv_id FROM processed_papers WHERE status LIKE 'success%'"
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]
    finally:
        conn.close()


def upsert_match_row(
    conn,
    *,
    paper_id: str,
    external_artifact_id: str,
    full_reference: str,
    res: MatchResult,
) -> None:
    # Extracted metadata should be stored even on misses (for debugging).
    extracted_title, extracted_authors = extract_title_and_authors(full_reference)
    conn.execute(
        """
        INSERT INTO external_reference_arxiv_matches (
            paper_id, external_artifact_id, matched_arxiv_id, match_method,
            extracted_title, extracted_authors_json,
            matched_title, matched_authors_json,
            title_score, author_overlap, arxiv_query,
            last_matched_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, external_artifact_id) DO UPDATE SET
            matched_arxiv_id=excluded.matched_arxiv_id,
            match_method=excluded.match_method,
            extracted_title=excluded.extracted_title,
            extracted_authors_json=excluded.extracted_authors_json,
            matched_title=excluded.matched_title,
            matched_authors_json=excluded.matched_authors_json,
            title_score=excluded.title_score,
            author_overlap=excluded.author_overlap,
            arxiv_query=excluded.arxiv_query,
            last_matched_at_utc=excluded.last_matched_at_utc
        """,
        (
            paper_id,
            external_artifact_id,
            res.matched_arxiv_id,
            res.match_method,
            extracted_title,
            json.dumps(extracted_authors or [], ensure_ascii=False),
            res.matched_title,
            json.dumps(res.matched_authors or [], ensure_ascii=False),
            res.title_score,
            res.author_overlap,
            res.arxiv_query,
            _utc_now_iso(),
        ),
    )


async def backfill_external_reference_arxiv_matches(
    *,
    db_path: str | Path,
    only_paper_ids: Optional[list[str]] = None,
    only_processed_success: bool = False,
    max_refs: Optional[int] = None,
    qps: float = 1.0,
    refresh_days: int = 30,
    force: bool = False,
    verbose: bool = False,
) -> BackfillStats:
    """Backfill arXiv matches for external references already stored in SQLite."""

    db_path = str(db_path)
    ensure_schema(db_path)

    if only_processed_success:
        only_paper_ids = _load_successfully_processed_arxiv_ids(db_path)
        logger.info(
            "External ref backfill: restricting to processed_papers status LIKE 'success%%' (count={})",
            len(only_paper_ids),
        )

    existing_ts = _load_existing_match_timestamps(db_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(refresh_days)))

    conn = connect(db_path)
    try:
        refs = _iter_external_reference_artifacts(
            conn, only_paper_ids=only_paper_ids, max_refs=max_refs
        )
    finally:
        conn.close()

    logger.info(
        f"External ref arXiv backfill: candidates={len(refs)} refresh_days={refresh_days} force={force}"
    )

    api = ArxivAPI()
    stats = BackfillStats()

    # Global throttling similar to OpenAlex tool
    throttle_lock = asyncio.Lock()
    last_request_at = 0.0

    async def throttle():
        nonlocal last_request_at
        min_interval = 1.0 / max(0.05, float(qps))
        async with throttle_lock:
            now = asyncio.get_event_loop().time()
            wait = (last_request_at + min_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            last_request_at = asyncio.get_event_loop().time()

    for r in refs:
        paper_id = str(r["paper_id"])
        external_artifact_id = str(r["artifact_id"])
        full_reference = str(r.get("content_tex") or "")

        stats.considered += 1

        if verbose:
            logger.info(
                "[extref] candidate paper_id={} artifact_id={} ref_snippet={!r}",
                paper_id,
                external_artifact_id,
                full_reference[:160],
            )

            # Candidate introspection (up to 4) so we can understand bad queries.
            try:
                cands, _authors = generate_title_candidates(full_reference, limit=4)
                logger.info(
                    "[extref] candidates paper_id={} artifact_id={} -> {}",
                    paper_id,
                    external_artifact_id,
                    [
                        {
                            "title": c.title,
                            "method": c.method,
                            "score": round(float(c.score), 3),
                        }
                        for c in cands
                    ],
                )
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "[extref] candidate_generation_failed paper_id={} artifact_id={} err={!r}",
                    paper_id,
                    external_artifact_id,
                    e,
                )

        ts = existing_ts.get((paper_id, external_artifact_id))
        if ts and not force:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt is not None and dt >= cutoff:
                stats.skipped_fresh += 1
                continue

        try:
            # Throttle only around actual API calls. The matcher has a direct
            # regex fast-path and a DB cache fast-path.
            await throttle()
            res = match_external_reference_to_arxiv(
                api=api,
                full_reference=full_reference,
                db_path_for_cache=db_path,
                refresh_cache=force,
                refresh_days=refresh_days,
            )

            if verbose:
                logger.info(
                    "[extref] decision paper_id={} artifact_id={} method={} extracted_title={!r} extracted_authors={} "
                    "query={!r} matched_arxiv_id={} matched_title={!r} matched_authors={} title_score={} author_overlap={} ",
                    paper_id,
                    external_artifact_id,
                    res.match_method,
                    res.extracted_title,
                    res.extracted_authors,
                    res.arxiv_query,
                    res.matched_arxiv_id,
                    res.matched_title,
                    res.matched_authors,
                    res.title_score,
                    res.author_overlap,
                )

            # Persist
            connw = connect(db_path)
            try:
                with connw:
                    upsert_match_row(
                        connw,
                        paper_id=paper_id,
                        external_artifact_id=external_artifact_id,
                        full_reference=full_reference,
                        res=res,
                    )
            finally:
                connw.close()

            if res.match_method == "direct_regex":
                stats.direct_regex += 1
            if res.matched_arxiv_id:
                stats.matched += 1
            else:
                if res.extracted_title is None:
                    stats.no_title += 1
                stats.none += 1

        except Exception as e:
            stats.failed += 1
            logger.warning(
                f"Failed matching paper_id={paper_id} external_artifact_id={external_artifact_id}: {e}",
                exc_info=True,
            )

        if stats.considered % 250 == 0:
            logger.info(
                "Progress: considered={c} matched={m} skipped_fresh={s} failed={f}".format(
                    c=stats.considered,
                    m=stats.matched,
                    s=stats.skipped_fresh,
                    f=stats.failed,
                )
            )

    try:
        api.close()
    except Exception:
        pass

    logger.info(
        "Backfill done: considered={c} matched={m} direct_regex={d} skipped_fresh={s} none={n} failed={f}".format(
            c=stats.considered,
            m=stats.matched,
            d=stats.direct_regex,
            s=stats.skipped_fresh,
            n=stats.none,
            f=stats.failed,
        )
    )

    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Backfill arXiv IDs for external reference artifacts by searching the arXiv API "
            "using heuristically extracted titles/authors."
        )
    )
    p.add_argument(
        "--db-path",
        required=True,
        help="Path to the arxitex SQLite database.",
    )
    p.add_argument(
        "--paper-id",
        action="append",
        default=None,
        help="Restrict to a specific paper_id (repeatable).",
    )
    p.add_argument(
        "--only-processed-success",
        action="store_true",
        help=(
            "Restrict to paper_ids in processed_papers with status LIKE 'success%'. "
            "Useful to avoid processing papers that failed upstream."
        ),
    )
    p.add_argument(
        "--max-refs",
        type=int,
        default=None,
        help="Limit number of external references (for testing).",
    )
    p.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help="Global arXiv request rate limit (requests/second).",
    )
    p.add_argument(
        "--refresh-days",
        type=int,
        default=30,
        help="Re-query if match row older than this many days.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force refresh (ignore refresh-days and cache freshness).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-reference extraction and matching details.",
    )

    args = p.parse_args(argv)

    asyncio.run(
        backfill_external_reference_arxiv_matches(
            db_path=args.db_path,
            only_paper_ids=args.paper_id,
            only_processed_success=bool(args.only_processed_success),
            max_refs=args.max_refs,
            qps=args.qps,
            refresh_days=args.refresh_days,
            force=bool(args.force),
            verbose=bool(args.verbose),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
