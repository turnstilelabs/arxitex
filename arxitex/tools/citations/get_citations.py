#!/usr/bin/env python3
"""Stage 2: identify arXiv-available works and extract target mention contexts.

Inputs:
- data/{target}_works.jsonl (from stage 1)

Outputs:
- data/{target}_arxiv_works.jsonl
- data/{target}_mentions.jsonl
- data/{target}_failures.jsonl
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.arxiv_utils import (
    choose_pdf_url,
    extract_arxiv_id_from_urls,
    normalize_arxiv_id,
    parse_arxiv_id,
)
from arxitex.tools.citations.mention_extraction import MentionExtractor
from arxitex.tools.citations.target_resolution import OpenAlexTargetResolver
from arxitex.tools.citations.utils import (
    append_jsonl,
    ensure_dir,
    read_jsonl,
    sha256_hash,
)


class HostThrottle:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = asyncio.Lock()
        self._last: Dict[str, float] = {}

    async def wait(self, url: str) -> None:
        if self.min_interval <= 0:
            return
        host = urlparse(url).netloc or "default"
        delay = 0.0
        async with self._lock:
            now = time.monotonic()
            last = self._last.get(host)
            if last is not None:
                delay = max(0.0, self.min_interval - (now - last))
                self._last[host] = now + delay
            else:
                self._last[host] = now
        if delay > 0:
            await asyncio.sleep(delay)


async def fetch_to_cache(
    url: str,
    cache_dir: str,
    ext: str,
    session: aiohttp.ClientSession,
    throttle: HostThrottle,
    *,
    offline: bool = False,
) -> str:
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, sha256_hash(url) + ext)
    if os.path.exists(cache_path):
        return cache_path
    if offline:
        raise RuntimeError(f"Offline mode enabled and cache miss for {url}")

    await throttle.wait(url)

    headers = {
        "User-Agent": "arxitex/0.1 (citation mention extractor)",
    }
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} for {url}: {text[:200]}")
        content = await resp.read()

    with open(cache_path, "wb") as f:
        f.write(content)

    return cache_path


class MentionExtractionStage:
    def __init__(
        self,
        *,
        works_file: str,
        target_title: str,
        target_id: str,
        out_dir: str,
        cache_dir: str,
        rate_limit: float,
        max_works: int,
        no_pdf: bool,
        concurrency: int,
        offline: bool,
    ) -> None:
        self.works_file = works_file
        self.target_title = target_title
        self.target_id = target_id
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self.max_works = max_works
        self.no_pdf = no_pdf
        self.concurrency = concurrency
        self.offline = offline
        self.extractor = MentionExtractor(target_title=target_title)

    async def run(self) -> int:
        ensure_dir(self.out_dir)
        ensure_dir(self.cache_dir)

        arxiv_works_path = os.path.join(
            self.out_dir, f"{self.target_id}_arxiv_works.jsonl"
        )
        mentions_path = os.path.join(self.out_dir, f"{self.target_id}_mentions.jsonl")
        failures_path = os.path.join(self.out_dir, f"{self.target_id}_failures.jsonl")

        for p in [arxiv_works_path, mentions_path, failures_path]:
            if os.path.exists(p):
                os.remove(p)

        logger.info("Stage 2: scanning works from {}", self.works_file)
        logger.info(
            "Outputs: {}", ", ".join([arxiv_works_path, mentions_path, failures_path])
        )
        logger.info("Cache dir: {}", self.cache_dir)
        logger.info(
            "Concurrency: {} | Rate limit: {}s per host",
            self.concurrency,
            self.rate_limit,
        )

        works = list(read_jsonl(self.works_file))
        total = len(works)

        arxiv_works: List[Dict[str, Any]] = []
        for w in works:
            arxiv_id = w.get("arxiv_id") or extract_arxiv_id_from_urls(
                w.get("source_urls") or []
            )
            if not arxiv_id:
                indexed_in = w.get("indexed_in") or []
                if "arxiv" in indexed_in:
                    append_jsonl(
                        failures_path,
                        {
                            "openalex_id": w.get("openalex_id"),
                            "title": w.get("title"),
                            "stage": "arxiv_id_missing",
                            "error": "indexed_in_arxiv_but_no_arxiv_id_or_url",
                        },
                    )
                    logger.warning(
                        "Indexed in arXiv but no arXiv ID/URL: {}",
                        w.get("openalex_id"),
                    )
                continue
            arxiv_id = normalize_arxiv_id(arxiv_id)
            w["arxiv_id"] = arxiv_id
            arxiv_works.append(w)

        if self.max_works:
            arxiv_works = arxiv_works[: self.max_works]

        logger.info("Total works: {} | arXiv works: {}", total, len(arxiv_works))

        locks = {
            "arxiv": asyncio.Lock(),
            "mentions": asyncio.Lock(),
            "failures": asyncio.Lock(),
            "counters": asyncio.Lock(),
        }

        counters = {
            "processed": 0,
            "mentions": 0,
        }

        throttle = HostThrottle(self.rate_limit)
        sem = asyncio.Semaphore(max(1, self.concurrency))

        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def append_locked(path: str, obj: Dict[str, Any], key: str) -> None:
                async with locks[key]:
                    append_jsonl(path, obj)

            async def process_work(work: Dict[str, Any]) -> None:
                arxiv_id = work.get("arxiv_id")
                if not arxiv_id:
                    return

                await append_locked(arxiv_works_path, work, "arxiv")

                base = {
                    "openalex_id": work.get("openalex_id"),
                    "arxiv_id": arxiv_id,
                    "title": work.get("title"),
                }

                mentions: List[Dict[str, Any]] = []
                ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
                try:
                    logger.debug("Fetching ar5iv: {}", ar5iv_url)
                    html_path = await fetch_to_cache(
                        ar5iv_url,
                        self.cache_dir,
                        ".html",
                        session,
                        throttle,
                        offline=self.offline,
                    )
                    mentions = await asyncio.to_thread(
                        self.extractor.extract_from_html,
                        html_path,
                        ar5iv_url,
                        base,
                    )
                    logger.debug("ar5iv mentions: {}", len(mentions))
                except Exception as e:
                    await append_locked(
                        failures_path,
                        {
                            **base,
                            "stage": "ar5iv",
                            "error": str(e),
                            "source_url": ar5iv_url,
                        },
                        "failures",
                    )
                    logger.warning("ar5iv failed for {}: {}", arxiv_id, e)

                if not mentions and not self.no_pdf:
                    pdf_url = choose_pdf_url(work.get("source_urls") or [])
                    if pdf_url:
                        try:
                            logger.debug("Fetching PDF: {}", pdf_url)
                            pdf_path = await fetch_to_cache(
                                pdf_url,
                                self.cache_dir,
                                ".pdf",
                                session,
                                throttle,
                                offline=self.offline,
                            )
                            mentions = await asyncio.to_thread(
                                self.extractor.extract_from_pdf,
                                pdf_path,
                                pdf_url,
                                base,
                            )
                            logger.debug("PDF mentions: {}", len(mentions))
                        except Exception as e:
                            await append_locked(
                                failures_path,
                                {
                                    **base,
                                    "stage": "pdf",
                                    "error": str(e),
                                    "source_url": pdf_url,
                                },
                                "failures",
                            )
                            logger.warning("PDF failed for {}: {}", arxiv_id, e)

                if not mentions:
                    await append_locked(
                        failures_path,
                        {
                            **base,
                            "stage": "mention_search",
                            "error": "no_matches_found",
                        },
                        "failures",
                    )
                    logger.info("No mentions found for {}", arxiv_id)
                else:
                    for m in mentions:
                        await append_locked(mentions_path, m, "mentions")
                    logger.info("Found {} mentions for {}", len(mentions), arxiv_id)

                async with locks["counters"]:
                    counters["processed"] += 1
                    counters["mentions"] += len(mentions)
                    processed = counters["processed"]
                    if processed == 1 or processed % 10 == 0:
                        logger.info("Processed {} / {}", processed, len(arxiv_works))

            work_queue: asyncio.Queue = asyncio.Queue()
            for work in arxiv_works:
                work_queue.put_nowait(work)

            async def worker() -> None:
                while True:
                    try:
                        work = work_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    async with sem:
                        await process_work(work)
                    work_queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
            await work_queue.join()
            await asyncio.gather(*workers)

        stage_counts: Dict[str, int] = {}
        if os.path.exists(failures_path):
            for row in read_jsonl(failures_path):
                stage = row.get("stage") or "unknown"
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        logger.info(
            "Scanned {} works; processed {} arXiv works.",
            total,
            counters["processed"],
        )
        logger.info("Wrote {} mentions to {}", counters["mentions"], mentions_path)
        logger.info("ArXiv works list: {}", arxiv_works_path)
        logger.info("Failures: {}", failures_path)
        if stage_counts:
            logger.info("Failure breakdown: {}", stage_counts)
        return 0


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract target mention contexts for arXiv works."
    )
    parser.add_argument("--works-file", default=None, help="Input works JSONL.")
    parser.add_argument(
        "--target-id", default=None, help="Output file prefix (target id)."
    )
    parser.add_argument(
        "--target-arxiv",
        "--target-arxiv-url",
        dest="target_arxiv",
        default=None,
        help="Target arXiv id or URL (to derive title/id).",
    )
    parser.add_argument(
        "--target-title",
        default=None,
        help="Target work title for bibliography matching.",
    )
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument(
        "--cache-dir", default="data/cache", help="Cache directory for HTML/PDF."
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.8,
        help="Minimum seconds between requests per host.",
    )
    parser.add_argument(
        "--max-works",
        type=int,
        default=0,
        help="Limit number of arXiv works processed (0 = no limit).",
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="Skip PDF fallback; only use ar5iv HTML."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)."
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Max concurrent downloads/parses."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use cached HTML/PDF; fail on cache misses.",
    )
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    resolver = OpenAlexTargetResolver(cache_dir=args.cache_dir)

    target_title = args.target_title
    target_id = args.target_id
    if not target_id and args.target_arxiv:
        target_id = resolver.derive_target_id(parse_arxiv_id(args.target_arxiv))
    if not target_id:
        target_id = "target"

    if not target_title and args.target_arxiv:
        if args.offline:
            raise SystemExit(
                "Offline mode requires --target-title when using --target-arxiv."
            )
        meta = resolver.fetch_arxiv_metadata(
            parse_arxiv_id(args.target_arxiv), ArxivAPI()
        )
        target_title = meta.title

    if not args.works_file:
        args.works_file = os.path.join(args.out_dir, f"{target_id}_works.jsonl")

    if not target_title:
        raise SystemExit("No target title provided. Pass --target-title.")

    stage = MentionExtractionStage(
        works_file=args.works_file,
        target_title=target_title,
        target_id=target_id,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        rate_limit=args.rate_limit,
        max_works=args.max_works,
        no_pdf=args.no_pdf,
        concurrency=args.concurrency,
        offline=args.offline,
    )
    return await stage.run()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
