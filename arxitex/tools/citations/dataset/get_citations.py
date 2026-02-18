#!/usr/bin/env python3
"""Stage 2: identify arXiv-available works and extract target mention contexts.

Inputs:
- data/{target}_works.jsonl (from sga45_pipeline.py)

Outputs:
- data/{target}_arxiv_works.jsonl
- data/{target}_mentions.jsonl
- data/{target}_failures.jsonl
"""

import argparse
import asyncio
import os
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from pdfminer.high_level import extract_text as pdf_extract_text

from arxitex.tools.citations.dataset.utils import (
    append_jsonl,
    ensure_dir,
    extract_refs,
    read_jsonl,
    sha256_hash,
)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", re.I)

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def title_matches_entry(
    entry_text: str, target_title: str, min_sim: float = 0.9
) -> bool:
    if not entry_text or not target_title:
        return False
    n_entry = _norm(entry_text)
    n_title = _norm(target_title)
    if not n_entry or not n_title:
        return False
    if n_title in n_entry:
        return True
    return title_similarity(entry_text, target_title) >= min_sim


def is_arxiv_url(url: str) -> bool:
    u = url.lower()
    return (
        "arxiv.org" in u
        or "export.arxiv.org" in u
        or "ar5iv.org" in u
        or "ar5iv.labs" in u
    )


def normalize_arxiv_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", raw, flags=re.I).strip()


def extract_arxiv_id_from_urls(urls: List[str]) -> Optional[str]:
    for u in urls:
        if not is_arxiv_url(u):
            continue
        m = ARXIV_ID_RE.search(u)
        if m:
            return normalize_arxiv_id(m.group(1))
    return None


def choose_pdf_url(urls: List[str]) -> Optional[str]:
    for u in urls:
        ul = u.lower()
        if is_arxiv_url(ul):
            continue
        if ".pdf" in ul:
            return u
    return None


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
                # Reserve the next slot to avoid stampede.
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
) -> str:
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, sha256_hash(url) + ext)
    if os.path.exists(cache_path):
        return cache_path

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


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return SENT_SPLIT_RE.split(text)


def normalize_for_match(text: str) -> str:
    return (
        text.replace("\u00bd", "1/2")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )


def build_label_regex(label: str) -> re.Pattern:
    normalized = normalize_for_match(label)
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        year = year_match.group(0)
        surname = normalized.split()[0]
        return re.compile(rf"\\b{re.escape(surname)}\\b\\W*{re.escape(year)}\\b")
    safe = re.escape(normalized)
    if re.fullmatch(r"\d+", normalized):
        return re.compile(rf"(?:\\[{safe}\\]|\\({safe}\\))")
    return re.compile(rf"(?:\\[{safe}\\]|\\({safe}\\)|\\b{safe}\\b)")


def extract_bib_label(text: str) -> str:
    m = re.match(r"^\s*\\[(.+?)\\]\s*", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\s*\\((.+?)\\)\s*", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\s*(\\d+)\\.\s*", text)
    if m:
        return m.group(1).strip()
    return ""


def derive_author_year_labels(text: str) -> List[str]:
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if not year_match:
        return []
    year = year_match.group(0)
    name_match = re.search(r"^\s*([A-Z][A-Za-z'`-]+)", text)
    if not name_match:
        name_match = re.search(r"\b([A-Z][A-Za-z'`-]+)\b", text)
    if not name_match:
        return []
    surname = name_match.group(1)
    return [f"{surname} {year}", f"{surname}, {year}"]


def derive_labels_from_entry(text: str) -> List[str]:
    labels: List[str] = []
    label = extract_bib_label(text)
    if label:
        labels.append(label)
    labels.extend(derive_author_year_labels(text))
    # De-duplicate while preserving order.
    seen = set()
    out: List[str] = []
    for label_text in labels:
        if label_text not in seen:
            seen.add(label_text)
            out.append(label_text)
    return out


def find_sentence_index(sentences: List[str], label_re: re.Pattern) -> int:
    for i, s in enumerate(sentences):
        if label_re.search(normalize_for_match(s)):
            return i
    return -1


def extract_mentions_from_paragraph(
    text: str,
    section_title: Optional[str],
    location_type: str,
    source: str,
    source_url: str,
    base: Dict[str, Any],
    labels: List[str],
    context_html: Optional[str] = None,
) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    sentences = split_sentences(text)
    if not labels:
        return mentions
    for label in labels:
        label_re = build_label_regex(label)
        idx = find_sentence_index(sentences, label_re)
        if not sentences or idx < 0:
            continue
        explicit_refs = extract_refs(sentences[idx])
        mentions.append(
            {
                **base,
                "match_text": label,
                "location_type": location_type,
                "section_title": section_title,
                "context_prev": sentences[idx - 1] if idx - 1 >= 0 else None,
                "context_sentence": sentences[idx],
                "context_next": (
                    sentences[idx + 1] if idx + 1 < len(sentences) else None
                ),
                "context_html": context_html,
                "source": source,
                "source_url": source_url,
                "explicit_refs": explicit_refs,
                "reference_precision": "explicit" if explicit_refs else "implicit",
            }
        )
    return mentions


def extract_mentions_from_html(
    html_path: str,
    source_url: str,
    base: Dict[str, Any],
    target_title: str,
) -> List[Dict[str, Any]]:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    mentions: List[Dict[str, Any]] = []
    # Build a map of bibliography entries that match the target title.
    bib_targets: Dict[str, Dict[str, str]] = {}
    for bib in soup.select(".ltx_bibliography .ltx_bibitem, .ltx_bibitem"):
        bib_id = bib.get("id")
        if not bib_id:
            continue
        text = bib.get_text(" ", strip=True)
        if not text:
            continue
        if title_matches_entry(text, target_title):
            tag = bib.select_one(".ltx_bibtag")
            label = tag.get_text(" ", strip=True) if tag else ""
            labels = [label] if label else []
            if not labels:
                labels = derive_labels_from_entry(text)
            bib_targets[bib_id] = {"labels": labels, "text": text}

    # If we have target bib entries, find in-text citations pointing to them.
    if bib_targets:
        seen: set = set()
        for a in soup.select("a.ltx_ref, a.ltx_cite"):
            href = a.get("href") or ""
            if not href.startswith("#"):
                continue
            bib_id = href[1:]
            if bib_id not in bib_targets:
                continue

            # Skip citations inside the bibliography itself.
            if a.find_parent(class_="ltx_bibliography") is not None:
                continue

            container = (
                a.find_parent(class_="ltx_para")
                or a.find_parent("p")
                or a.find_parent("li")
            )
            if container is None:
                continue

            section = None
            heading = container.find_previous(["h1", "h2", "h3", "h4", "h5"])
            if heading is not None:
                section = heading.get_text(" ", strip=True) or None

            # Build paragraph text with a marker at this anchor to locate the exact sentence.
            marker = "__CITE_MARKER__"
            container_copy = BeautifulSoup(str(container), "lxml")
            marker_anchor = container_copy.find("a", href=f"#{bib_id}")
            if marker_anchor is not None:
                marker_anchor.replace_with(marker)
            para_text = container_copy.get_text(" ", strip=True)
            if not para_text:
                continue
            context_html = str(container)

            sentences = split_sentences(para_text)
            labels = bib_targets[bib_id].get("labels") or []
            label = labels[0] if labels else a.get_text(" ", strip=True)
            idx = 0
            for i, s in enumerate(sentences):
                if marker in s:
                    idx = i
                    sentences[i] = s.replace(marker, label or "citation")
                    break
            else:
                if label:
                    label_re = build_label_regex(label)
                    for i, s in enumerate(sentences):
                        if label_re.search(normalize_for_match(s)):
                            idx = i
                            break

            key = (base.get("arxiv_id"), bib_id, sentences[idx] if sentences else "")
            if key in seen:
                continue
            seen.add(key)

            explicit_refs = extract_refs(sentences[idx] if sentences else para_text)
            mentions.append(
                {
                    **base,
                    "match_text": label or "citation",
                    "location_type": "body_citation",
                    "section_title": section,
                    "context_prev": sentences[idx - 1] if idx - 1 >= 0 else None,
                    "context_sentence": sentences[idx] if sentences else para_text,
                    "context_next": (
                        sentences[idx + 1] if idx + 1 < len(sentences) else None
                    ),
                    "context_html": context_html,
                    "source": "ar5iv",
                    "source_url": source_url,
                    "cite_target": bib_id,
                    "cite_label": label or None,
                    "bib_entry": bib_targets[bib_id].get("text"),
                    "explicit_refs": explicit_refs,
                    "reference_precision": "explicit" if explicit_refs else "implicit",
                }
            )

    return mentions


def extract_mentions_from_pdf(
    pdf_path: str,
    source_url: str,
    base: Dict[str, Any],
    target_title: str,
) -> List[Dict[str, Any]]:
    text = pdf_extract_text(pdf_path) or ""
    text = text.replace("\x0c", "\n")

    # Detect bibliography start roughly.
    lower = text.lower()
    bib_idx = None
    for term in ["references", "bibliography"]:
        idx = lower.find(term)
        if idx != -1 and (bib_idx is None or idx < bib_idx):
            bib_idx = idx

    body_text = text if bib_idx is None else text[:bib_idx]
    bib_text = "" if bib_idx is None else text[bib_idx:]

    labels: List[str] = []
    if bib_text and target_title:
        entry_start_re = re.compile(r"^\s*(?:\[[^\]]+\]|\([^\)]+\)|\d+\.)\s+")
        author_year_start_re = re.compile(
            r"^\s*[A-Z][A-Za-z'`-]+(?:,|\s)\s+.*\b(19|20)\d{2}\b"
        )
        entries: List[str] = []
        current: List[str] = []
        for line in bib_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if entry_start_re.match(line) or author_year_start_re.match(line):
                if current:
                    entries.append(" ".join(current))
                current = [line]
            else:
                if current:
                    current.append(line)
        if current:
            entries.append(" ".join(current))

        for entry in entries:
            if title_matches_entry(entry, target_title):
                labels.extend(derive_labels_from_entry(entry))

    mentions: List[Dict[str, Any]] = []
    for m in re.finditer(r"\S.*?(?:\n{2,}|\Z)", body_text, flags=re.S):
        para = m.group(0).strip()
        if not para:
            continue
        mentions.extend(
            extract_mentions_from_paragraph(
                text=para,
                section_title=None,
                location_type="body",
                source="pdf",
                source_url=source_url,
                base=base,
                labels=labels,
            )
        )

    return mentions


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract target mention contexts for arXiv works."
    )
    parser.add_argument("--works-file", default=None, help="Input works JSONL.")
    parser.add_argument(
        "--target-id", default="target", help="Output file prefix (target id)."
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
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    target_id = args.target_id
    target_title = args.target_title

    if not args.works_file:
        args.works_file = os.path.join(args.out_dir, f"{target_id}_works.jsonl")

    if not target_title:
        raise SystemExit("No target title provided. Pass --target-title.")

    arxiv_works_path = os.path.join(args.out_dir, f"{target_id}_arxiv_works.jsonl")
    mentions_path = os.path.join(args.out_dir, f"{target_id}_mentions.jsonl")
    failures_path = os.path.join(args.out_dir, f"{target_id}_failures.jsonl")

    # Overwrite outputs on each run for now.
    for p in [arxiv_works_path, mentions_path, failures_path]:
        if os.path.exists(p):
            os.remove(p)

    logger.info("Stage 2: scanning works from {}", args.works_file)
    logger.info(
        "Outputs: {}", ", ".join([arxiv_works_path, mentions_path, failures_path])
    )
    logger.info("Cache dir: {}", args.cache_dir)
    logger.info(
        "Concurrency: {} | Rate limit: {}s per host", args.concurrency, args.rate_limit
    )

    works = list(read_jsonl(args.works_file))
    total = len(works)

    # Prepare arXiv works list
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
                    "Indexed in arXiv but no arXiv ID/URL: {}", w.get("openalex_id")
                )
            continue
        arxiv_id = normalize_arxiv_id(arxiv_id)
        w["arxiv_id"] = arxiv_id
        arxiv_works.append(w)

    if args.max_works:
        arxiv_works = arxiv_works[: args.max_works]

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

    throttle = HostThrottle(args.rate_limit)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def append_locked(path: str, obj: Dict[str, Any], key: str) -> None:
            async with locks[key]:
                append_jsonl(path, obj)

        async def process_work(work: Dict[str, Any]) -> None:
            async with sem:
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
                        ar5iv_url, args.cache_dir, ".html", session, throttle
                    )
                    mentions = await asyncio.to_thread(
                        extract_mentions_from_html,
                        html_path,
                        ar5iv_url,
                        base,
                        target_title,
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

                if not mentions and not args.no_pdf:
                    pdf_url = choose_pdf_url(work.get("source_urls") or [])
                    if pdf_url:
                        try:
                            logger.debug("Fetching PDF: {}", pdf_url)
                            pdf_path = await fetch_to_cache(
                                pdf_url, args.cache_dir, ".pdf", session, throttle
                            )
                            mentions = await asyncio.to_thread(
                                extract_mentions_from_pdf,
                                pdf_path,
                                pdf_url,
                                base,
                                target_title,
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

        tasks = [asyncio.create_task(process_work(w)) for w in arxiv_works]
        await asyncio.gather(*tasks)

    # Summarize failures by stage for quick diagnosis.
    stage_counts: Dict[str, int] = {}
    if os.path.exists(failures_path):
        for row in read_jsonl(failures_path):
            stage = row.get("stage") or "unknown"
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    logger.info(
        "Scanned {} works; processed {} arXiv works.", total, counters["processed"]
    )
    logger.info("Wrote {} mentions to {}", counters["mentions"], mentions_path)
    logger.info("ArXiv works list: {}", arxiv_works_path)
    logger.info("Failures: {}", failures_path)
    if stage_counts:
        logger.info("Failure breakdown: {}", stage_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
