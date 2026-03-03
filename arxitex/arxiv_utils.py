"""Shared helpers for arXiv identifiers and URLs."""

from __future__ import annotations

import re
from typing import Iterable, Optional

ARXIV_ID_RE = re.compile(
    r"(?P<id>(\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)


def parse_arxiv_id(value: str, *, preserve_version: bool = False) -> str:
    """Extract an arXiv id from a URL or id string."""

    raw = (value or "").strip()
    if not raw:
        raise ValueError("Empty arXiv input")
    match = ARXIV_ID_RE.search(raw)
    if not match:
        raise ValueError(f"Unrecognized arXiv id in '{value}'")
    arxiv_id = match.group("id")
    if preserve_version:
        return arxiv_id
    return normalize_arxiv_id(arxiv_id)


def try_parse_arxiv_id(value: str) -> Optional[str]:
    try:
        return parse_arxiv_id(value)
    except Exception:
        return None


def is_arxiv_url(url: str) -> bool:
    u = (url or "").lower()
    return (
        "arxiv.org" in u
        or "export.arxiv.org" in u
        or "ar5iv.org" in u
        or "ar5iv.labs" in u
    )


def normalize_arxiv_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", (raw or ""), flags=re.IGNORECASE).strip()


def extract_arxiv_id_from_urls(urls: Iterable[str]) -> Optional[str]:
    for u in urls or []:
        if not is_arxiv_url(u):
            continue
        parsed = try_parse_arxiv_id(u)
        if parsed:
            return normalize_arxiv_id(parsed)
    return None


def choose_pdf_url(urls: Iterable[str]) -> Optional[str]:
    for u in urls or []:
        ul = (u or "").lower()
        if not u:
            continue
        if is_arxiv_url(ul):
            continue
        if ".pdf" in ul:
            return u
    return None
