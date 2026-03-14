"""Utilities for extracting math symbol definitions from text."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from bs4 import BeautifulSoup

from arxitex.tools.mentions.extraction.mention_utils import split_sentences

DEF_CUE_RE = re.compile(
    r"\b(let|define|denote|we call|we say|we write)\b",
    re.IGNORECASE,
)
MATH_INLINE_RE = re.compile(r"\$(.+?)\$")
MATH_PAREN_RE = re.compile(r"\\\((.+?)\\\)")
MATH_BRACK_RE = re.compile(r"\\\[(.+?)\\\]")
SYMBOL_RE = re.compile(r"\\[A-Za-z]+(?:\{[^}]+\})?")
WRAPPER_RE = re.compile(r"\\(bar|tilde|hat|overline|underline)\{(.+)\}")


def _extract_math_segments(text: str) -> List[str]:
    segments: List[str] = []
    for regex in (MATH_INLINE_RE, MATH_PAREN_RE, MATH_BRACK_RE):
        segments.extend(regex.findall(text or ""))
    return segments


def _normalize_symbol(symbol: str) -> str:
    if not symbol:
        return symbol
    m = WRAPPER_RE.fullmatch(symbol)
    if m:
        return m.group(2)
    return symbol


def _extract_symbols_from_text(text: str) -> List[str]:
    symbols: List[str] = []
    for seg in _extract_math_segments(text):
        for sym in SYMBOL_RE.findall(seg):
            norm = _normalize_symbol(sym)
            if norm:
                symbols.append(norm)
    return symbols


def _is_definition_sentence(sentence: str) -> bool:
    return bool(sentence and DEF_CUE_RE.search(sentence))


def _collect_paragraphs(soup: BeautifulSoup) -> List[BeautifulSoup]:
    paragraphs: List[BeautifulSoup] = []
    for tag in soup.select(".ltx_para, p, li"):
        if tag.find_parent(class_="ltx_bibliography") is not None:
            continue
        paragraphs.append(tag)
    return paragraphs


def _build_definition_index(paragraphs: List[str]) -> Dict[str, List[tuple[int, str]]]:
    index: Dict[str, List[tuple[int, str]]] = defaultdict(list)
    for i, para_text in enumerate(paragraphs):
        for sentence in split_sentences(para_text):
            if not _is_definition_sentence(sentence):
                continue
            symbols = _extract_symbols_from_text(sentence)
            if not symbols:
                continue
            for sym in symbols:
                index[sym].append((i, sentence))
    return index


def _definitions_for_symbols(
    index: Dict[str, List[tuple[int, str]]],
    symbols: List[str],
    before_idx: int,
) -> List[str]:
    if not symbols:
        return []
    collected: List[tuple[int, str]] = []
    for sym in symbols:
        for idx, sent in index.get(sym, []):
            if idx < before_idx:
                collected.append((idx, sent))
    if not collected:
        return []
    collected.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    out: List[str] = []
    for _, sent in collected:
        if sent in seen:
            continue
        seen.add(sent)
        out.append(sent)
    return out
