"""Utility helpers for mention extraction."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

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


def normalize_for_match(text: str) -> str:
    return (
        text.replace("\u00bd", "1/2")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return SENT_SPLIT_RE.split(text)


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
