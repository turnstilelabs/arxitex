"""Shared helpers for citation_dataset stages."""

from __future__ import annotations

import re
from typing import Dict, List

# Canonicalize statement kind aliases used in citations.
TYPE_ALIASES = {
    "theorem": ["theorem", "th", "thm"],
    "proposition": ["proposition", "prop", "prop."],
    "lemma": ["lemma", "lem", "lem."],
    "corollary": ["corollary", "cor", "cor."],
    "definition": ["definition", "def", "def."],
    "example": ["example", "ex", "ex."],
    "remark": ["remark", "rem", "rem."],
}

# Match explicit refs like "Theorem 2.1", "Prop. 3", "Theorem A".
TYPE_PATTERN = re.compile(
    r"\b(?P<kind>theorem|th\.?|thm\.?|proposition|prop\.?|lemma|lem\.?|corollary|cor\.?|definition|def\.?|example|ex\.?|remark|rem\.?)"
    r"(?=\s|\(|\d|[A-Z])"  # avoid matching inside words; allow letter-only numbers
    r"\s*\(?"  # optional space/paren before number
    r"(?P<num>(?:[IVXLCDM]+\s*\.)?\s*\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\)?",
    re.IGNORECASE,
)

# Match named theorems like "Theorem Hahn-Banach" (capitalized tokens only).
NAMED_PATTERN = re.compile(
    r"(?P<kind>Theorem|Th\.?|Thm\.?|Definition|Def\.?|Proposition|Prop\.?|Lemma|Lem\.?|Corollary|Cor\.?)\s+"
    r"(?P<name>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,5})"
)


def normalize_kind(kind: str) -> str:
    k = kind.lower().strip(".")
    for canonical, aliases in TYPE_ALIASES.items():
        if k == canonical:
            return canonical
        if k in [a.strip(".") for a in aliases]:
            return canonical
    return k


def extract_refs(text: str) -> List[Dict]:
    refs: List[Dict] = []
    for match in TYPE_PATTERN.finditer(text or ""):
        raw_kind = match.group("kind")
        kind = normalize_kind(raw_kind)
        num = match.group("num").replace(" ", "")
        refs.append({"kind": kind, "number": num, "raw": match.group(0)})
    return refs


def extract_named(text: str) -> List[Dict]:
    named: List[Dict] = []
    for match in NAMED_PATTERN.finditer(text or ""):
        raw_kind = match.group("kind")
        kind = normalize_kind(raw_kind)
        name = match.group("name").strip()
        name = re.split(r"[,;\)\.]", name)[0].strip()
        if not name:
            continue
        if len(name.split()) > 6:
            continue
        named.append({"kind": kind, "name": name, "raw": match.group(0)})
    return named
