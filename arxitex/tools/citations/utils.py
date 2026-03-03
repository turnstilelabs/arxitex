"""Shared helpers for citation_dataset stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, Iterable, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: str, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TYPE_ALIASES = {
    "theorem": ["theorem", "th", "thm"],
    "proposition": ["proposition", "prop", "prop."],
    "lemma": ["lemma", "lem", "lem."],
    "corollary": ["corollary", "cor", "cor."],
    "definition": ["definition", "def", "def."],
    "example": ["example", "ex", "ex."],
    "remark": ["remark", "rem", "rem."],
}

TYPE_PATTERN = re.compile(
    r"(?P<kind>theorem|th\.?|thm\.?|proposition|prop\.?|lemma|lem\.?|corollary|cor\.?|definition|def\.?|example|ex\.?|remark|rem\.?)\s*"
    r"(?P<num>(?:[IVXLCDM]+\s*\.)?\s*\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

NAMED_PATTERN = re.compile(
    r"(?P<kind>Theorem|Th\.?|Thm\.?|Definition|Def\.?|Proposition|Prop\.?|Lemma|Lem\.?|Corollary|Cor\.?)\s+"
    r"(?P<name>[A-Z][A-Za-z\-\s]{1,60})"
)

LOWER_NAME_WHITELIST = {
    "finitude",
    "finiteness",
    "sommes",
    "trig",
    "trigonometriques",
    "sommes trig",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "general",
    "criterion",
    "statement",
    "result",
    "construction",
    "discussion",
}


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
        first = name.split()[0].lower()
        if first in STOPWORDS:
            continue
        if len(name.split()) > 6:
            continue
        named.append({"kind": kind, "name": name, "raw": match.group(0)})

    # Lowercase names after "Th." etc (e.g., "Th. finitude")
    for match in TYPE_PATTERN.finditer(text or ""):
        raw_kind = match.group("kind")
        if raw_kind.lower().startswith("th"):
            tail = (text or "")[match.end() :]
            m = re.match(r"\s*([a-z][a-z\-]{3,20})", tail)
            if m:
                name = m.group(1).strip()
                if name in LOWER_NAME_WHITELIST:
                    named.append(
                        {"kind": "theorem", "name": name, "raw": f"{raw_kind} {name}"}
                    )
    return named
