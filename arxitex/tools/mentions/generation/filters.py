"""Post-generation query filters for leakage and type conflicts."""

from __future__ import annotations

import re
from typing import Optional

# Reject queries that ask for location rather than statement content.
LOCATION_PATTERNS = [
    r"\bsection\b",
    r"\bsubsection\b",
    r"\bchapter\b",
    r"\bpage\b",
    r"\bappendix\b",
]

# Detect explicit definition requests (used to avoid type mismatch).
DEFINITION_PATTERNS = [
    r"\bdefinition\s+of\b",
    r"\bdefine\b",
    r"\bdef\.?\b",
]

# Detect explicit theorem-like requests (used to avoid type mismatch).
NON_DEFINITION_PATTERNS = [
    r"\btheorem\b",
    r"\bthm\.?\b",
    r"\blemma\b",
    r"\blem\.?\b",
    r"\bproposition\b",
    r"\bprop\.?\b",
    r"\bcorollary\b",
    r"\bcor\.?\b",
]

# Reject queries that leak labels or bibliographic clues.
LEAK_PATTERNS = [
    # Theorem/Lemma/etc + number/letter
    r"\b(?:Theorem|Thm\.?|Lemma|Lem\.?|Proposition|Prop\.?|Corollary|Cor\.?|"
    r"Definition|Def\.?|Example|Ex\.?|Remark|Rem\.?)\s*"
    r"(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s*(?:\([ivxIVX]+\))?\b",
    # Section numbers
    r"\b(?:Section|Sec\.?)\s*\d+(?:\.\d+)*\b",
    r"§\s*\d+(?:\.\d+)*",
    # Bracketed citations like [Sch12]
    r"\[[A-Za-z]{2,}\d{2,}[^\]]*\]",
    # Roman numeral condition references like (i), (ii), (iii)
    r"\(\s*[ivxIVX]+\s*\)",
    # Conjecture numbers
    r"\bConjecture\s*\d+(?:\.\d+)*\b",
    # Abbreviated citations like Sch12, Sch 2012, Scholze 2012
    r"\bSch(?:olze)?\s*\d{2,4}\b",
]


_LEAK_REGEXES = [re.compile(pat) for pat in LEAK_PATTERNS]
_LOCATION_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in LOCATION_PATTERNS]
_DEFINITION_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in DEFINITION_PATTERNS]
_NON_DEFINITION_REGEXES = [
    re.compile(pat, re.IGNORECASE) for pat in NON_DEFINITION_PATTERNS
]


def is_leaky(text: str, target_name: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if target_name and target_name.lower() in lower:
        return True
    return any(rx.search(text) for rx in _LEAK_REGEXES)


def has_location_terms(text: str) -> bool:
    if not text:
        return True
    return any(rx.search(text) for rx in _LOCATION_REGEXES)


def explicit_definition_request(text: str) -> bool:
    return any(rx.search(text) for rx in _DEFINITION_REGEXES)


def explicit_statement_request(text: str) -> bool:
    return any(rx.search(text) for rx in _NON_DEFINITION_REGEXES)


def type_conflict(text: str, explicit_kind: Optional[str]) -> bool:
    if not text or not explicit_kind:
        return False
    kind = explicit_kind.strip().lower()
    if kind == "definition":
        return explicit_statement_request(text)
    if kind in {"theorem", "lemma", "proposition", "corollary"}:
        return explicit_definition_request(text)
    return False
