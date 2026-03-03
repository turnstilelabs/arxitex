"""Regex tokenizer that preserves LaTeX commands as single tokens."""

from __future__ import annotations

import re
from typing import Iterable, List

TOKEN_RE = re.compile(
    r"""
    (\\[A-Za-z]+)        # LaTeX command
    |(\\[^A-Za-z\s])      # single-char LaTeX command like \, \_, \^, \{
    |([A-Za-z0-9]+)        # words/numbers
    |([^\s])              # any other single non-space char
    """,
    re.VERBOSE,
)


def tokenize_latex(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.startswith("\\"):
            tokens.append(token)
        else:
            tokens.append(token.lower())
    return tokens


def tokenize_many(texts: Iterable[str]) -> List[List[str]]:
    return [tokenize_latex(t) for t in texts]
