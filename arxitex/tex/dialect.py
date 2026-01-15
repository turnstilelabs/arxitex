from __future__ import annotations

import enum
import re


class TeXDialect(str, enum.Enum):
    LATEX = "latex"
    AMS_TEX = "ams_tex"
    PLAIN_TEX = "plain_tex"
    UNKNOWN = "unknown"


_LATEX_MARKERS = (
    "\\documentclass",
    "\\begin{document}",
    "\\usepackage",
)

_AMS_MARKERS = (
    "\\proclaim",
    "\\endproclaim",
    "\\demo",
    "\\enddemo",
    "\\input amstex",
    "\\documentstyle{amsppt",
    "\\documentstyle{ams",
)

_PLAIN_MARKERS = (
    "\\bye",
    "\\magnification",
    "\\headline",
    "\\footline",
    "\\nopagenumbers",
)


def detect_tex_dialect(content: str) -> TeXDialect:
    """Best-effort dialect detection.

    The goal is not to be perfect, but to decide whether to run a lightweight
    normalization pass before feeding the content into the existing LaTeX
    environment-based extractor.
    """

    if not content:
        return TeXDialect.UNKNOWN

    # Fast lowercase scan.
    lower = content.lower()

    if any(m in lower for m in _LATEX_MARKERS):
        return TeXDialect.LATEX

    if any(m in lower for m in _AMS_MARKERS):
        return TeXDialect.AMS_TEX

    if any(m in lower for m in _PLAIN_MARKERS):
        return TeXDialect.PLAIN_TEX

    # Heuristic fallback: if it looks like TeX (many backslashes), but we
    # can't confidently say which dialect.
    if "\\" in content and re.search(r"\\[a-zA-Z@]+", content):
        return TeXDialect.UNKNOWN

    return TeXDialect.UNKNOWN
