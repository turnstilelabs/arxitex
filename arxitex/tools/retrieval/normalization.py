"""Normalization helpers using math_verify."""

from __future__ import annotations

from typing import Optional

from loguru import logger

try:
    from math_verify import parse as mv_parse
except Exception:  # pragma: no cover - optional dependency
    mv_parse = None


UNICODE_MATH_MAP = {
    "∑": "\\sum",
    "∏": "\\prod",
    "∐": "\\coprod",
    "∫": "\\int",
    "√": "\\sqrt",
    "∞": "\\infty",
    "≤": "\\le",
    "≥": "\\ge",
    "≠": "\\neq",
    "≈": "\\approx",
    "≅": "\\cong",
    "→": "\\to",
    "↦": "\\mapsto",
    "↔": "\\leftrightarrow",
    "⊂": "\\subset",
    "⊆": "\\subseteq",
    "⊃": "\\supset",
    "⊇": "\\supseteq",
    "∈": "\\in",
    "∉": "\\notin",
    "∅": "\\emptyset",
    "∀": "\\forall",
    "∃": "\\exists",
    "∧": "\\wedge",
    "∨": "\\vee",
    "⊗": "\\otimes",
    "⊕": "\\oplus",
    "⊥": "\\bot",
    "⊤": "\\top",
    "ℤ": "\\mathbb{Z}",
    "ℚ": "\\mathbb{Q}",
    "ℝ": "\\mathbb{R}",
    "ℂ": "\\mathbb{C}",
    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "ε": "\\epsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ι": "\\iota",
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",
    "ξ": "\\xi",
    "π": "\\pi",
    "ρ": "\\rho",
    "σ": "\\sigma",
    "τ": "\\tau",
    "υ": "\\upsilon",
    "φ": "\\phi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",
    "Γ": "\\Gamma",
    "Δ": "\\Delta",
    "Θ": "\\Theta",
    "Λ": "\\Lambda",
    "Ξ": "\\Xi",
    "Π": "\\Pi",
    "Σ": "\\Sigma",
    "Υ": "\\Upsilon",
    "Φ": "\\Phi",
    "Ψ": "\\Psi",
    "Ω": "\\Omega",
}


def _normalize_unicode_math(text: str) -> str:
    return "".join(UNICODE_MATH_MAP.get(ch, ch) for ch in text)


def normalize_text(
    text: Optional[str], *, use_math_verify: bool = True
) -> Optional[str]:
    if not text:
        return text
    normalized = _normalize_unicode_math(text)
    if not use_math_verify or mv_parse is None:
        return normalized
    try:
        parsed = mv_parse(normalized)
    except Exception as exc:
        logger.debug("math_verify.parse failed: {}", exc)
        return normalized

    if isinstance(parsed, str):
        return parsed

    for attr in (
        "normalized",
        "normalized_latex",
        "latex",
        "normalized_text",
        "text",
    ):
        value = getattr(parsed, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    if hasattr(parsed, "to_string"):
        try:
            return parsed.to_string()
        except Exception:  # pragma: no cover - defensive
            pass

    return str(parsed)
