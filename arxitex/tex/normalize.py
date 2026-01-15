from __future__ import annotations

import re
from dataclasses import dataclass

from arxitex.tex.dialect import TeXDialect


@dataclass(frozen=True)
class NormalizationResult:
    content: str
    changed: bool


_CANONICAL_TYPES = [
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "remark",
    "example",
    "claim",
    "observation",
    "fact",
    "conjecture",
]


def _infer_artifact_type_from_title(title: str) -> str:
    t = (title or "").strip().lower()
    for k in _CANONICAL_TYPES:
        if k in t:
            return k
    # Common shorthand
    if "prop" in t:
        return "proposition"
    if "cor" in t:
        return "corollary"
    if "def" in t:
        return "definition"
    if "thm" in t:
        return "theorem"
    if "lem" in t:
        return "lemma"
    return "unknown"


def normalize_tex(content: str, dialect: TeXDialect) -> NormalizationResult:
    """Normalize AMS/plain TeX constructs into LaTeX-like environments.

    This is intentionally *best-effort* and designed to support the existing
    `BaseGraphBuilder` environment parser.
    """

    if not content:
        return NormalizationResult(content="", changed=False)

    if dialect == TeXDialect.LATEX:
        return NormalizationResult(content=content, changed=False)

    if dialect not in (TeXDialect.AMS_TEX, TeXDialect.PLAIN_TEX, TeXDialect.UNKNOWN):
        return NormalizationResult(content=content, changed=False)

    out = content
    changed = False

    # --- Proof blocks: \demo ... \enddemo  -> \begin{proof} ... \end{proof}
    demo_pat = re.compile(r"\\demo\b(.*?)(?:\\enddemo\b)", re.DOTALL)

    def _demo_repl(m: re.Match) -> str:
        nonlocal changed
        changed = True
        body = (m.group(1) or "").strip()
        return "\\begin{proof}\n" + body + "\n\\end{proof}"

    out2 = demo_pat.sub(_demo_repl, out)
    out = out2

    # --- Proclaim blocks: \\proclaim{Title} ... \\endproclaim
    proclaim_braced = re.compile(
        r"\\proclaim\s*\{(?P<title>[^}]*)\}\s*(?P<body>.*?)(?:\\endproclaim\b)",
        re.DOTALL,
    )

    proof_env_pat = re.compile(
        r"\\begin\{proof\}(?P<body>.*?)\\end\{proof\}", re.DOTALL
    )

    def _proclaim_repl(m: re.Match) -> str:
        nonlocal changed
        changed = True
        title = (m.group("title") or "").strip()
        body = (m.group("body") or "").strip()
        env = _infer_artifact_type_from_title(title)
        opt = f"[{title}]" if title else ""

        # IMPORTANT: AMS-TeX proofs (\demo ... \enddemo) are often *inside* the
        # proclaim block. If we keep a nested \begin{proof} inside
        # \begin{theorem}, our current regex builder will NOT discover the proof
        # as a standalone environment (it does not recursively parse nested
        # environments). So we lift proof blocks out and append them after the
        # statement environment.
        lifted_proofs: list[str] = []

        def _lift_proof(pm: re.Match) -> str:
            lifted_proofs.append(
                "\\begin{proof}\n" + (pm.group("body") or "").strip() + "\n\\end{proof}"
            )
            return ""  # remove from statement body

        body_wo_proofs = proof_env_pat.sub(_lift_proof, body).strip()

        statement = f"\\begin{{{env}}}{opt}\n{body_wo_proofs}\n\\end{{{env}}}"
        if lifted_proofs:
            return statement + "\n" + "\n".join(lifted_proofs)
        return statement

    out2 = proclaim_braced.sub(_proclaim_repl, out)
    out = out2

    # Some AMS/plain sources omit \endproclaim; be conservative and stop at
    # the next \proclaim/\demo/\bye/end-of-file.
    proclaim_unterminated = re.compile(
        r"\\proclaim\s*\{(?P<title>[^}]*)\}\s*(?P<body>.*?)(?=(\\proclaim\b|\\demo\b|\\bye\b|\\end\{document\}|\Z))",
        re.DOTALL,
    )

    out2 = proclaim_unterminated.sub(_proclaim_repl, out)
    out = out2

    return NormalizationResult(content=out, changed=changed)
