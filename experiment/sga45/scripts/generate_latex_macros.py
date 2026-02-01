r"""Generate a MathJax v3-compatible macro map for SGA 4 1/2.

This script extracts macros from the LaTeX style file used by the SGA 4.5
sources and writes a JSON file that can be injected into sgagraph's graph JSON
under the `latex_macros` key.

Why a custom generator?

The SGA sources define a large family of macros via TeX loops / \\csname tricks
(e.g. \\cA..\\cZ, \\dA..\\dZ, \\fA..\\fZ, \\eA..\\eZ, \\sA..\\sZ and
\\fa..\\fz (skipping i)), which are hard to recover via naive regex scanning.

MathJax expects `tex.macros` values to be either:
  - string replacement bodies
  - [body, nArgs]

We emit a map of macro names *without* the leading backslash.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Dict, Tuple, Union

MacroValue = Union[str, Tuple[str, int]]


def _parse_brace_group(text: str, start: int) -> tuple[str, int]:
    """Parse a `{...}` group starting at/after `start`.

    Returns (content_without_outer_braces, next_index).
    """

    i = start
    while i < len(text) and text[i].isspace():
        i += 1

    if i >= len(text) or text[i] != "{":
        raise ValueError(f"Expected '{{' at index {i}")

    depth = 0
    content_start = i + 1
    i += 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[content_start:i], i + 1
            depth -= 1
        i += 1

    raise ValueError("Unterminated brace group")


def _iter_declare_math_operator(style_text: str):
    r"""Yield (name_without_backslash, body) for each \DeclareMathOperator call."""

    text = style_text or ""
    needle = "\\DeclareMathOperator"
    i = 0
    while True:
        idx = text.find(needle, i)
        if idx == -1:
            return

        j = idx + len(needle)
        if j < len(text) and text[j] == "*":
            j += 1

        try:
            raw_name, j = _parse_brace_group(text, j)
            raw_body, j = _parse_brace_group(text, j)
        except Exception:
            i = idx + len(needle)
            continue

        name = raw_name.strip()
        if name.startswith("\\"):
            name = name[1:]
        body = raw_body.strip()

        if name and body:
            yield name, body

        i = j


def _add(d: Dict[str, MacroValue], name: str, body: str, n_args: int = 0) -> None:
    name = name.strip()
    body = body.strip()
    # IMPORTANT:
    # The JSON emitted by this script is consumed directly as MathJax v3
    # `tex.macros` configuration. Those macro bodies must use *single* TeX
    # backslashes (e.g. "\\mathcal{A}").
    #
    # Earlier versions of this generator over-escaped backslashes because the
    # strings were written as Python raw strings (r"\\mathcal{A}") and then
    # dumped to JSON. That caused MathJax to see expansions like "\\\\mathcal"
    # which break parsing.
    #
    # We normalize here so the output macro body always contains single
    # backslashes.
    body = body.replace("\\\\", "\\")
    if not name or not body:
        return
    if n_args <= 0:
        d[name] = body
    else:
        d[name] = (body, int(n_args))


def extract_from_style(style_text: str) -> Dict[str, MacroValue]:
    """Extract explicit macros that MathJax can benefit from.

    We intentionally focus on constructs that appear in `sga-style.sty`.
    """

    macros: Dict[str, MacroValue] = {}

    # Common operators are declared via \DeclareMathOperator{\foo}{Foo}.
    # NOTE: Some SGA operators use \mathscr + calligra hacks that MathJax
    # doesn't understand. In those cases we fall back to a simpler operator
    # label.
    fallback_operator_text = {
        "Ext": "Ext",
        "Hom": "Hom",
        "Tor": "Tor",
        "rHom": "RHom",
        "rhom": "Rhom",
        "hh": "H",
    }

    for name, body in _iter_declare_math_operator(style_text):
        op_text = body
        if name in fallback_operator_text:
            op_text = fallback_operator_text[name]
        elif re.search(r"\\calligra|\\kern|\\text", body):
            # Best-effort fallback when the body is likely to break MathJax.
            op_text = name
        _add(macros, name, rf"\\operatorname{{{op_text}}}")

    # Single explicit newcommands in the style:
    #   \newcommand{\dmu}{{\bm\mu}}
    #   \newcommand{\an}[1]{{#1}^{\textnormal{an}}}
    #   \newcommand{\const}[1]{\underline{#1}}
    #   \newcommand{\et}[1]{{#1}_{\textnormal{et}}}
    #   \newcommand{\iso}{\xrightarrow\sim}
    #   \newcommand{\lotimes}{{\overset{\mathsf{L}}{\otimes}}}
    #   \newcommand{\pr}{\mathrm{pr}}
    # and a couple of renewcommands.
    #
    # Parsing arbitrary \newcommand is non-trivial; for this known style we
    # just hardcode the macros we know are used heavily throughout the text.
    _add(macros, "dmu", r"{\\bm\\mu}")
    # MathJax doesn't always support \textnormal, so use \mathrm.
    _add(macros, "an", r"{#1}^{\\mathrm{an}}", n_args=1)
    _add(macros, "const", r"\\underline{#1}", n_args=1)
    _add(macros, "et", r"{#1}_{\\mathrm{et}}", n_args=1)
    _add(macros, "iso", r"\\xrightarrow\\sim")
    _add(macros, "lotimes", r"{\\overset{\\mathsf{L}}{\\otimes}}")
    _add(macros, "pr", r"\\mathrm{pr}")

    # ---- MathJax compatibility shims --------------------------------------
    # Many LaTeX sources use these commands, but MathJax may not have them
    # enabled by default depending on the input configuration.
    #
    # - \mathscr is typically provided by `mathrsfs` in LaTeX; for MathJax we
    #   map it to \mathcal as a reasonable visual fallback.
    _add(macros, "mathscr", r"\\mathcal{#1}", n_args=1)
    # - \bm is provided by `bm` in LaTeX; map to \boldsymbol.
    _add(macros, "bm", r"\\boldsymbol{#1}", n_args=1)

    # \renewcommand{\setminus}{\smallsetminus}
    # MathJax already supports \setminus, but this helps match the book.
    _add(macros, "setminus", r"\\smallsetminus")

    # \renewcommand{\bullet}{\textnormal{\tiny$\oldbullet$}}
    # This is cosmetic and can confuse MathJax; we omit it.

    return macros


def synthesize_loop_macros() -> Dict[str, MacroValue]:
    """Re-create the macros defined by TeX loops and \\csname in the style."""

    macros: Dict[str, MacroValue] = {}

    # \cA..\cZ  -> \mathcal{A}..\mathcal{Z}
    # \dA..\dZ  -> \mathbb{A}..\mathbb{Z}
    # \fA..\fZ  -> \mathfrak{A}..\mathfrak{Z}
    # \eA..\eZ  -> \mathsf{A}..\mathsf{Z}
    # \sA..\sZ  -> \mathscr{A}..\mathscr{Z}
    # In the LaTeX sources this uses `mathrsfs` (\mathscr). MathJax's TeX input
    # doesn't always support \mathscr by default, so we map \sA..\sZ to
    # \mathcal{A..Z} for a reliable fallback.
    for ch in string.ascii_uppercase:
        _add(macros, f"c{ch}", rf"\\mathcal{{{ch}}}")
        _add(macros, f"d{ch}", rf"\\mathbb{{{ch}}}")
        _add(macros, f"f{ch}", rf"\\mathfrak{{{ch}}}")
        _add(macros, f"e{ch}", rf"\\mathsf{{{ch}}}")
        _add(macros, f"s{ch}", rf"\\mathcal{{{ch}}}")

    # Lower-case fraktur via \def\mydeff#1{\expandafter\def\csname f#1\endcsname{\mathfrak{#1}}}
    # and then \mydefallf abcdefghjklmnopqrstuvwxyz (note: skips 'i')
    for ch in "abcdefghjklmnopqrstuvwxyz":
        _add(macros, f"f{ch}", rf"\\mathfrak{{{ch}}}")

    return macros


def generate_macros(style_path: Path) -> Dict[str, MacroValue]:
    style_text = style_path.read_text(encoding="utf-8")
    macros: Dict[str, MacroValue] = {}
    macros.update(extract_from_style(style_text))
    macros.update(synthesize_loop_macros())
    return macros


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        type=Path,
        default=Path("experiment/sga45/src/sga4.5/sga-style.sty"),
        help="Path to sga-style.sty",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment/sga45/output/sga4-5_latex_macros.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    macros = generate_macros(args.style)

    # Convert tuple values to the JSON list form expected by sgagraph.
    out: Dict[str, object] = {}
    for k, v in sorted(macros.items(), key=lambda kv: kv[0]):
        if isinstance(v, tuple):
            out[k] = [v[0], v[1]]
        else:
            out[k] = v

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(out)} macros to {args.output}")


if __name__ == "__main__":
    main()
