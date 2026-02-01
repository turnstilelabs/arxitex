r"""Sanitize sgagraph dataset LaTeX so MathJax can render it without custom macros.

Option 2 implementation: rewrite node text fields so they no longer rely on
SGA-specific macro families (\cA, \sF, \spec, \fa, etc.).

Strategy:
  1) Strip known-problematic constructs (\label{...}, \xymatrix{...}).
  2) Expand macros using the dataset's own `latex_macros` map:
     - 0-arg macros: replace `\foo` -> body
     - selected 1-arg macros: replace `\foo{...}` via brace parsing

This is intentionally best-effort: it prefers robustness over perfect TeX
fidelity.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

MacroValue = str | list[Any]


def strip_labels(s: str) -> str:
    return re.sub(r"\\label\{[^}]*\}", "", s)


def strip_xymatrix_blocks(s: str) -> str:
    # Best-effort non-nested match; mirrors frontend cleaning.
    s = re.sub(r"\\xymatrix\s*\{[\s\S]*?\}", "[diagram]", s)
    # If the source contains a broken/unbalanced xymatrix block (common in the
    # HTML-ish previews with <br> tags), fall back to nuking the command name so
    # MathJax doesn't fail the whole expression.
    return s.replace(r"\xymatrix", "[diagram]")


def _parse_brace_group(s: str, start: int) -> tuple[str, int] | None:
    """Parse a balanced `{...}` group starting at `start` (which must be '{')."""

    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start + 1
    content_start = i
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return s[content_start:i], i + 1
            depth -= 1
        i += 1
    return None


def expand_one_arg_macro(s: str, name: str, body: str) -> str:
    """Replace occurrences of `\name{...}` by substituting #1 in body."""

    out: list[str] = []
    i = 0
    needle = "\\" + name
    while i < len(s):
        j = s.find(needle, i)
        if j == -1:
            out.append(s[i:])
            break

        out.append(s[i:j])
        k = j + len(needle)

        # Allow whitespace before brace
        while k < len(s) and s[k].isspace():
            k += 1

        if k >= len(s) or s[k] != "{":
            # Not a braced-arg invocation; keep literal.
            out.append(needle)
            i = j + len(needle)
            continue

        parsed = _parse_brace_group(s, k)
        if not parsed:
            out.append(needle)
            i = j + len(needle)
            continue

        arg, next_i = parsed
        repl = body.replace("#1", arg)
        out.append(repl)
        i = next_i

    return "".join(out)


def expand_zero_arg_macros(s: str, zero_arg: Dict[str, str]) -> str:
    # Sort longer names first to avoid partial overlap.
    names = sorted(zero_arg.keys(), key=len, reverse=True)
    for name in names:
        body = zero_arg[name]
        # Replace occurrences of \name not followed by a letter (word boundary-ish).
        # Use a function replacement so backslashes in the replacement body are
        # not interpreted as regex escapes.
        s = re.sub(
            r"\\" + re.escape(name) + r"(?![A-Za-z])",
            lambda _m, _body=body: _body,
            s,
        )
    return s


def sanitize_text(s: str, macros: Dict[str, MacroValue], max_passes: int = 5) -> str:
    if not s:
        return s

    s = strip_labels(s)
    s = strip_xymatrix_blocks(s)

    zero_arg: Dict[str, str] = {}
    one_arg: Dict[str, str] = {}

    for k, v in macros.items():
        if isinstance(v, str):
            zero_arg[k] = v
        elif (
            isinstance(v, list)
            and len(v) == 2
            and isinstance(v[0], str)
            and int(v[1]) == 1
        ):
            one_arg[k] = v[0]

    # Avoid expanding core TeX primitives that might be present as macros.
    # (We only generated SGA macros, but be defensive.)
    for reserved in ["begin", "end", "label", "ref", "text"]:
        zero_arg.pop(reserved, None)
        one_arg.pop(reserved, None)

    # Iterative expansion to allow macro bodies that contain other macros.
    for _ in range(max_passes):
        before = s

        # Expand selected 1-arg macros via brace parsing.
        # (Doing all 1-arg macros via parsing is safer than regex.)
        for name, body in one_arg.items():
            s = expand_one_arg_macro(s, name, body)

        s = expand_zero_arg_macros(s, zero_arg)

        if s == before:
            break

    # Some SGA macros like \an and \et are frequently used without braces
    # (e.g. "\an f_!" meaning (f_!)^{an}). Expand these best-effort.
    #
    # We only do this for a small known set and only when followed by a space
    # or a TeX non-letter token, to avoid accidentally rewriting other commands.
    an = one_arg.get("an")
    if an:
        # \an <token>  -> replace #1 with <token>
        s = re.sub(
            r"\\an\s+([^\s$}]+)",
            lambda m: an.replace("#1", m.group(1)),
            s,
        )

        # Handle \an\foo (no space, control sequence argument)
        s = re.sub(
            r"\\an\\([A-Za-z]+)",
            lambda m: an.replace("#1", "\\" + m.group(1)),
            s,
        )

        # Handle double-escaped form that can appear in HTML-ish previews
        # (e.g. "\\an\\mathcal{F}")
        s = re.sub(
            r"\\\\an\\\\([A-Za-z]+)",
            lambda m: an.replace("#1", "\\" + m.group(1)),
            s,
        )

        # Also handle \an(<arg>) used like a function call.
        s = re.sub(
            r"\\an\(([^)]*)\)",
            lambda m: an.replace("#1", m.group(1)),
            s,
        )

    et = one_arg.get("et")
    if et:
        s = re.sub(
            r"\\et\s+([^\s$}]+)",
            lambda m: et.replace("#1", m.group(1)),
            s,
        )

    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("../sgagraph/public/data/sga4-5.json"),
        help="Input sgagraph dataset JSON",
    )
    ap.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=None,
        help="Output path (default: overwrite input)",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak backup when overwriting",
    )
    args = ap.parse_args()

    data = json.loads(args.inp.read_text(encoding="utf-8"))
    macros = data.get("latex_macros") or {}
    if not isinstance(macros, dict):
        raise SystemExit("Expected top-level latex_macros dict")

    graph = data.get("graph") or {}
    nodes = graph.get("nodes") or []
    if not isinstance(nodes, list):
        raise SystemExit("Expected graph.nodes list")

    target_fields = [
        "content",
        "content_preview",
        "prerequisites_preview",
        "proof",
        "label",
    ]

    changed = 0
    for n in nodes:
        if not isinstance(n, dict):
            continue
        for f in target_fields:
            v = n.get(f)
            if not isinstance(v, str) or not v:
                continue
            nv = sanitize_text(v, macros)
            if nv != v:
                n[f] = nv
                changed += 1

    # Optionally remove latex_macros entirely to guarantee frontend does not
    # depend on runtime injection.
    # data.pop('latex_macros', None)

    out_path = args.out or args.inp
    if args.out is None and args.backup:
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        bak.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Sanitized {changed} field-values across {len(nodes)} nodes -> {out_path}")


if __name__ == "__main__":
    main()
