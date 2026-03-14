"""MSC2020 CSV ingestion and symbolic context matching."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

FIVE_DIGIT_RE = re.compile(r"^\d{2}[A-Z]\d{2}$")
THREE_DIGIT_RE = re.compile(r"^\d{2}[A-Z]xx$")
TWO_DIGIT_RE = re.compile(r"^\d{2}-XX$")
TOKEN_RE = re.compile(r"[a-z0-9]+")

_ADJECTIVE_WORDS = {
    "abstract",
    "smooth",
    "compact",
    "proper",
    "projective",
    "affine",
    "classical",
    "modern",
    "finite",
    "infinite",
    "local",
    "global",
    "algebraic",
    "analytic",
    "topological",
    "differential",
    "geometric",
    "homological",
    "categorical",
    "commutative",
    "noncommutative",
    "linear",
    "nonlinear",
}


@dataclass(frozen=True)
class MSCMatch:
    code: Optional[str]
    level: Optional[int]


class MSCDictionary:
    def __init__(
        self,
        *,
        codes_5_digit: Set[str],
        codes_3_digit: Set[str],
        codes_2_digit: Set[str],
        tokens_5_digit: Dict[str, Set[str]],
        tokens_3_digit: Dict[str, Set[str]],
        tokens_2_digit: Dict[str, Set[str]],
    ) -> None:
        self.codes_5_digit = codes_5_digit
        self.codes_3_digit = codes_3_digit
        self.codes_2_digit = codes_2_digit
        self._tokens_5_digit = tokens_5_digit
        self._tokens_3_digit = tokens_3_digit
        self._tokens_2_digit = tokens_2_digit

    @classmethod
    def from_csv(cls, path: str | Path) -> "MSCDictionary":
        p = Path(path)
        fieldnames, rows = _read_csv_rows(p)

        code_key = _pick_key(fieldnames, ["code", "msc", "msc_code", "MSC", "MSC2020"])
        desc_key = _pick_key(
            fieldnames, ["description", "name", "label", "title", "desc"]
        )
        if not code_key or not desc_key:
            raise ValueError("MSC CSV must provide code and description columns")

        tokens_5: Dict[str, Set[str]] = {}
        tokens_3: Dict[str, Set[str]] = {}
        tokens_2: Dict[str, Set[str]] = {}
        for row in rows:
            raw_code = (row.get(code_key) or "").strip()
            desc = (row.get(desc_key) or "").strip()
            code = _normalize_code(raw_code)
            if not code or not desc:
                continue
            tokens = set(_tokenize(desc))
            if not tokens:
                continue
            if FIVE_DIGIT_RE.match(code):
                tokens_5.setdefault(code, set()).update(tokens)
            elif THREE_DIGIT_RE.match(code):
                tokens_3.setdefault(code, set()).update(tokens)
            elif TWO_DIGIT_RE.match(code):
                tokens_2.setdefault(code, set()).update(tokens)

        return cls(
            codes_5_digit=set(tokens_5.keys()),
            codes_3_digit=set(tokens_3.keys()),
            codes_2_digit=set(tokens_2.keys()),
            tokens_5_digit=tokens_5,
            tokens_3_digit=tokens_3,
            tokens_2_digit=tokens_2,
        )

    def match_context(self, context: str) -> MSCMatch:
        tokens = _tokenize(context)
        if not tokens:
            return MSCMatch(code=None, level=None)

        current = tokens
        while current:
            m5 = _best_subset_match(current, self._tokens_5_digit)
            if m5:
                return MSCMatch(code=m5, level=5)

            m3 = _best_subset_match(current, self._tokens_3_digit)
            if m3:
                return MSCMatch(code=m3, level=3)

            m2 = _best_subset_match(current, self._tokens_2_digit)
            if m2:
                return MSCMatch(code=m2, level=2)

            reduced = _strip_one_adjective(current)
            if reduced == current:
                break
            current = reduced

        return MSCMatch(code=None, level=None)


def _read_csv_rows(path: Path) -> tuple[List[str], List[dict]]:
    raw = path.read_bytes()
    text: Optional[str] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(f"Unable to decode MSC CSV: {path}")

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        # Official MSC2020 files are often tab-delimited.
        dialect = (
            csv.excel_tab if sample.count("\t") >= sample.count(",") else csv.excel
        )

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return list(reader.fieldnames or []), list(reader)


def context_similarity(
    query_code: Optional[str], artifact_code: Optional[str]
) -> float:
    if not query_code or not artifact_code:
        return 0.0
    if query_code == artifact_code:
        return 1.0
    if _to_section_prefix(query_code, width=3) == _to_section_prefix(
        artifact_code, width=3
    ):
        return 0.75
    if _to_section_prefix(query_code) == _to_section_prefix(artifact_code):
        return 0.5
    return 0.0


def _pick_key(keys: Iterable[str], preferred: List[str]) -> Optional[str]:
    by_lower = {k.lower(): k for k in keys}
    for p in preferred:
        if p.lower() in by_lower:
            return by_lower[p.lower()]
    return None


def _normalize_code(code: str) -> Optional[str]:
    if not code:
        return None
    c = code.replace(" ", "").upper()
    if re.match(r"^\d{2}[A-Z]\d{2}$", c):
        return c
    if re.match(r"^\d{2}[A-Z]XX$", c):
        return c[:3] + "xx"
    if re.match(r"^\d{2}-XX$", c):
        return c
    return None


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _is_adjective(token: str) -> bool:
    if token in _ADJECTIVE_WORDS:
        return True
    return token.endswith(("al", "ic", "ive", "ous", "ary", "ory", "ed"))


def _strip_one_adjective(tokens: List[str]) -> List[str]:
    for i, tok in enumerate(tokens):
        if _is_adjective(tok):
            return tokens[:i] + tokens[i + 1 :]
    return tokens


def _best_subset_match(
    tokens: List[str], token_map: Dict[str, Set[str]]
) -> Optional[str]:
    required = set(tokens)
    if not required:
        return None
    matches = []
    for code, corpus_tokens in token_map.items():
        if required.issubset(corpus_tokens):
            matches.append(code)
    if not matches:
        return None
    # deterministic tie-break: smaller token space first, then lexical code
    matches.sort(key=lambda c: (len(token_map[c]), c))
    return matches[0]


def _to_section_prefix(code: str, *, width: int = 2) -> Optional[str]:
    if width == 3:
        if FIVE_DIGIT_RE.match(code):
            return code[:3]
        if THREE_DIGIT_RE.match(code):
            return code[:3]
        return None
    if THREE_DIGIT_RE.match(code):
        return code[:2]
    if TWO_DIGIT_RE.match(code):
        return code[:2]
    if FIVE_DIGIT_RE.match(code):
        return code[:2]
    return None
