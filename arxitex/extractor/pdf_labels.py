"""Annotate graph nodes with PDF-visible labels using SyncTeX.

Flow:
1) Map a node's source line to a PDF location via SyncTeX.
2) Extract PDF text (with bboxes when available).
3) Find the nearest numbered label (e.g., "Theorem 1.1") and attach it to the node.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLine
except Exception as exc:  # pragma: no cover - runtime dependency
    extract_pages = None
    LTTextContainer = None
    LTTextLine = None
    _PDFMINER_IMPORT_ERROR = exc
else:
    _PDFMINER_IMPORT_ERROR = None


ENV_TYPES = {
    "theorem": "Theorem",
    "thm": "Theorem",
    "lemma": "Lemma",
    "lem": "Lemma",
    "proposition": "Proposition",
    "prop": "Proposition",
    "corollary": "Corollary",
    "cor": "Corollary",
    "claim": "Claim",
    "clm": "Claim",
    "conjecture": "Conjecture",
    "conj": "Conjecture",
    "definition": "Definition",
    "def": "Definition",
    "example": "Example",
    "ex": "Example",
    "remark": "Remark",
    "rem": "Remark",
}

PDF_LABEL_RE = re.compile(
    r"\b(Theorem|Thm\.|Lemma|Lem\.|Proposition|Prop\.|Corollary|Cor\.|Claim|Clm\.|Conjecture|Conj\.|Definition|Def\.|Example|Ex\.|Remark|Rem\.)\s*"
    r"([0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)


class _SourceLoc:
    def __init__(self, file_path: Path, line: int):
        self.file_path = file_path
        self.line = line


def _build_combined_line_map(tex_root: Path) -> Dict[int, Optional[_SourceLoc]]:
    """Map combined line numbers (from concatenated .tex files) to source files."""
    tex_files = sorted(list(tex_root.rglob("*.tex")))
    mapping: Dict[int, Optional[_SourceLoc]] = {}
    combined_line = 0
    for tex_file in tex_files:
        # Match read_and_combine_tex_files: "\n% --- Source File: X ---\n{content}"
        combined_line += 1  # leading blank line
        mapping[combined_line] = None
        combined_line += 1  # source file comment line
        mapping[combined_line] = None
        try:
            lines = tex_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            logger.warning("Could not read {}: {}", tex_file, exc)
            continue
        for idx, _ in enumerate(lines, start=1):
            combined_line += 1
            mapping[combined_line] = _SourceLoc(file_path=tex_file, line=idx)
    return mapping


def _resolve_source(
    mapping: Dict[int, Optional[_SourceLoc]], line: int
) -> Optional[_SourceLoc]:
    if not line:
        return None
    for offset in range(0, 6):
        loc = mapping.get(line + offset)
        if loc is not None:
            return loc
    return None


def _run_synctex_view(
    pdf_path: Path, source_file: Path, line: int, column: int
) -> Optional[Dict[str, float]]:
    """Resolve a source line to a PDF location using SyncTeX."""
    cmd = [
        "synctex",
        "view",
        "-i",
        f"{line}:{column}:{source_file}",
        "-o",
        str(pdf_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
    except FileNotFoundError:
        logger.error("synctex CLI not found; install TeX Live or Synctex.")
        return None
    except subprocess.CalledProcessError as exc:
        logger.debug("synctex view failed: {}", exc.stderr.strip())
        return None

    page = None
    x = y = w = h = 0.0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Page:"):
            try:
                page = int(line.split(":", 1)[1].strip())
            except Exception:
                page = None
        elif line.startswith("x:"):
            try:
                x = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("y:"):
            try:
                y = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("W:"):
            try:
                w = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("H:"):
            try:
                h = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
    if page is None:
        return None
    return {"page": page, "x": x, "y": y, "w": w, "h": h}


def _load_pdf_text(
    pdf_path: Path,
) -> Tuple[
    Dict[int, List[Tuple[str, Optional[Tuple[float, float, float, float]]]]], bool
]:
    """Extract per-page text (and bbox when available) from the PDF."""
    if extract_pages is not None:
        page_text: Dict[
            int, List[Tuple[str, Optional[Tuple[float, float, float, float]]]]
        ] = {}
        for page_num, page_layout in enumerate(extract_pages(str(pdf_path)), start=1):
            items: List[Tuple[str, Tuple[float, float, float, float]]] = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for line in element:
                        if not isinstance(line, LTTextLine):
                            continue
                        text = line.get_text().strip()
                        if text:
                            items.append((text, line.bbox))
            page_text[page_num] = items
        return page_text, True

    if not shutil.which("pdftotext"):
        raise RuntimeError(
            "pdfminer.six is required and pdftotext is not available. "
            f"pdfminer error: {_PDFMINER_IMPORT_ERROR}"
        )

    cmd = ["pdftotext", "-bbox-layout", str(pdf_path), "-"]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        logger.warning("pdftotext -bbox-layout failed, falling back to text-only mode.")
        page_text: Dict[
            int, List[Tuple[str, Optional[Tuple[float, float, float, float]]]]
        ] = {}
        page_count = None
        if shutil.which("qpdf"):
            info = subprocess.run(
                ["qpdf", "--show-npages", str(pdf_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            info_text = info.stdout.decode("utf-8", errors="ignore").strip()
            if info_text.isdigit():
                page_count = int(info_text)
        if page_count is None:
            page_count = 1
        for page in range(1, page_count + 1):
            proc = subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-layout",
                    str(pdf_path),
                    "-",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            text_out = proc.stdout.decode("utf-8", errors="ignore")
            lines: List[Tuple[str, Optional[Tuple[float, float, float, float]]]] = []
            for line in text_out.splitlines():
                if line.strip():
                    lines.append((line.rstrip(), None))
            page_text[page] = lines
        return page_text, False

    page_text: Dict[
        int, List[Tuple[str, Optional[Tuple[float, float, float, float]]]]
    ] = {}
    page_num = None
    text_re = re.compile(
        r'<text[^>]*?bbox="([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)"[^>]*>(.*?)</text>'
    )
    text_out = proc.stdout.decode("utf-8", errors="ignore")
    for line in text_out.splitlines():
        line = line.strip()
        if line.startswith("<page "):
            m = re.search(r'number="(\d+)"', line)
            if m:
                page_num = int(m.group(1))
                page_text.setdefault(page_num, [])
            continue
        if page_num is None:
            continue
        m = text_re.search(line)
        if not m:
            continue
        x0, y0, x1, y1, text = m.groups()
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            continue
        bbox = (float(x0), float(y0), float(x1), float(y1))
        page_text[page_num].append((text, bbox))
    return page_text, True


def _overlaps(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _rect_distance(
    rect: Tuple[float, float, float, float],
    bbox: Tuple[float, float, float, float],
) -> float:
    rx0, ry0, rx1, ry1 = rect
    bx0, by0, bx1, by1 = bbox
    cx = (rx0 + rx1) / 2.0
    cy = (ry0 + ry1) / 2.0
    dx = 0.0 if bx0 <= cx <= bx1 else min(abs(cx - bx0), abs(cx - bx1))
    dy = 0.0 if by0 <= cy <= by1 else min(abs(cy - by0), abs(cy - by1))
    return (dx * dx + dy * dy) ** 0.5


def _find_label_near(
    lines: List[Tuple[str, Optional[Tuple[float, float, float, float]]]],
    hit: Dict[str, float],
    max_distance: float,
    expected_label: str,
) -> Optional[Tuple[str, str]]:
    """Find a numbered label near a SyncTeX hit using bbox proximity."""
    x = hit["x"]
    y = hit["y"]
    w = hit["w"]
    h = hit["h"]
    rect = (x, y, x + max(w, 50.0), y + max(h, 12.0))
    margin = 80.0
    expanded = (rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin)

    candidates: List[Tuple[float, str, str]] = []
    for text, bbox in lines:
        match = PDF_LABEL_RE.search(text)
        if match:
            label = f"{match.group(1).rstrip('.')}".capitalize()
            if label.lower() != expected_label.lower():
                continue
            number = match.group(2)
            if bbox is None:
                dist = 0.0
            elif _overlaps(expanded, bbox):
                dist = _rect_distance(rect, bbox)
            else:
                rx0, ry0, rx1, ry1 = rect
                cx = (rx0 + rx1) / 2.0
                cy = (ry0 + ry1) / 2.0
                bx0, by0, bx1, by1 = bbox
                lx = (bx0 + bx1) / 2.0
                ly = (by0 + by1) / 2.0
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            candidates.append((dist, label, number))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    dist, label, number = candidates[0]
    if dist > max_distance:
        return None
    return label, number


def _strip_tex_to_anchor(text: str) -> str:
    s = re.sub(r"\\[a-zA-Z@]+(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    s = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\{|\}|\[|\]|\(|\)", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _canonicalize_tex_text(text: str) -> str:
    """Flatten LaTeX into a comparable, alphanumeric text string."""
    if not text:
        return ""
    s = text
    # Preserve common math tokens like Q_p -> Qp before stripping commands.
    s = re.sub(r"\\mathbb\{([A-Za-z])\}_\{?([A-Za-z0-9]+)\}?", r"\1\2", s)
    s = re.sub(r"\\mathbb\{([A-Za-z])\}", r"\1", s)
    s = re.sub(r"([A-Za-z])_\{?([A-Za-z0-9]+)\}?", r"\1\2", s)
    s = re.sub(r"([A-Za-z])\^\{?([A-Za-z0-9]+)\}?", r"\1\2", s)
    # Drop LaTeX commands but keep arguments.
    s = re.sub(r"\\[a-zA-Z@]+(\[[^\]]*\])?", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    # Replace math markers and punctuation with spaces.
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _canonicalize_pdf_line(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", " ", text)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


_STOPWORDS = {
    "the",
    "of",
    "and",
    "are",
    "is",
    "a",
    "an",
    "to",
    "for",
    "in",
    "on",
    "with",
    "by",
    "let",
    "be",
    "then",
    "that",
    "this",
    "these",
    "those",
    "as",
    "we",
    "our",
    "their",
    "its",
    "from",
    "over",
    "under",
}


_SINGLE_CHAR_TOKENS = {"t", "k", "q", "f", "r", "s", "z"}


def _tokenize(text: str) -> List[str]:
    tokens = []
    for tok in text.split():
        if tok in _STOPWORDS:
            continue
        if (
            len(tok) >= 2
            or any(ch.isdigit() for ch in tok)
            or tok in _SINGLE_CHAR_TOKENS
        ):
            tokens.append(tok)
    return tokens


def _find_best_line_by_similarity(
    lines: List[Tuple[str, Optional[Tuple[float, float, float, float]]]],
    canonical_text: str,
) -> Optional[int]:
    if not canonical_text:
        return None
    target_tokens = _tokenize(canonical_text)
    if not target_tokens:
        return None
    freq: Dict[str, int] = {}
    line_tokens: List[List[str]] = []
    for line, _ in lines:
        cand = _canonicalize_pdf_line(line)
        tokens = _tokenize(cand) if cand else []
        line_tokens.append(tokens)
        for tok in set(tokens):
            freq[tok] = freq.get(tok, 0) + 1
    best_idx = None
    best_score = 0.0
    second_best = 0.0
    target_set = set(target_tokens)
    for i, cand_tokens in enumerate(line_tokens):
        if not cand_tokens:
            continue
        cand_set = set(cand_tokens)
        overlap = target_set & cand_set
        if not overlap:
            continue
        score = sum(1.0 / float(freq.get(tok, 1)) for tok in overlap)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_idx = i
        elif score > second_best:
            second_best = score
    if best_idx is None:
        return None
    # Require a distinctive match and a small margin.
    if best_score < 0.5 or (best_score - second_best) < 0.1:
        return None
    return best_idx


def _extract_anchor(text: str) -> str:
    """Extract the longest non-math text span to use as a PDF anchor."""
    cleaned = _strip_tex_to_anchor(text)
    if not cleaned:
        return ""
    # Split into candidate spans on sentence-like boundaries.
    spans = re.split(r"[.;:!?]\s+", cleaned)
    spans = [s.strip() for s in spans if s.strip()]
    if not spans:
        return ""
    # Choose the span with the most words.
    spans.sort(key=lambda s: len(s.split()), reverse=True)
    best = spans[0]
    words = best.split()
    if len(words) >= 3:
        return " ".join(words)
    # Fallback to first 4 words of the cleaned text.
    return " ".join(cleaned.split()[:4])


def _find_label_in_lines(
    lines: List[Tuple[str, Optional[Tuple[float, float, float, float]]]],
    anchor_text: str,
    expected_label: str,
) -> Optional[Tuple[str, str]]:
    """Locate label in nearby text lines using a light token overlap check."""
    if not anchor_text:
        return None
    target_tokens = set(_tokenize(anchor_text))
    if not target_tokens:
        return None
    line_tokens: List[List[str]] = []
    for line, _ in lines:
        cand = _canonicalize_pdf_line(line)
        tokens = _tokenize(cand) if cand else []
        line_tokens.append(tokens)
    # Candidate label lines of the expected type.
    candidate_indices = []
    for i, (line, _) in enumerate(lines):
        match = PDF_LABEL_RE.search(line)
        if not match:
            continue
        label = f"{match.group(1).rstrip('.')}".capitalize()
        if label.lower() != expected_label.lower():
            continue
        candidate_indices.append(i)
    if not candidate_indices:
        return None
    # Simple overlap check: label line + next two lines.
    good_candidates = []
    for i in candidate_indices:
        window_tokens: List[str] = []
        for j in range(i, min(i + 3, len(line_tokens))):
            window_tokens.extend(line_tokens[j])
        overlap = target_tokens & set(window_tokens)
        if len(overlap) >= 2:
            good_candidates.append(i)
    # Use anchor similarity to approximate location on the page.
    anchor_idx = _find_best_line_by_similarity(lines, anchor_text)

    def _closest_to_anchor(indices: List[int]) -> Optional[int]:
        if anchor_idx is None:
            return indices[0] if indices else None
        best = None
        best_dist = None
        for i in indices:
            dist = abs(i - anchor_idx)
            if (
                best_dist is None
                or dist < best_dist
                or (dist == best_dist and i < best)
            ):
                best = i
                best_dist = dist
        return best

    # Prefer overlap-validated candidates; otherwise closest label line.
    idx = (
        _closest_to_anchor(good_candidates)
        if good_candidates
        else _closest_to_anchor(candidate_indices)
    )
    if idx is not None:
        match = PDF_LABEL_RE.search(lines[idx][0])
        if match:
            return f"{match.group(1).rstrip('.')}".capitalize(), match.group(2)
    return None


def annotate_nodes_with_pdf_labels(
    nodes: List,
    tex_root: Path,
    pdf_path: Path,
    synctex_column: int = 1,
    pdf_label_max_distance: float = 200.0,
) -> int:
    """Annotate graph nodes with PDF-visible labels (e.g. 'Theorem 1.1')."""
    mapping = _build_combined_line_map(tex_root)
    if not mapping:
        return 0
    pdf_text, pdf_has_bbox = _load_pdf_text(pdf_path)
    updated = 0
    # Track nodes for a post-pass disambiguation on pages with multiple labels.
    page_type_nodes: Dict[Tuple[int, str], List] = {}
    for node in nodes:
        node_type = getattr(node, "type", None) or (
            node.get("type") if isinstance(node, dict) else None
        )
        node_type_value = getattr(node_type, "value", None) or node_type
        if not node_type_value or node_type_value not in ENV_TYPES:
            continue
        expected_label = ENV_TYPES[node_type_value]
        # Clear any stale labels before re-annotating.
        if isinstance(node, dict):
            node["pdf_label"] = None
            node["pdf_label_type"] = None
            node["pdf_label_number"] = None
            node["pdf_page"] = None
            node["source_file"] = None
            node["source_line_start"] = None
        else:
            node.pdf_label = None
            node.pdf_label_type = None
            node.pdf_label_number = None
            node.pdf_page = None
            node.source_file = None
            node.source_line_start = None
        position = getattr(node, "position", None) or (
            node.get("position") if isinstance(node, dict) else {}
        )
        line_start = getattr(position, "line_start", None) or (
            position.get("line_start") if isinstance(position, dict) else None
        )
        if not line_start:
            continue
        source_loc = _resolve_source(mapping, int(line_start))
        if not source_loc:
            continue
        hit = _run_synctex_view(
            pdf_path, source_loc.file_path, source_loc.line, synctex_column
        )
        if not hit:
            continue
        lines = pdf_text.get(hit["page"], [])
        content = getattr(node, "content", None) or (
            node.get("content") if isinstance(node, dict) else ""
        )
        anchor = _canonicalize_tex_text(content or "")
        found = _find_label_in_lines(lines, anchor, expected_label)
        if not found and pdf_has_bbox:
            found = _find_label_near(lines, hit, pdf_label_max_distance, expected_label)
        if not found:
            continue
        pdf_label, pdf_number = found
        if isinstance(node, dict):
            node["source_file"] = str(source_loc.file_path)
            node["source_line_start"] = source_loc.line
            node["pdf_page"] = hit["page"]
            node["pdf_label"] = f"{pdf_label} {pdf_number}"
            node["pdf_label_type"] = pdf_label
            node["pdf_label_number"] = pdf_number
        else:
            node.source_file = str(source_loc.file_path)
            node.source_line_start = source_loc.line
            node.pdf_page = hit["page"]
            node.pdf_label = f"{pdf_label} {pdf_number}"
            node.pdf_label_type = pdf_label
            node.pdf_label_number = pdf_number
        updated += 1
        page_type_nodes.setdefault((hit["page"], expected_label), []).append(node)

    # Post-pass: disambiguate pages with multiple labels of the same type.
    for (page, expected_label), group in page_type_nodes.items():
        if len(group) <= 1:
            continue
        lines = pdf_text.get(page, [])
        label_lines = []
        for idx, (line, _) in enumerate(lines):
            match = PDF_LABEL_RE.search(line)
            if not match:
                continue
            label = f"{match.group(1).rstrip('.')}".capitalize()
            if label.lower() != expected_label.lower():
                continue
            label_lines.append((idx, match.group(2)))
        if len(label_lines) <= 1:
            continue
        label_lines.sort(key=lambda x: x[0])

        # Order nodes by anchor position (or by source line as fallback).
        ordered = []
        for node in group:
            content = getattr(node, "content", None) or (
                node.get("content") if isinstance(node, dict) else ""
            )
            anchor = _canonicalize_tex_text(content or "")
            anchor_idx = (
                _find_best_line_by_similarity(lines, anchor) if anchor else None
            )
            source_line = (
                getattr(node, "source_line_start", None)
                or (node.get("source_line_start") if isinstance(node, dict) else None)
                or 0
            )
            order_key = anchor_idx if anchor_idx is not None else source_line
            ordered.append((order_key, node))
        ordered.sort(key=lambda x: x[0])

        for (_, label_number), (_, node) in zip(label_lines, ordered):
            if isinstance(node, dict):
                node["pdf_label"] = f"{expected_label} {label_number}"
                node["pdf_label_type"] = expected_label
                node["pdf_label_number"] = label_number
            else:
                node.pdf_label = f"{expected_label} {label_number}"
                node.pdf_label_type = expected_label
                node.pdf_label_number = label_number
    return updated
