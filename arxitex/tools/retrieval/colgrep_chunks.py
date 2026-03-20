"""Generate ColGREP-friendly statement chunks from a graph + TeX sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from arxitex.downloaders.utils import read_and_combine_tex_files

ALLOWED_TYPES = {
    "theorem",
    "lemma",
    "definition",
    "proposition",
    "corollary",
    "example",
    "remark",
    "claim",
}

SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
SUBSECTION_RE = re.compile(r"\\subsection\*?\{([^}]*)\}")


@dataclass
class SectionInfo:
    section: str
    subsection: str


@dataclass
class ChunkRecord:
    statement_id: str
    filename: str
    path: str
    type: str
    number: str
    section: str
    subsection: str
    title: str
    arxiv_id: str
    line_start: int
    prev_paragraph: str
    text_preview: str


def _strip_comments(tex: str) -> str:
    # Match extractor: remove comments not escaped by backslash.
    return re.sub(r"(?<!\\)%.*", "", tex)


def _collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def _split_paragraphs_with_index(lines: List[str]) -> Tuple[List[str], List[int]]:
    paragraphs: List[str] = []
    line_to_para = [-1 for _ in lines]
    current: List[str] = []
    para_idx = -1
    for i, line in enumerate(lines):
        if line.strip():
            if not current:
                para_idx += 1
                paragraphs.append("")
            current.append(line.strip())
            paragraphs[para_idx] = " ".join(current).strip()
            line_to_para[i] = para_idx
        else:
            current = []
    return paragraphs, line_to_para


def _build_section_map(lines: List[str]) -> List[SectionInfo]:
    section = 0
    subsection = 0
    section_by_line: List[SectionInfo] = []
    for line in lines:
        sec_match = SECTION_RE.search(line)
        if sec_match:
            section += 1
            subsection = 0
        sub_match = SUBSECTION_RE.search(line)
        if sub_match and section > 0:
            subsection += 1
        section_by_line.append(
            SectionInfo(
                section=str(section) if section > 0 else "",
                subsection=(
                    f"{section}.{subsection}" if subsection > 0 and section > 0 else ""
                ),
            )
        )
    return section_by_line


def _safe_filename(statement_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", statement_id)
    digest = hashlib.sha1(statement_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}.md"


def _load_graph(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "graph" in payload:
        return payload["graph"]
    return payload


def _extract_number(node: Dict) -> str:
    number = (node.get("pdf_label_number") or "").strip()
    if not number and node.get("pdf_label"):
        m = re.search(r"(\d+(?:\.\d+)*)", node.get("pdf_label") or "")
        if m:
            number = m.group(1)
    return number


def _extract_prev_paragraph(
    line_start: int,
    paragraphs: List[str],
    line_to_para: List[int],
) -> str:
    if line_start <= 0:
        return ""
    idx = line_start - 1
    if idx < 0 or idx >= len(line_to_para):
        return ""
    para_idx = line_to_para[idx]
    if para_idx <= 0:
        return ""
    prev = paragraphs[para_idx - 1] if para_idx - 1 < len(paragraphs) else ""
    return _collapse_ws(prev)


def build_chunks(
    *,
    graph_path: Path,
    source_dir: Path,
    out_dir: Path,
    arxiv_id: str,
    title: str,
) -> List[ChunkRecord]:
    graph = _load_graph(graph_path)
    nodes = graph.get("nodes") or []

    raw_tex = read_and_combine_tex_files(source_dir)
    if not raw_tex:
        raise RuntimeError(f"No TeX sources found in {source_dir}")
    cleaned = _strip_comments(raw_tex)
    lines = cleaned.splitlines()

    paragraphs, line_to_para = _split_paragraphs_with_index(lines)
    section_by_line = _build_section_map(lines)

    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[ChunkRecord] = []

    for node in nodes:
        node_type = (node.get("type") or "").lower().strip(".")
        if node_type not in ALLOWED_TYPES:
            continue
        statement_id = node.get("id") or ""
        if not statement_id:
            continue
        line_start = int((node.get("position") or {}).get("line_start") or 0)
        section = ""
        subsection = ""
        if 0 < line_start <= len(section_by_line):
            info = section_by_line[line_start - 1]
            section = info.section
            subsection = info.subsection

        number = _extract_number(node)
        text = (node.get("content") or "").strip()
        proof = (node.get("proof") or "").strip()
        prev_para = _extract_prev_paragraph(line_start, paragraphs, line_to_para)

        filename = _safe_filename(statement_id)
        path = out_dir / filename

        header_lines = [
            f"TITLE: {_collapse_ws(title)}",
            f"SECTION: {section}",
            f"SUBSECTION: {subsection}",
            f"TYPE: {node_type}",
            f"NUMBER: {number}",
            f"ARXIV: {arxiv_id}",
            f"ID: {statement_id}",
        ]

        content_parts = ["\n".join(header_lines), ""]
        if prev_para:
            content_parts.append("PREV_PARAGRAPH:\n" + prev_para)
            content_parts.append("")
        if text:
            content_parts.append("TEXT:\n" + text)
            content_parts.append("")
        if proof:
            content_parts.append("PROOF:\n" + proof)
            content_parts.append("")

        path.write_text("\n".join(content_parts).strip() + "\n", encoding="utf-8")

        preview = _collapse_ws(text)[:400]
        records.append(
            ChunkRecord(
                statement_id=statement_id,
                filename=filename,
                path=str(path),
                type=node_type,
                number=number,
                section=section,
                subsection=subsection,
                title=title,
                arxiv_id=arxiv_id,
                line_start=line_start,
                prev_paragraph=prev_para,
                text_preview=preview,
            )
        )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    logger.info("Wrote {} chunk files to {}", len(records), out_dir)
    logger.info("Wrote manifest to {}", manifest_path)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ColGREP statement chunks.")
    parser.add_argument("--graph", required=True, help="Graph JSON path.")
    parser.add_argument("--source-dir", required=True, help="TeX source directory.")
    parser.add_argument("--out-dir", required=True, help="Output chunk directory.")
    parser.add_argument("--arxiv-id", required=True, help="Target arXiv id.")
    parser.add_argument("--title", default="", help="Target paper title.")
    args = parser.parse_args()

    title = args.title or args.arxiv_id
    build_chunks(
        graph_path=Path(args.graph),
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        arxiv_id=args.arxiv_id,
        title=title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
