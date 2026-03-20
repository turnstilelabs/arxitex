"""Statement graph merge + gold-link eligibility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from arxitex.downloaders.utils import read_and_combine_tex_files
from arxitex.tools.mentions.mapping.ref_artifact_mapper import build_target_registry

ALLOWED_STATEMENT_TYPES = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "example",
    "remark",
}


def filter_nodes(nodes: List[Dict]) -> List[Dict]:
    return [
        node
        for node in nodes
        if (node.get("type") or "").lower().strip(".") in ALLOWED_STATEMENT_TYPES
    ]


def prefix_label(arxiv_id: str, label: str) -> str:
    if not label:
        return label
    if label.startswith(f"{arxiv_id}:"):
        return label
    return f"{arxiv_id}:{label}"


def compute_section_map(source_dir: Optional[str]) -> Optional[List[int]]:
    if not source_dir:
        return None
    text = read_and_combine_tex_files(Path(source_dir))
    if not text:
        return None
    lines = text.splitlines()
    section = 0
    section_by_line: List[int] = []
    for line in lines:
        if "\\section{" in line or "\\section*" in line:
            section += 1
        section_by_line.append(section)
    return section_by_line


def assign_fallback_numbers(
    nodes: List[Dict], section_by_line: Optional[List[int]]
) -> None:
    counters: Dict[Tuple[str, int], int] = {}
    theorem_like = {
        "theorem",
        "lemma",
        "proposition",
        "corollary",
        "definition",
        "remark",
        "example",
    }
    for node in sorted(
        nodes, key=lambda n: (n.get("position", {}).get("line_start") or 0)
    ):
        if (node.get("pdf_label_number") or "").strip():
            continue
        kind = (node.get("type") or "").lower().strip(".")
        counter_kind = "theorem_like" if kind in theorem_like else kind
        line_start = (node.get("position") or {}).get("line_start")
        section = 0
        if section_by_line and isinstance(line_start, int) and line_start > 0:
            if line_start - 1 < len(section_by_line):
                section = section_by_line[line_start - 1]
        key = (counter_kind, section)
        counters[key] = counters.get(key, 0) + 1
        num = f"{section}.{counters[key]}" if section > 0 else str(counters[key])
        node["pdf_label_number"] = num


def merge_statements(
    statement_paths: List[Path],
    out_path: Path,
    source_map: Dict[str, Optional[str]],
) -> Dict:
    merged_nodes: List[Dict] = []
    for p in statement_paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload_arxiv_id = payload.get("arxiv_id") or p.stem
        nodes = filter_nodes(payload.get("nodes") or [])
        section_by_line = compute_section_map(source_map.get(payload_arxiv_id))
        assign_fallback_numbers(nodes, section_by_line)
        for node in nodes:
            old_id = node.get("id")
            node_arxiv_id = node.get("arxiv_id") or payload_arxiv_id
            if old_id:
                if old_id.startswith(f"{node_arxiv_id}:"):
                    node["id"] = old_id
                else:
                    node["id"] = f"{node_arxiv_id}:{old_id}"
            node["arxiv_id"] = node_arxiv_id
            label_num = (node.get("pdf_label_number") or "").strip()
            if label_num:
                node["pdf_label_number"] = prefix_label(node_arxiv_id, label_num)
            merged_nodes.append(node)

    merged = {
        "arxiv_id": "combined",
        "extractor_mode": "merged-statements",
        "stats": {"nodes": len(merged_nodes)},
        "nodes": merged_nodes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    logger.info("Wrote combined statements {}", out_path)
    return merged


def find_duplicate_statement_keys(
    nodes: List[Dict],
) -> Dict[Tuple[str, str, str], List[str]]:
    # Gold-link mapping assumes a unique target per (arxiv_id, kind, number).
    by_key: Dict[Tuple[str, str, str], List[str]] = {}
    for node in nodes:
        kind = (node.get("type") or "").lower().strip(".")
        number = (node.get("pdf_label_number") or "").strip().lower()
        node_id = node.get("id")
        if not kind or not number or not node_id:
            continue
        if ":" in number:
            number = number.split(":", 1)[1]
        arxiv_id = (node.get("arxiv_id") or "").strip() or node_id.split(":", 1)[0]
        if not arxiv_id:
            continue
        key = (arxiv_id, kind, number)
        by_key.setdefault(key, []).append(node_id)
    return {k: sorted(set(v)) for k, v in by_key.items() if len(set(v)) > 1}


def write_statement_uniqueness_report(
    *,
    duplicates: Dict[Tuple[str, str, str], List[str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (arxiv_id, kind, number), node_ids in sorted(duplicates.items()):
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "kind": kind,
                "number": number,
                "node_ids": node_ids,
                "count": len(node_ids),
            }
        )
    out_path.write_text(
        json.dumps(
            {
                "duplicate_key_count": len(rows),
                "duplicate_node_count": sum(r["count"] for r in rows),
                "duplicates": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_statement_corpus(
    statements: Dict,
    out_path: Path,
    *,
    ineligible_statement_ids: Optional[Set[str]] = None,
) -> None:
    ineligible_statement_ids = ineligible_statement_ids or set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for node in statements.get("nodes") or []:
            text = (
                node.get("content")
                or node.get("semantic_tag")
                or node.get("content_preview")
                or ""
            )
            if not text:
                continue
            f.write(
                json.dumps(
                    {
                        "statement_id": node.get("id"),
                        "statement_text": text,
                        "arxiv_id": (node.get("id") or "").split(":", 1)[0],
                        "kind": node.get("type"),
                        "number": node.get("pdf_label_number"),
                        "eligible_for_gold_links": (
                            (node.get("id") or "") not in ineligible_statement_ids
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_statement_registry_and_corpus(
    *,
    statement_paths: List[Path],
    source_map: Dict[str, Optional[str]],
    out_dir: Path,
    fail_on_duplicates: bool = False,
):
    """Build combined statements, write corpus artifacts, and return mapping registry."""

    combined_statements_path = out_dir / "combined_statements.json"
    merged = merge_statements(statement_paths, combined_statements_path, source_map)
    nodes = merged.get("nodes") or []

    duplicates = find_duplicate_statement_keys(nodes)
    ineligible_ids = {
        node_id for node_ids in duplicates.values() for node_id in node_ids
    }
    uniqueness_report_path = out_dir / "statement_uniqueness_report.json"
    write_statement_uniqueness_report(
        duplicates=duplicates,
        out_path=uniqueness_report_path,
    )
    if duplicates:
        logger.warning(
            "Found {} duplicate statement kind/number keys; excluding {} nodes from gold-link mapping (report: {})",
            len(duplicates),
            len(ineligible_ids),
            uniqueness_report_path,
        )
        if fail_on_duplicates:
            raise RuntimeError(
                f"Duplicate statement keys found. See {uniqueness_report_path}"
            )

    gold_link_registry_nodes = [
        node for node in nodes if (node.get("id") or "") not in ineligible_ids
    ]
    write_statement_corpus(
        merged,
        out_dir / "statements.jsonl",
        ineligible_statement_ids=ineligible_ids,
    )
    return build_target_registry(
        gold_link_registry_nodes,
        default_version_id="combined",
    )
