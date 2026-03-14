#!/usr/bin/env python3
"""Build mention-supervised statement retrieval dataset.

Pipeline:
1) Extract statement lists for each target arXiv paper (artifacts only).
2) Collect citation mentions via OpenAlex + ar5iv/PDF.
3) Build unified statement corpus + mention dataset + queries/qrels.
4) Split by target paper (train/val/test).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.downloaders.utils import read_and_combine_tex_files
from arxitex.extractor.pipeline import agenerate_artifact_graph as agenerate_statements
from arxitex.tools.mentions.acquisition.openalex_citations import (
    OpenAlexCitingWorksStage,
)
from arxitex.tools.mentions.acquisition.target_resolution import OpenAlexTargetResolver
from arxitex.tools.mentions.dataset.definition_context import (
    _build_definition_index,
    _collect_paragraphs,
    _definitions_for_symbols,
    _extract_symbols_from_text,
)
from arxitex.tools.mentions.extraction.extract_mentions_cli import (
    MentionContextExtractionStage,
)
from arxitex.tools.mentions.extraction.mention_utils import (
    normalize_for_match,
    split_sentences,
)
from arxitex.tools.mentions.utils import extract_refs
from arxitex.utils import ensure_dir, read_jsonl, sha256_hash

ALLOWED_TYPES = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "example",
    "remark",
}

TYPE_KEYWORDS = {
    "definition": [r"\bdefinition\b", r"\bdef\.?\b"],
    "theorem": [r"\btheorem\b", r"\bthm\.?\b"],
    "lemma": [r"\blemma\b", r"\blem\.?\b"],
    "proposition": [r"\bproposition\b", r"\bprop\.?\b"],
    "corollary": [r"\bcorollary\b", r"\bcor\.?\b"],
}


def _text_mentions_type(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in patterns)


def _type_conflict(text: str, explicit_kind: str) -> bool:
    if not text or not explicit_kind:
        return False
    kind = explicit_kind.strip().lower()
    if kind == "definition":
        other: List[str] = []
        for k, pats in TYPE_KEYWORDS.items():
            if k != "definition":
                other.extend(pats)
        return _text_mentions_type(text, other)
    if kind in {"theorem", "lemma", "proposition", "corollary"}:
        return _text_mentions_type(text, TYPE_KEYWORDS["definition"])
    return False


@dataclass
class Target:
    arxiv_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    local_statements: Optional[str] = None
    openalex_id: Optional[str] = None
    local_source_dir: Optional[str] = None


DEFAULT_TARGETS = [
    Target("perfectoid", local_statements="data/statements/perfectoid.json"),
    Target("math/0608640"),  # Caffarelli–Silvestre
    Target("1709.10033"),  # Buckmaster–Vicol
    Target("1303.5113"),  # Hairer Regularity Structures
]


def _load_targets(path: Optional[str], arxiv_ids: Optional[List[str]]) -> List[Target]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Target(**row) for row in data]
    if arxiv_ids:
        return [Target(a) for a in arxiv_ids]
    return DEFAULT_TARGETS


async def _build_statements(
    arxiv_id: str, out_path: Path, local_source_dir: Optional[str]
) -> None:
    logger.info("Extracting statements for {}", arxiv_id)
    results = await agenerate_statements(
        arxiv_id=arxiv_id,
        infer_dependencies=False,
        enrich_content=False,
        dependency_mode="pairwise",
        dependency_config=None,
        source_dir=None,
        local_source_dir=local_source_dir,
        local_source_id=arxiv_id,
    )
    artifact = results.get("graph")
    if not artifact:
        raise RuntimeError(f"No statements extracted for {arxiv_id}")
    artifact_dict = artifact.to_dict(arxiv_id=arxiv_id, extractor_mode="regex-only")
    nodes = artifact_dict.get("nodes") or []
    statements = {
        "arxiv_id": arxiv_id,
        "extractor_mode": "statements-only",
        "stats": {"nodes": len(nodes)},
        "nodes": nodes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(statements, ensure_ascii=False, indent=2))


def _filter_nodes(nodes: List[Dict]) -> List[Dict]:
    kept = []
    for node in nodes:
        kind = (node.get("type") or "").lower().strip(".")
        if kind not in ALLOWED_TYPES:
            continue
        kept.append(node)
    return kept


def _prefix_label(arxiv_id: str, label: str) -> str:
    if not label:
        return label
    if label.startswith(f"{arxiv_id}:"):
        return label
    return f"{arxiv_id}:{label}"


def _strip_explicit_refs(text: str) -> str:
    if not text:
        return text
    # Remove LaTeX-style refs.
    text = re.sub(r"\\(?:Cref|cref|ref|eqref)\{[^}]+\}", " ", text)
    # Remove explicit identifiers like "Theorem 2.3", "Lemma (1.2)", "Corollary 4".
    text = re.sub(
        r"(?i)\b(theorem|lemma|proposition|corollary|definition|remark|example|claim|equation)s?\b"
        r"\s*(?:~|\\,|\\ )*"
        r"\(?\d+(?:\.\d+)*\)?",
        " ",
        text,
    )
    # Remove phrases like "Conclusion 3 of Theorem" / "Part 2 of Lemma".
    text = re.sub(
        r"(?i)\b(conclusion|part|item)\s+\d+\s+of\s+"
        r"(theorem|lemma|proposition|corollary|definition|remark|example|claim)\b",
        " ",
        text,
    )
    # Collapse whitespace.
    return " ".join(text.split())


def _mask_citations(text: str) -> str:
    if not text:
        return text
    # Mask LaTeX citation commands.
    text = re.sub(
        r"\\cite[a-zA-Z]*\s*\{[^}]*\}",
        " CITATION ",
        text,
    )
    # Mask bracketed numeric citations like [12] or [1, 3, 5].
    text = re.sub(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", " CITATION ", text)
    return " ".join(text.split())


def _cap_definitions(
    def_sentences: List[str],
    max_sentences: int,
    max_words: int,
) -> str:
    if not def_sentences:
        return ""
    sentences = list(def_sentences)
    if max_sentences > 0:
        sentences = sentences[:max_sentences]
    text = " ".join(s.strip() for s in sentences if s).strip()
    if not text:
        return ""
    if max_words > 0:
        text = _truncate_words(text, max_words)
    return text


def _extract_explicit_refs_from_text(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    seen = set()
    refs: List[Dict[str, str]] = []
    for ref in extract_refs(text):
        kind = (ref.get("kind") or "").lower().strip()
        number = (ref.get("number") or "").strip()
        if not kind or not number:
            continue
        if kind not in ALLOWED_TYPES:
            continue
        key = (kind, number)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"kind": kind, "number": number, "raw": ref.get("raw")})
    return refs


def _refs_text_for_inference(row: Dict, context_mode: str) -> str:
    if context_mode == "paragraph":
        para = row.get("context_paragraph") or _html_to_text(
            row.get("context_html") or ""
        )
        if para:
            return para
    parts = [
        row.get("context_prev") or "",
        row.get("context_sentence") or "",
        row.get("context_next") or "",
    ]
    text = " ".join(p for p in parts if p)
    if not text:
        text = row.get("context_sentence") or _html_to_text(
            row.get("context_html") or ""
        )
    return text


def _find_paragraph_index(paragraphs: List[str], para_text: str) -> int:
    if not paragraphs or not para_text:
        return -1
    norm_target = normalize_for_match(para_text)
    if not norm_target:
        return -1
    for i, p in enumerate(paragraphs):
        norm_p = normalize_for_match(p)
        if not norm_p:
            continue
        if norm_target in norm_p or norm_p in norm_target:
            return i
    return -1


class DefinitionIndexCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._cache: Dict[str, Tuple[List[str], Dict[str, List[Tuple[int, str]]]]] = {}

    def _load_html(self, arxiv_id: str) -> Optional[str]:
        if not arxiv_id:
            return None
        url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        cache_path = self.cache_dir / f"{sha256_hash(url)}.html"
        if not cache_path.exists():
            return None
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    def _build(
        self, arxiv_id: str
    ) -> Optional[Tuple[List[str], Dict[str, List[Tuple[int, str]]]]]:
        html = self._load_html(arxiv_id)
        if not html:
            return None
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        paragraphs = _collect_paragraphs(soup)
        paragraph_texts = [p.get_text(" ", strip=True) for p in paragraphs]
        if not paragraph_texts:
            return None
        index = _build_definition_index(paragraph_texts)
        return (paragraph_texts, index)

    def get(
        self, arxiv_id: str
    ) -> Optional[Tuple[List[str], Dict[str, List[Tuple[int, str]]]]]:
        if arxiv_id in self._cache:
            return self._cache[arxiv_id]
        built = self._build(arxiv_id)
        if built is None:
            return None
        self._cache[arxiv_id] = built
        return built

    def definitions_for(self, arxiv_id: str, para_text: str) -> List[str]:
        if not arxiv_id or not para_text:
            return []
        built = self.get(arxiv_id)
        if not built:
            return []
        paragraphs, index = built
        para_idx = _find_paragraph_index(paragraphs, para_text)
        if para_idx < 0:
            return []
        symbols = _extract_symbols_from_text(para_text)
        return _definitions_for_symbols(index, symbols, para_idx)


def _extract_paragraph(lines: List[str], idx: int) -> str:
    if not lines:
        return ""
    start = idx
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = idx
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    para = " ".join(line.strip() for line in lines[start : end + 1]).strip()
    return para


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


def _find_sentence_index(sentences: List[str], target: str) -> int:
    if not sentences or not target:
        return -1
    norm_target = normalize_for_match(target)
    for i, s in enumerate(sentences):
        norm_s = normalize_for_match(s)
        if not norm_s:
            continue
        if norm_s == norm_target or norm_target in norm_s:
            return i
    return -1


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def _select_context(
    row: Dict,
    context_mode: str,
    context_window: int,
    context_max_words: int,
) -> str:
    prev = row.get("context_prev") or ""
    sent = row.get("context_sentence") or ""
    nxt = row.get("context_next") or ""
    para = row.get("context_paragraph") or ""
    if not para:
        para = _html_to_text(row.get("context_html") or "")
    if context_mode == "sentence":
        context = sent
    elif context_mode == "paragraph":
        context = para or " ".join([prev, sent, nxt])
    elif context_mode == "window":
        if para:
            sentences = split_sentences(para)
            idx = _find_sentence_index(sentences, sent)
            if idx >= 0:
                start = max(0, idx - context_window)
                end = min(len(sentences), idx + context_window + 1)
                context = " ".join(sentences[start:end])
            else:
                context = para
        else:
            context = " ".join([prev, sent, nxt])
    else:
        context = " ".join([prev, sent, nxt])
    context = " ".join(context.split())
    return _truncate_words(context, context_max_words)


def _compute_section_map(source_dir: Optional[str]) -> Optional[List[int]]:
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


def _assign_fallback_numbers(
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


def _merge_statements(
    statement_paths: List[Path],
    out_path: Path,
    source_map: Dict[str, Optional[str]],
) -> Dict:
    merged_nodes: List[Dict] = []
    for p in statement_paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        arxiv_id = payload.get("arxiv_id") or p.stem
        nodes = _filter_nodes(payload.get("nodes") or [])
        section_by_line = _compute_section_map(source_map.get(arxiv_id))
        _assign_fallback_numbers(nodes, section_by_line)
        for node in nodes:
            old_id = node.get("id")
            new_id = f"{arxiv_id}:{old_id}" if old_id else None
            if old_id and new_id:
                node["id"] = new_id
            node["arxiv_id"] = arxiv_id
            label_num = (node.get("pdf_label_number") or "").strip()
            if label_num:
                node["pdf_label_number"] = _prefix_label(arxiv_id, label_num)
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


def _build_statement_index(statements: Dict) -> Dict[Tuple[str, str], List[Dict]]:
    index: Dict[Tuple[str, str], List[Dict]] = {}
    for node in statements.get("nodes") or []:
        kind = (node.get("type") or "").lower().strip(".")
        number = (node.get("pdf_label_number") or "").strip()
        if not kind or not number:
            continue
        index.setdefault((kind, number), []).append(node)
    return index


def _statement_text(node: Dict) -> str:
    return (
        node.get("content")
        or node.get("semantic_tag")
        or node.get("content_preview")
        or ""
    )


def _write_statement_corpus(statements: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for node in statements.get("nodes") or []:
            text = _statement_text(node)
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
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _extract_internal_mentions(
    arxiv_id: str,
    source_dir: Optional[str],
    nodes: List[Dict],
) -> List[Dict]:
    if not source_dir:
        return []
    text = read_and_combine_tex_files(Path(source_dir))
    if not text:
        return []
    label_to_node = {(n.get("label") or ""): n for n in nodes if (n.get("label") or "")}
    if not label_to_node:
        return []
    lines = text.splitlines()
    paragraphs, line_to_para = _split_paragraphs_with_index(lines)
    def_index = _build_definition_index(paragraphs) if paragraphs else {}
    pattern = re.compile(
        r"(?i)\b(theorem|lemma|proposition|corollary|definition|remark|example|claim)s?"
        r"\s*(?:~|\\,|\\ )*\\(?:Cref|cref|ref)\{([^}]+)\}"
    )
    seen = set()
    mentions: List[Dict] = []
    for i, line in enumerate(lines):
        for match in pattern.finditer(line):
            kind = match.group(1).lower()
            label = match.group(2)
            key = (i, label, kind)
            if key in seen:
                continue
            seen.add(key)
            node = label_to_node.get(label)
            if not node:
                continue
            number = (node.get("pdf_label_number") or "").strip()
            if ":" in number:
                number = number.split(":", 1)[1]
            if not number:
                continue
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            paragraph = _extract_paragraph(lines, i)
            para_idx = line_to_para[i] if i < len(line_to_para) else -1
            definition_sentences: List[str] = []
            if para_idx >= 0 and def_index:
                symbols = _extract_symbols_from_text(paragraph)
                definition_sentences = _definitions_for_symbols(
                    def_index, symbols, para_idx
                )
            mentions.append(
                {
                    "arxiv_id": arxiv_id,
                    "context_prev": prev_line,
                    "context_sentence": line.strip(),
                    "context_next": next_line,
                    "context_paragraph": paragraph,
                    "definition_sentences": definition_sentences,
                    "explicit_refs": [
                        {
                            "kind": kind,
                            "number": number,
                            "raw": f"{kind.title()} {number}",
                        }
                    ],
                    "target_arxiv_id": arxiv_id,
                    "source": "internal",
                }
            )
    return mentions


def _convert_queries_to_mentions(
    queries_path: Path, out_path: Path, target_arxiv_id: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in read_jsonl(str(queries_path)):
            query_text = row.get("query_text") or ""
            explicit_refs = row.get("explicit_refs") or []
            if not query_text or not explicit_refs:
                continue
            f.write(
                json.dumps(
                    {
                        "openalex_id": None,
                        "arxiv_id": None,
                        "context_sentence": query_text,
                        "context_prev": None,
                        "context_next": None,
                        "context_paragraph": query_text,
                        "cite_label": None,
                        "location_type": "query_text",
                        "explicit_refs": explicit_refs,
                        "reference_precision": "explicit",
                        "target_arxiv_id": target_arxiv_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _build_mentions_dataset(
    *,
    mentions_paths: List[Path],
    statement_index: Dict[Tuple[str, str], List[Dict]],
    out_path: Path,
    queries_out_path: Path,
    qrels_out_path: Path,
    strip_explicit_refs: bool = False,
    infer_explicit_refs: bool = False,
    external_only: bool = False,
    mask_citations: bool = False,
    prepend_definitions: bool = False,
    def_cap_sentences: int = 5,
    def_cap_words: int = 250,
    section_prefix: bool = False,
    drop_paragraph_refs: bool = True,
    context_mode: str = "tri_sentence",
    context_window: int = 2,
    context_max_words: int = 0,
    cache_dir: Optional[Path] = None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    queries_out_path.parent.mkdir(parents=True, exist_ok=True)
    qrels: Dict[str, List[str]] = {}
    total = 0
    def_cache = None
    if prepend_definitions and cache_dir:
        def_cache = DefinitionIndexCache(cache_dir)
    with out_path.open("w", encoding="utf-8") as df, queries_out_path.open(
        "w", encoding="utf-8"
    ) as qf:
        for mp in mentions_paths:
            for row in read_jsonl(str(mp)):
                explicit_refs = row.get("explicit_refs") or []
                ref_precision = row.get("reference_precision") or (
                    "explicit" if explicit_refs else None
                )
                if not explicit_refs:
                    if infer_explicit_refs:
                        refs_text = _refs_text_for_inference(row, context_mode)
                        explicit_refs = _extract_explicit_refs_from_text(refs_text)
                        if explicit_refs:
                            ref_precision = "heuristic"
                    if not explicit_refs:
                        continue
                if (
                    drop_paragraph_refs
                    and row.get("explicit_ref_source") == "paragraph"
                ):
                    logger.debug(
                        "Skipping row with paragraph-level explicit refs for arXiv {}",
                        row.get("arxiv_id"),
                    )
                    continue
                arxiv_id = row.get("target_arxiv_id")
                if not arxiv_id:
                    continue
                if external_only:
                    source_id = row.get("arxiv_id")
                    if not source_id or source_id == arxiv_id:
                        continue
                    if (row.get("source") or "").lower() == "internal":
                        continue
                context = _select_context(
                    row,
                    context_mode=context_mode,
                    context_window=context_window,
                    context_max_words=0,
                )
                if section_prefix:
                    section = (row.get("section_title") or "").strip()
                    if section:
                        context = f"SECTION: {section} {context}".strip()
                if prepend_definitions:
                    defs = row.get("definition_sentences") or []
                    if isinstance(defs, str):
                        defs = [defs]
                    if not defs and def_cache:
                        para_text = (
                            row.get("context_paragraph")
                            or _html_to_text(row.get("context_html") or "")
                            or row.get("context_sentence")
                            or ""
                        )
                        defs = def_cache.definitions_for(
                            row.get("arxiv_id") or "", para_text
                        )
                    defs_text = _cap_definitions(defs, def_cap_sentences, def_cap_words)
                    if defs_text:
                        context = f"{defs_text} {context}".strip()
                if mask_citations:
                    context = _mask_citations(context)
                if strip_explicit_refs:
                    context = _strip_explicit_refs(context)
                if context_max_words > 0:
                    context = _truncate_words(context, context_max_words)
                if not context:
                    continue
                kind = (explicit_refs[0].get("kind") or "").strip().lower()
                if kind and _type_conflict(context, kind):
                    logger.debug(
                        "Skipping query due to type mismatch (kind={}, arXiv={})",
                        kind,
                        row.get("arxiv_id"),
                    )
                    continue
                base_mention_id = sha256_hash(
                    "|".join(
                        [
                            row.get("openalex_id") or "",
                            row.get("arxiv_id") or "",
                            row.get("context_sentence") or "",
                            row.get("cite_label") or "",
                        ]
                    )
                )
                for ref in explicit_refs:
                    kind = (ref.get("kind") or "").lower().strip(".")
                    number = (ref.get("number") or "").strip()
                    if not kind or not number:
                        continue
                    pref_number = _prefix_label(arxiv_id, number)
                    matches = statement_index.get((kind, pref_number)) or []
                    if not matches:
                        continue
                    target_node = matches[0]
                    statement_id = target_node.get("id")
                    statement_text = _statement_text(target_node)
                    if not statement_id or not statement_text:
                        continue
                    mention_id = sha256_hash(f"{base_mention_id}:{kind}:{pref_number}")
                    query_id = sha256_hash(f"{mention_id}:{statement_id}")
                    df.write(
                        json.dumps(
                            {
                                "mention_id": mention_id,
                                "query_id": query_id,
                                "query_text": context,
                                "source_arxiv_id": row.get("arxiv_id"),
                                "target_arxiv_id": arxiv_id,
                                "target_statement_id": statement_id,
                                "target_statement_text": statement_text,
                                "kind": kind,
                                "number": pref_number,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    qf.write(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "query_text": context,
                                "query_style": (
                                    "claim" if strip_explicit_refs else "mention"
                                ),
                                "source_arxiv_id": row.get("arxiv_id"),
                                "target_arxiv_id": arxiv_id,
                                "explicit_refs": (
                                    []
                                    if strip_explicit_refs
                                    else [{"kind": kind, "number": pref_number}]
                                ),
                                "reference_precision": (
                                    "implicit"
                                    if strip_explicit_refs
                                    else (ref_precision or "explicit")
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    qrels.setdefault(query_id, []).append(statement_id)
                    total += 1
    qrels_out_path.write_text(json.dumps(qrels, indent=2))
    logger.info("Wrote {} mention pairs to {}", total, out_path)
    return total


def _split_by_target(
    dataset_path: Path,
    out_dir: Path,
    *,
    val_target: str,
    test_target: str,
    strip_explicit_refs: bool = False,
) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for row in read_jsonl(str(dataset_path)):
        tgt = row.get("target_arxiv_id")
        if tgt == val_target:
            splits["val"].append(row)
        elif tgt == test_target:
            splits["test"].append(row)
        else:
            splits["train"].append(row)

    counts = {}
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
        qrels = {}
        for r in rows:
            qid = r.get("query_id")
            sid = r.get("target_statement_id")
            if qid and sid:
                qrels.setdefault(qid, []).append(sid)
        (out_dir / f"{name}_qrels.json").write_text(json.dumps(qrels, indent=2))
        # Write queries for retrieval_benchmark compatibility.
        qpath = out_dir / f"{name}_queries.jsonl"
        with qpath.open("w", encoding="utf-8") as qf:
            for r in rows:
                qid = r.get("query_id")
                text = r.get("query_text")
                kind = r.get("kind")
                number = r.get("number")
                if not qid or not text or not kind or not number:
                    continue
                qf.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "query_text": text,
                            "query_style": (
                                "claim" if strip_explicit_refs else "mention"
                            ),
                            "target_arxiv_id": r.get("target_arxiv_id"),
                            "explicit_refs": (
                                []
                                if strip_explicit_refs
                                else [{"kind": kind, "number": number}]
                            ),
                            "reference_precision": (
                                "implicit" if strip_explicit_refs else "explicit"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    logger.info("Split counts: {}", counts)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mention-supervised dataset.")
    parser.add_argument("--targets-json", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--out-dir", default="data/mentions")
    parser.add_argument("--statements-dir", default="data/statements/mentions")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--max-works", type=int, default=0)
    parser.add_argument("--rate-limit", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-statements", action="store_true")
    parser.add_argument("--skip-mentions", action="store_true")
    parser.add_argument(
        "--no-local-queries",
        action="store_true",
        help="Skip using local query JSONL for targets with local_statements.",
    )
    parser.add_argument(
        "--strip-explicit-refs",
        action="store_true",
        help="Remove explicit reference cues from query text.",
    )
    parser.add_argument(
        "--infer-explicit-refs",
        action="store_true",
        help="Heuristically extract explicit refs from context when missing.",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Keep only mentions where source_arxiv_id != target_arxiv_id.",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include internal (within-paper) references as mention pairs.",
    )
    parser.add_argument(
        "--mask-citations",
        action="store_true",
        help="Replace citation markers like \\\\cite{...} or [12] with CITATION.",
    )
    parser.add_argument(
        "--prepend-definitions",
        action="store_true",
        help="Prepend symbol definition sentences inferred from source document.",
    )
    parser.add_argument(
        "--def-cap-sentences",
        type=int,
        default=5,
        help="Max definition sentences to prepend (0 = no cap).",
    )
    parser.add_argument(
        "--def-cap-words",
        type=int,
        default=250,
        help="Max words allowed in prepended definitions (0 = no cap).",
    )
    parser.add_argument(
        "--section-prefix",
        action="store_true",
        help="Prefix query text with SECTION: <title> when available.",
    )
    parser.add_argument(
        "--context-mode",
        default="tri_sentence",
        choices=["tri_sentence", "sentence", "paragraph", "window"],
        help="How to build query context from mention text.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Sentence window on each side (only for context-mode=window).",
    )
    parser.add_argument(
        "--context-max-words",
        type=int,
        default=0,
        help="Max words to keep in query text (0 = no cap).",
    )
    parser.add_argument(
        "--keep-paragraph-refs",
        action="store_true",
        help="Keep explicit refs extracted only from full paragraphs.",
    )
    parser.add_argument("--fallback-arxiv", action="store_true")
    parser.add_argument("--val-target", default="1303.5113")
    parser.add_argument("--test-target", default="1709.10033")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    statements_dir = Path(args.statements_dir)
    cache_dir = Path(args.cache_dir)
    ensure_dir(str(out_dir))
    ensure_dir(str(cache_dir))

    resolver = OpenAlexTargetResolver(
        cache_dir=str(cache_dir), mailto=args.mailto, api_key=args.api_key
    )
    targets = _load_targets(args.targets_json, args.targets)

    # Resolve missing titles/authors only when we need OpenAlex mentions.
    if not args.skip_mentions:
        api = ArxivAPI()
        for t in targets:
            if t.local_statements:
                continue
            if not t.title or t.authors is None:
                meta = resolver.fetch_arxiv_metadata(t.arxiv_id, api)
                if not t.title:
                    t.title = meta.title
                if t.authors is None:
                    t.authors = meta.authors

    if not args.skip_statements:
        statements_dir.mkdir(parents=True, exist_ok=True)
        for t in targets:
            if t.local_statements:
                continue
            statements_path = statements_dir / f"{t.arxiv_id.replace('/', '_')}.json"
            if statements_path.exists():
                continue
            asyncio.run(
                _build_statements(t.arxiv_id, statements_path, t.local_source_dir)
            )

    if not args.skip_mentions:
        for t in targets:
            if t.local_statements:
                # Use existing perfectoid queries as mention surrogates.
                if not args.no_local_queries:
                    queries_path = Path(
                        "data/citation_dataset/perfectoid_queries.jsonl"
                    )
                    if queries_path.exists():
                        target_id = resolver.derive_target_id(t.arxiv_id)
                        out_path = out_dir / f"{target_id}_mentions.jsonl"
                        _convert_queries_to_mentions(queries_path, out_path, t.arxiv_id)
                    continue
            target_id = resolver.derive_target_id(t.arxiv_id)
            openalex_id = t.openalex_id or resolver.resolve_openalex_work_id(
                title=t.title or "", authors=t.authors or []
            )
            if not openalex_id:
                raise RuntimeError(f"Unable to resolve OpenAlex ID for {t.arxiv_id}")
            stage1 = OpenAlexCitingWorksStage(
                target_ids=[openalex_id],
                target_id=target_id,
                out_dir=str(out_dir),
                cache_dir=str(cache_dir),
                mailto=args.mailto,
                api_key=args.api_key,
                per_page=args.per_page,
                max_works=args.max_works,
                rate_limit=args.rate_limit,
                fallback_arxiv=bool(args.fallback_arxiv),
                fallback_cache_db=str(cache_dir / "arxiv_fallback_cache.db"),
                fallback_refresh_days=30,
            )
            stage1.run()
            works_path = out_dir / f"{target_id}_works.jsonl"
            if not works_path.exists():
                works_path.write_text("", encoding="utf-8")
            if works_path.stat().st_size == 0:
                # No citing works; still write an empty mentions file for consistency.
                (out_dir / f"{target_id}_mentions.jsonl").write_text(
                    "", encoding="utf-8"
                )
                continue
            stage2 = MentionContextExtractionStage(
                works_file=str(works_path),
                target_title=t.title or "",
                target_id=target_id,
                out_dir=str(out_dir),
                cache_dir=str(cache_dir),
                rate_limit=args.rate_limit,
                max_works=args.max_works,
                no_pdf=False,
                concurrency=args.concurrency,
                offline=False,
            )
            asyncio.run(stage2.run())

    # Merge statements + build dataset
    statement_paths = []
    for t in targets:
        if t.local_statements:
            statement_paths.append(Path(t.local_statements))
        else:
            statement_paths.append(
                statements_dir / f"{t.arxiv_id.replace('/', '_')}.json"
            )
    combined_statements_path = out_dir / "combined_statements.json"
    source_map = {t.arxiv_id: t.local_source_dir for t in targets}
    merged = _merge_statements(statement_paths, combined_statements_path, source_map)
    statement_index = _build_statement_index(merged)
    _write_statement_corpus(merged, out_dir / "statements.jsonl")

    mentions_paths = []
    for t in targets:
        target_id = resolver.derive_target_id(t.arxiv_id)
        path = out_dir / f"{target_id}_mentions.jsonl"
        if path.exists():
            # annotate target arxiv id to each row for downstream mapping
            annotated = out_dir / f"{target_id}_mentions_annotated.jsonl"
            with annotated.open("w", encoding="utf-8") as f:
                for row in read_jsonl(str(path)):
                    row["target_arxiv_id"] = t.arxiv_id
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            mentions_paths.append(annotated)
        if args.include_internal and t.local_source_dir:
            nodes_for_target = [
                n for n in merged.get("nodes") or [] if n.get("arxiv_id") == t.arxiv_id
            ]
            internal_rows = _extract_internal_mentions(
                t.arxiv_id, t.local_source_dir, nodes_for_target
            )
            if internal_rows:
                internal_path = out_dir / f"{target_id}_internal_mentions.jsonl"
                with internal_path.open("w", encoding="utf-8") as f:
                    for row in internal_rows:
                        row["target_arxiv_id"] = t.arxiv_id
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                mentions_paths.append(internal_path)

    dataset_path = out_dir / "mentions_dataset.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.json"
    _build_mentions_dataset(
        mentions_paths=mentions_paths,
        statement_index=statement_index,
        out_path=dataset_path,
        queries_out_path=queries_path,
        qrels_out_path=qrels_path,
        strip_explicit_refs=args.strip_explicit_refs,
        infer_explicit_refs=args.infer_explicit_refs,
        external_only=args.external_only,
        mask_citations=args.mask_citations,
        prepend_definitions=args.prepend_definitions,
        def_cap_sentences=args.def_cap_sentences,
        def_cap_words=args.def_cap_words,
        section_prefix=args.section_prefix,
        drop_paragraph_refs=not args.keep_paragraph_refs,
        context_mode=args.context_mode,
        context_window=args.context_window,
        context_max_words=args.context_max_words,
        cache_dir=cache_dir,
    )

    split_dir = out_dir / "splits"
    _split_by_target(
        dataset_path,
        split_dir,
        val_target=args.val_target,
        test_target=args.test_target,
        strip_explicit_refs=args.strip_explicit_refs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
