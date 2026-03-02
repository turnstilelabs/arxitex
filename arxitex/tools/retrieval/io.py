"""Load graph + query datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loguru import logger

from .normalization import normalize_text

TYPE_TAGS = {
    "definition": "[DEF]",
    "theorem": "[THM]",
    "lemma": "[LEM]",
    "proposition": "[PROP]",
    "corollary": "[COR]",
    "remark": "[REM]",
    "example": "[EX]",
    "proof": "[PROOF]",
}


@dataclass
class Artifact:
    artifact_id: str
    node_id: str
    type: str
    text: str
    normalized_text: str
    index_text: str


@dataclass
class GraphData:
    nodes: Dict[str, Dict]
    edges: List[Dict]


def load_graph(path: str) -> GraphData:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    graph = payload.get("graph", payload)
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    logger.info("Loaded graph: {} nodes, {} edges", len(nodes), len(edges))
    return GraphData(nodes=nodes, edges=edges)


def _tag_for_type(node_type: str) -> str:
    return TYPE_TAGS.get(node_type, f"[{node_type.upper()}]")


def _flatten_prereqs(node: Dict) -> str:
    prereq_defs = node.get("prerequisite_defs") or {}
    if prereq_defs:
        parts = [f"{term}: {definition}" for term, definition in prereq_defs.items()]
        return " ".join(part for part in parts if part)
    preview = node.get("prerequisites_preview") or ""
    if not preview:
        return ""
    text = preview.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    text = re.sub(r"</?b>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _maybe_normalize(text: str, normalize_mode: str) -> str:
    if not text:
        return ""
    if normalize_mode == "none":
        return text
    if normalize_mode == "unicode":
        return normalize_text(text, use_math_verify=False) or text
    # default: auto (unicode + math_verify when available)
    return normalize_text(text, use_math_verify=True) or text


def _build_index_text(
    *,
    node: Dict,
    node_type: str,
    base_text: str,
    index_mode: str,
    include_type_prefix: bool,
    normalize_mode: str,
    structured: Dict | None = None,
    structured_math_verify: bool = False,
) -> str:
    parts: List[str] = []
    if include_type_prefix:
        parts.append(_tag_for_type(node_type))

    include_prereqs = index_mode in ("content+prereqs", "content+all")
    include_semantic = index_mode in ("content+semantic", "content+all")

    if structured:
        terms = structured.get("math_terms") or []
        exprs = structured.get("math_exprs") or []
        domain = structured.get("domain_terms") or []
        if terms:
            parts.append("[TERMS] " + " ; ".join(terms))
        if exprs:
            expr_text = " ; ".join(exprs)
            if structured_math_verify:
                expr_text = _maybe_normalize(expr_text, "auto")
            parts.append("[EXPRS] " + expr_text)
        if domain:
            parts.append("[DOMAIN] " + " ; ".join(domain))

    if include_semantic and node.get("semantic_tag"):
        parts.append(node.get("semantic_tag") or "")
    if include_prereqs:
        prereq_text = _flatten_prereqs(node)
        if prereq_text:
            parts.append(prereq_text)

    parts.append(base_text)
    joined = " ".join(p for p in parts if p)
    return _maybe_normalize(joined, normalize_mode)


def build_artifacts(
    graph: GraphData,
    include_proofs: bool = True,
    *,
    index_mode: str = "content",
    include_type_prefix: bool = True,
    normalize_mode: str = "auto",
    structured_map: Dict[str, Dict] | None = None,
    structured_math_verify: bool = False,
) -> List[Artifact]:
    artifacts: List[Artifact] = []
    for node_id, node in graph.nodes.items():
        node_type = node.get("type") or "unknown"
        content = node.get("content") or ""
        if content.strip():
            norm_content = _maybe_normalize(content, normalize_mode)
            index_text = _build_index_text(
                node=node,
                node_type=node_type,
                base_text=content,
                index_mode=index_mode,
                include_type_prefix=include_type_prefix,
                normalize_mode=normalize_mode,
                structured=(structured_map or {}).get(node_id),
                structured_math_verify=structured_math_verify,
            )
            artifacts.append(
                Artifact(
                    artifact_id=node_id,
                    node_id=node_id,
                    type=node_type,
                    text=content,
                    normalized_text=norm_content,
                    index_text=index_text,
                )
            )
        proof = node.get("proof") or ""
        if include_proofs and proof.strip():
            norm_proof = _maybe_normalize(proof, normalize_mode)
            index_text = _build_index_text(
                node=node,
                node_type="proof",
                base_text=proof,
                index_mode=index_mode,
                include_type_prefix=include_type_prefix,
                normalize_mode=normalize_mode,
                structured=(structured_map or {}).get(f"{node_id}#proof"),
                structured_math_verify=structured_math_verify,
            )
            artifacts.append(
                Artifact(
                    artifact_id=f"{node_id}#proof",
                    node_id=node_id,
                    type="proof",
                    text=proof,
                    normalized_text=norm_proof,
                    index_text=index_text,
                )
            )
    logger.info(
        "Built {} artifacts (proofs included: {})", len(artifacts), include_proofs
    )
    return artifacts


def load_queries(path: str) -> List[Dict]:
    queries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    logger.info("Loaded {} queries", len(queries))
    return queries


def load_qrels(path: str) -> Dict[str, List[str]]:
    qrels: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("query_id")
            rel = row.get("relevant_ids") or []
            if not qid:
                continue
            qrels[qid] = rel
    logger.info("Loaded {} qrels", len(qrels))
    return qrels
