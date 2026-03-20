"""Target acquisition/extraction helpers for the mentions dataset pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from arxitex.arxiv_api import ArxivAPI
from arxitex.extractor.pipeline import agenerate_artifact_graph as agenerate_statements
from arxitex.tools.mentions.acquisition.openalex_citations import (
    OpenAlexCitingWorksStage,
)
from arxitex.tools.mentions.acquisition.target_resolution import (
    OpenAlexTargetResolver,
    TargetWorkProfile,
)
from arxitex.tools.mentions.extraction.extract_mentions_cli import (
    MentionContextExtractionStage,
)
from arxitex.utils import read_jsonl


@dataclass
class Target:
    arxiv_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    local_statements: Optional[str] = None
    openalex_id: Optional[str] = None
    local_source_dir: Optional[str] = None


DEFAULT_TARGETS = [
    Target(
        "perfectoid",
        title="Perfectoid Spaces",
        authors=["Peter Scholze"],
        openalex_id="https://openalex.org/W4255501032",
        local_statements="data/statements/perfectoid.json",
    ),
    Target("math/0608640"),
    Target("1709.10033"),
    Target("1303.5113"),
]


def derive_target_id(arxiv_id: str) -> str:
    safe = (arxiv_id or "").replace("/", "_").strip()
    if not safe:
        raise ValueError("arXiv id is required")
    return f"arxiv_{safe}"


def load_targets(path: Optional[str], arxiv_ids: Optional[List[str]]) -> List[Target]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Target(**row) for row in data]
    if arxiv_ids:
        return [Target(arxiv_id=a) for a in arxiv_ids]
    return DEFAULT_TARGETS


async def _build_statements(
    arxiv_id: str,
    out_path: Path,
    local_source_dir: Optional[str],
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
    payload = {
        "arxiv_id": arxiv_id,
        "extractor_mode": "statements-only",
        "stats": {"nodes": len(nodes)},
        "nodes": nodes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _resolve_target_profile(
    resolver: OpenAlexTargetResolver,
    target: Target,
    openalex_id: str,
) -> TargetWorkProfile:
    profile = TargetWorkProfile(title=target.title or "")
    try:
        work = resolver.fetch_openalex_work(openalex_id)
        if work:
            profile = TargetWorkProfile(
                title=(work.get("title") or target.title or ""),
                doi=(work.get("doi") or None),
                year=work.get("publication_year"),
            )
    except Exception as exc:
        logger.warning(
            "Could not enrich target profile for {}: {}",
            target.arxiv_id,
            exc,
        )
    return profile


def ensure_target_metadata(
    *,
    targets: List[Target],
    resolver: OpenAlexTargetResolver,
) -> None:
    api = ArxivAPI()
    for target in targets:
        if target.local_statements:
            continue
        if not target.title or target.authors is None:
            meta = resolver.fetch_arxiv_metadata(target.arxiv_id, api)
            if not target.title:
                target.title = meta.title
            if target.authors is None:
                target.authors = meta.authors


def _run_mentions_for_target(
    *,
    target: Target,
    resolver: OpenAlexTargetResolver,
    out_dir: Path,
    cache_dir: Path,
    args: argparse.Namespace,
) -> None:
    target_id = derive_target_id(target.arxiv_id)
    mentions_path = out_dir / f"{target_id}_mentions.jsonl"
    if mentions_path.exists() and mentions_path.stat().st_size > 0:
        logger.info("Reusing mentions file {}", mentions_path)
        return

    openalex_id = target.openalex_id or resolver.resolve_openalex_work_id(
        title=target.title or "",
        authors=target.authors or [],
    )
    if not openalex_id:
        raise RuntimeError(f"Unable to resolve OpenAlex ID for {target.arxiv_id}")

    target_profile = _resolve_target_profile(resolver, target, openalex_id)

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
        mentions_path.write_text("", encoding="utf-8")
        return

    stage2 = MentionContextExtractionStage(
        works_file=str(works_path),
        target_title=target_profile.title or target.title or "",
        target_profile=target_profile,
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


def ensure_statements_for_targets(
    *,
    targets: List[Target],
    statements_dir: Path,
) -> None:
    statements_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        if target.local_statements:
            continue
        statements_path = statements_dir / f"{target.arxiv_id.replace('/', '_')}.json"
        if statements_path.exists():
            continue
        asyncio.run(
            _build_statements(
                target.arxiv_id,
                statements_path,
                target.local_source_dir,
            )
        )


def extract_mentions_for_targets(
    *,
    targets: List[Target],
    resolver: OpenAlexTargetResolver,
    out_dir: Path,
    cache_dir: Path,
    args: argparse.Namespace,
) -> None:
    for target in targets:
        _run_mentions_for_target(
            target=target,
            resolver=resolver,
            out_dir=out_dir,
            cache_dir=cache_dir,
            args=args,
        )


def prepare_statement_paths(
    targets: List[Target],
    statements_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    for target in targets:
        if target.local_statements:
            paths.append(Path(target.local_statements))
        else:
            paths.append(statements_dir / f"{target.arxiv_id.replace('/', '_')}.json")
    return paths


def collect_mentions_rows(
    *,
    targets: List[Target],
    out_dir: Path,
) -> List[Dict]:
    rows: List[Dict] = []
    for target in targets:
        target_id = derive_target_id(target.arxiv_id)
        path = out_dir / f"{target_id}_mentions.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(str(path)):
            row["target_arxiv_id"] = target.arxiv_id
            rows.append(row)
    return rows
