from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from arxitex.llms.llms import aexecute_prompt
from arxitex.tools.mentions.generation.filters import (
    has_location_terms,
    is_leaky,
    type_conflict,
)
from arxitex.tools.mentions.generation.models import MentionContext
from arxitex.tools.mentions.generation.prompt import QueryPromptGenerator
from arxitex.tools.mentions.utils import extract_named, extract_refs
from arxitex.utils import append_jsonl, sha256_hash


class QuerySingle(BaseModel):
    query_text: str = Field(..., min_length=4)


MAX_QUERY_WORDS = 30
MAX_QUERY_ATTEMPTS = 3


def _extract_source_refs(
    row: Dict[str, Any], *, max_chars: int = 2000
) -> Dict[str, Any]:
    context = " ".join(
        [
            row.get("context_sentence") or "",
            row.get("context_prev") or "",
            row.get("context_next") or "",
            row.get("context_paragraph") or "",
            row.get("context_html") or "",
            row.get("section_title") or "",
        ]
    )
    if max_chars > 0 and len(context) > max_chars:
        context = context[:max_chars]
    refs = extract_refs(context)
    named = extract_named(context)
    return {"source_refs": refs, "source_named_refs": named, "source_ref_text": context}


def _make_mention_id(ctx: MentionContext) -> str:
    parts = [
        str(ctx.openalex_id or ""),
        str(ctx.arxiv_id or ""),
        str(ctx.context_sentence or ""),
        str(ctx.cite_label or ""),
        str(ctx.location_type or ""),
    ]
    return sha256_hash("|".join(parts))


class QueryGenerator:
    def __init__(
        self,
        *,
        model: str,
        target_name: str,
        temperature: Optional[float] = None,
        concurrency: int = 4,
    ) -> None:
        self.model = model
        self.target_name = target_name
        self.temperature = temperature
        self.concurrency = max(1, int(concurrency))
        self.prompt_generator = QueryPromptGenerator()

    def _too_long(self, text: str) -> bool:
        if not text:
            return True
        return len(text.split()) > MAX_QUERY_WORDS

    async def generate_from_mentions(
        self,
        mentions: List[Dict[str, Any]],
        out_path: str,
    ) -> Dict[str, int]:
        sem = asyncio.Semaphore(self.concurrency)
        write_lock = asyncio.Lock()
        counters_lock = asyncio.Lock()
        counters = {"processed": 0, "failed": 0, "queries": 0}

        async def process_row(row: Dict[str, Any]) -> None:
            async with sem:
                try:
                    ctx = MentionContext.from_row(row)
                    explicit_kind = None
                    if ctx.explicit_refs:
                        first = ctx.explicit_refs[0] or {}
                        explicit_kind = (
                            first.get("kind") or ""
                        ).strip().lower() or None
                    styles = ["precise", "vague"]
                    cleaned: List[tuple[str, QuerySingle]] = []
                    for style in styles:
                        accepted = None
                        for attempt in range(1, MAX_QUERY_ATTEMPTS + 1):
                            prompt_id = (
                                f"synth-query-{style}-{attempt}-{_make_mention_id(ctx)}"
                            )
                            prompt = self.prompt_generator.make_prompt(
                                ctx,
                                style=style,
                                target_name=self.target_name,
                                prompt_id=prompt_id,
                            )
                            temp = self.temperature
                            if temp is None and attempt > 1:
                                temp = 0.2
                            if temp is None:
                                result = await aexecute_prompt(
                                    prompt,
                                    QuerySingle,
                                    model=self.model,
                                )
                            else:
                                result = await aexecute_prompt(
                                    prompt,
                                    QuerySingle,
                                    model=self.model,
                                    temperature=temp,
                                )
                            if not result or not getattr(result, "query_text", None):
                                logger.warning(
                                    "No {} query for arXiv {} (attempt {})",
                                    style,
                                    row.get("arxiv_id"),
                                    attempt,
                                )
                                continue
                            text = result.query_text.strip()
                            if not text:
                                logger.warning(
                                    "Empty {} query for arXiv {} (attempt {})",
                                    style,
                                    row.get("arxiv_id"),
                                    attempt,
                                )
                                continue
                            if self._too_long(text):
                                logger.warning(
                                    "Rejected {} query (too long) for arXiv {} (attempt {})",
                                    style,
                                    row.get("arxiv_id"),
                                    attempt,
                                )
                                continue
                            if is_leaky(text, self.target_name):
                                logger.warning(
                                    "Rejected {} query (leaky) for arXiv {} (attempt {})",
                                    style,
                                    row.get("arxiv_id"),
                                    attempt,
                                )
                                continue
                            if has_location_terms(text):
                                logger.warning(
                                    "Rejected {} query (location wording) for arXiv {} (attempt {})",
                                    style,
                                    row.get("arxiv_id"),
                                    attempt,
                                )
                                continue
                            if type_conflict(text, explicit_kind):
                                logger.warning(
                                    "Rejected {} query (type mismatch) for arXiv {} (attempt {})",
                                    style,
                                    row.get("arxiv_id"),
                                    attempt,
                                )
                                continue
                            accepted = result
                            break
                        if accepted:
                            cleaned.append((style, accepted))

                    if not cleaned:
                        async with counters_lock:
                            counters["processed"] += 1
                            counters["failed"] += 1
                        return

                    mention_id = _make_mention_id(ctx)
                    now = datetime.now(timezone.utc).isoformat()
                    source_ref_payload = _extract_source_refs(row)
                    stitched_context = " ".join(
                        [
                            s
                            for s in [
                                row.get("context_prev") or "",
                                row.get("context_sentence") or "",
                                row.get("context_next") or "",
                            ]
                            if s
                        ]
                    ).strip()

                    async with write_lock:
                        for idx, (style, q) in enumerate(cleaned):
                            query_id = sha256_hash(
                                f"{mention_id}:{style}:{q.query_text}"
                            )
                            append_jsonl(
                                out_path,
                                {
                                    "query_id": query_id,
                                    "query_text": q.query_text,
                                    "query_style": style,
                                    "source_arxiv_id": row.get("arxiv_id"),
                                    "source_openalex_id": row.get("openalex_id"),
                                    "mention_id": mention_id,
                                    "location_type": row.get("location_type"),
                                    "reference_precision": row.get(
                                        "reference_precision"
                                    ),
                                    "section_title": row.get("section_title"),
                                    "cite_label": row.get("cite_label"),
                                    "bib_entry": row.get("bib_entry"),
                                    "explicit_refs": row.get("explicit_refs") or [],
                                    "explicit_ref_source": row.get(
                                        "explicit_ref_source"
                                    ),
                                    "explicit_ref_kind": explicit_kind,
                                    "context_sentence": row.get("context_sentence"),
                                    "context_prev": row.get("context_prev"),
                                    "context_next": row.get("context_next"),
                                    "context_paragraph": stitched_context
                                    or row.get("context_paragraph")
                                    or row.get("context_html"),
                                    "context_html": row.get("context_html"),
                                    "model_name": self.model,
                                    "generated_at": now,
                                    "query_variant_index": idx,
                                    **source_ref_payload,
                                },
                            )

                    async with counters_lock:
                        counters["processed"] += 1
                        counters["queries"] += len(cleaned)
                        processed = counters["processed"]
                        if processed == 1 or processed % 10 == 0:
                            logger.info(
                                "Processed {} / {} mentions", processed, len(mentions)
                            )
                except Exception as e:
                    logger.error("Failed mention {}: {}", row.get("arxiv_id"), e)
                    async with counters_lock:
                        counters["processed"] += 1
                        counters["failed"] += 1
                        processed = counters["processed"]
                        if processed == 1 or processed % 10 == 0:
                            logger.info(
                                "Processed {} / {} mentions", processed, len(mentions)
                            )

        tasks = [asyncio.create_task(process_row(row)) for row in mentions]
        await asyncio.gather(*tasks)
        return counters
