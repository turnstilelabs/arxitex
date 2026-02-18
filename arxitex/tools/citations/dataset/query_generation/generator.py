from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from arxitex.llms.llms import aexecute_prompt
from arxitex.tools.citations.dataset.query_generation.models import MentionContext
from arxitex.tools.citations.dataset.query_generation.prompt import QueryPromptGenerator
from arxitex.tools.citations.dataset.utils import (
    append_jsonl,
    extract_named,
    extract_refs,
    sha256_hash,
)


class QuerySingle(BaseModel):
    query_text: str = Field(..., min_length=4)


def _extract_source_refs(row: Dict[str, Any]) -> Dict[str, Any]:
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
                    styles = ["precise", "vague"]
                    cleaned: List[tuple[str, QuerySingle]] = []
                    for style in styles:
                        prompt_id = f"synth-query-{style}-{_make_mention_id(ctx)}"
                        prompt = self.prompt_generator.make_prompt(
                            ctx,
                            style=style,
                            target_name=self.target_name,
                            prompt_id=prompt_id,
                        )
                        if self.temperature is None:
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
                                temperature=self.temperature,
                            )
                        if not result or not getattr(result, "query_text", None):
                            logger.warning(
                                "No {} query for arXiv {}", style, row.get("arxiv_id")
                            )
                            continue
                        text = result.query_text.strip()
                        if not text:
                            logger.warning(
                                "Empty {} query for arXiv {}",
                                style,
                                row.get("arxiv_id"),
                            )
                            continue
                        cleaned.append((style, result))

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
