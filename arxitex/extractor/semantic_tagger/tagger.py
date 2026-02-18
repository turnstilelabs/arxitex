#!/usr/bin/env python3
"""Generate semantic tags for artifacts using an LLM."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from arxitex.extractor.models import ArtifactNode
from arxitex.extractor.semantic_tagger.models import SemanticTag
from arxitex.extractor.semantic_tagger.prompt import SemanticTagPromptGenerator
from arxitex.llms.llms import aexecute_prompt
from arxitex.llms.usage_context import llm_usage_stage
from arxitex.tools.citations.dataset.utils import append_jsonl, ensure_dir, read_jsonl

MAX_CONTEXT_CHARS = 2000


def _truncate(text: str, max_len: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class SemanticTagger:
    def __init__(self, model: str, concurrency: int) -> None:
        self.model = model
        self.concurrency = max(1, concurrency)

    async def _tag_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prompt_id = f"semantic-tag-{digest}"
        prompt = SemanticTagPromptGenerator().make_prompt(_truncate(text), prompt_id)
        with llm_usage_stage("semantic_tagging"):
            result = await aexecute_prompt(prompt, SemanticTag, model=self.model)
        return (result.semantic_tag or "").strip() if result else ""

    async def tag_artifacts(
        self,
        rows: List[Dict[str, Any]],
        out_path: str,
    ) -> Dict[str, int]:
        sem = asyncio.Semaphore(self.concurrency)
        write_lock = asyncio.Lock()
        counters = {"processed": 0, "failed": 0, "tagged": 0}

        async def process_row(row: Dict[str, Any]) -> None:
            async with sem:
                try:
                    tag = await self._tag_text(row.get("text") or "")
                    enriched = dict(row)
                    enriched["semantic_tag"] = tag
                    async with write_lock:
                        append_jsonl(out_path, enriched)
                    counters["tagged"] += 1
                except Exception as exc:
                    logger.error("Failed artifact {}: {}", row.get("artifact_id"), exc)
                    counters["failed"] += 1
                finally:
                    counters["processed"] += 1
                    if counters["processed"] == 1 or counters["processed"] % 25 == 0:
                        logger.info(
                            "Processed {} / {}", counters["processed"], len(rows)
                        )

        tasks = [asyncio.create_task(process_row(r)) for r in rows]
        await asyncio.gather(*tasks)
        return counters

    async def tag_nodes(
        self,
        nodes: List[ArtifactNode],
        include_external: bool = False,
    ) -> Dict[str, int]:
        sem = asyncio.Semaphore(self.concurrency)
        counters = {"processed": 0, "failed": 0, "tagged": 0}
        targets = [n for n in nodes if include_external or not n.is_external]

        async def process_node(node: ArtifactNode) -> None:
            async with sem:
                try:
                    node.semantic_tag = await self._tag_text(node.content or "")
                    counters["tagged"] += 1
                except Exception as exc:
                    logger.error("Failed semantic tag for {}: {}", node.id, exc)
                    counters["failed"] += 1
                finally:
                    counters["processed"] += 1
                    if counters["processed"] == 1 or counters["processed"] % 25 == 0:
                        logger.info(
                            "Tagged {} / {}", counters["processed"], len(targets)
                        )

        tasks = [asyncio.create_task(process_node(n)) for n in targets]
        await asyncio.gather(*tasks)
        return counters


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate semantic tags for artifacts."
    )
    parser.add_argument("--artifacts", required=True, help="Input artifacts JSONL.")
    parser.add_argument(
        "--out",
        default="data/citation_dataset/target_tex_artifacts_tagged.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model", default="gpt-5-mini-2025-08-07", help="LLM model name (OpenAI)."
    )
    parser.add_argument(
        "--max-artifacts", type=int, default=0, help="Limit artifacts processed."
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Max concurrent LLM calls."
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.artifacts):
        logger.error("Artifacts file not found: {}", args.artifacts)
        return 1

    ensure_dir(os.path.dirname(args.out) or ".")

    rows = list(read_jsonl(args.artifacts))
    if args.max_artifacts:
        rows = rows[: args.max_artifacts]

    if os.path.exists(args.out):
        os.remove(args.out)

    tagger = SemanticTagger(model=args.model, concurrency=args.concurrency)
    counters = await tagger.tag_artifacts(rows=rows, out_path=args.out)

    logger.info(
        "Done. Processed {} artifacts (failed {}). Wrote {} rows to {}",
        counters["processed"],
        counters["failed"],
        counters["tagged"],
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
