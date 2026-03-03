"""Structured extraction for retrieval inputs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger
from pydantic import BaseModel, Field

from arxitex.llms.llms import aexecute_prompt
from arxitex.llms.prompt import Prompt


class StructuredFields(BaseModel):
    math_terms: List[str] = Field(default_factory=list)
    math_exprs: List[str] = Field(default_factory=list)
    domain_terms: List[str] = Field(default_factory=list)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # drop control chars and normalize whitespace
    text = _CONTROL_RE.sub(" ", text)
    return " ".join(text.split())


def _clean_list(items: Iterable[str], max_items: int = 30) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        s = " ".join(str(item).split())
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _build_prompt(text: str, prompt_id: str) -> Prompt:
    system = """You extract structured information from mathematical text.
Return JSON only with keys: math_terms, math_exprs, domain_terms.
- math_terms: canonical math entities or objects (nouns).
- math_exprs: LaTeX or symbolic expressions as they appear.
- domain_terms: technical phrases that are not purely symbolic.
Keep each list short and specific (max ~12 items each)."""

    user = f"""Text:
{text}

Return JSON only."""
    return Prompt(id=prompt_id, system=system, user=user)


async def _extract_one(text: str, model: str, prompt_id: str) -> StructuredFields:
    prompt = _build_prompt(_clean_text(text), prompt_id)
    result = await aexecute_prompt(prompt, StructuredFields, model=model)
    # Normalize lists
    return StructuredFields(
        math_terms=_clean_list(result.math_terms),
        math_exprs=_clean_list(result.math_exprs),
        domain_terms=_clean_list(result.domain_terms),
    )


def _load_cache(path: Path) -> Dict[str, StructuredFields]:
    cache: Dict[str, StructuredFields] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("id")
        fields = row.get("fields") or {}
        if not key:
            continue
        cache[key] = StructuredFields(**fields)
    return cache


def _append_cache(path: Path, key: str, fields: StructuredFields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": key, "fields": fields.model_dump()}, ensure_ascii=False)
            + "\n"
        )


def extract_structured(
    *,
    texts: List[str],
    ids: List[str],
    model: str,
    cache_path: Path,
    concurrency: int = 4,
) -> Dict[str, StructuredFields]:
    cache = _load_cache(cache_path)
    pending = [(tid, text) for tid, text in zip(ids, texts) if tid not in cache]
    if not pending:
        return cache

    async def run() -> Dict[str, StructuredFields]:
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def process(tid: str, text: str) -> None:
            async with sem:
                try:
                    prompt_id = f"struct-{_hash(text)[:12]}"
                    fields = await _extract_one(text, model, prompt_id)
                    cache[tid] = fields
                    _append_cache(cache_path, tid, fields)
                except Exception as exc:
                    logger.warning("Structured extraction failed for {}: {}", tid, exc)

        await asyncio.gather(*(process(tid, text) for tid, text in pending))
        return cache

    return asyncio.run(run())
