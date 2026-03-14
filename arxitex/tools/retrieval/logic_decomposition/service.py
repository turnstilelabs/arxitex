"""LLM-backed logic decomposition extraction with caching and fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

from loguru import logger

from arxitex.llms.llms import aexecute_prompt
from arxitex.tools.retrieval.logic_decomposition.models import (
    LogicDecomposition,
    LogicGoal,
)
from arxitex.tools.retrieval.logic_decomposition.prompt import (
    LogicDecompositionPromptGenerator,
)
from arxitex.tools.retrieval.normalization import normalize_text

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_OP_RE = re.compile(
    r"(\\sum|\\prod|\\int|\\forall|\\exists|<=|>=|!=|=|\+|\-|\*|/|\\to|\\mapsto)"
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = _CONTROL_RE.sub(" ", text)
    return " ".join(text.split())


def _fallback_decomposition(text: str) -> LogicDecomposition:
    canonical = normalize_text(text, use_math_verify=False) or text
    ops = _OP_RE.findall(canonical)
    return LogicDecomposition(
        context="",
        hypotheses=[],
        goal=LogicGoal(raw=text, canonical_latex=canonical, ops=ops),
    )


def _load_cache(path: Path) -> Dict[str, LogicDecomposition]:
    cache: Dict[str, LogicDecomposition] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row.get("id")
        dec = row.get("decomposition") or {}
        if not rid:
            continue
        try:
            cache[rid] = LogicDecomposition.model_validate(dec)
        except Exception:
            logger.warning("Invalid logic decomposition cache row for id={}", rid)
    return cache


def _append_cache(path: Path, rid: str, dec: LogicDecomposition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"id": rid, "decomposition": dec.model_dump()}, ensure_ascii=False
            )
            + "\n"
        )


async def _extract_one(text: str, model: str, prompt_id: str) -> LogicDecomposition:
    prompt = LogicDecompositionPromptGenerator().make_prompt(
        _clean_text(text), prompt_id
    )
    result = await aexecute_prompt(prompt, LogicDecomposition, model=model)
    return LogicDecomposition.model_validate(result)


def extract_logic_decompositions(
    *,
    texts: List[str],
    ids: List[str],
    model: str,
    cache_path: Path,
    concurrency: int = 4,
) -> Dict[str, LogicDecomposition]:
    cache = _load_cache(cache_path)
    # Avoid redundant calls for duplicate texts in the same run.
    seen_texts = {_clean_text(dec.goal.raw): rid for rid, dec in cache.items()}
    pending = []
    cache_hits = 0
    dedup_hits = 0
    for rid, text in zip(ids, texts):
        if rid in cache:
            cache_hits += 1
            continue
        cleaned = _clean_text(text)
        if cleaned in seen_texts:
            cache[rid] = cache[seen_texts[cleaned]]
            _append_cache(cache_path, rid, cache[rid])
            dedup_hits += 1
            continue
        seen_texts[cleaned] = rid
        pending.append((rid, text))
    total = len(ids)
    pending_total = len(pending)
    logger.info(
        "Logic decomposition: total={} cache_hits={} dedup_hits={} pending={} cache_path={}",
        total,
        cache_hits,
        dedup_hits,
        pending_total,
        cache_path,
    )
    if not pending:
        return cache

    async def run() -> Dict[str, LogicDecomposition]:
        sem = asyncio.Semaphore(max(1, int(concurrency)))
        lock = asyncio.Lock()
        write_lock = asyncio.Lock()
        completed = 0
        failed = 0
        report_every = max(1, pending_total // 20)

        async def process(rid: str, text: str) -> None:
            async with sem:
                prompt_id = f"logic-decomp-{_hash(text)[:12]}"
                had_error = False
                try:
                    dec = await _extract_one(text, model, prompt_id)
                except Exception as exc:
                    logger.warning("Logic decomposition failed for {}: {}", rid, exc)
                    dec = _fallback_decomposition(text)
                    had_error = True
                cache[rid] = dec
                async with write_lock:
                    _append_cache(cache_path, rid, dec)
                async with lock:
                    nonlocal completed, failed
                    completed += 1
                    if had_error:
                        failed += 1
                    if completed == pending_total or completed % report_every == 0:
                        pct = (
                            (completed / pending_total) * 100
                            if pending_total
                            else 100.0
                        )
                        logger.info(
                            "Logic decomposition progress: {}/{} ({:.1f}%) failed={}",
                            completed,
                            pending_total,
                            pct,
                            failed,
                        )

        await asyncio.gather(*(process(rid, text) for rid, text in pending))
        logger.info(
            "Logic decomposition done: completed={} failed={} pending=0",
            pending_total,
            failed,
        )
        return cache

    return asyncio.run(run())
