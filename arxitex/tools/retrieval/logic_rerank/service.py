"""Logic-aware reranking service."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from arxitex.llms.llms import aexecute_prompt
from arxitex.tools.retrieval.logic_decomposition.models import LogicDecomposition
from arxitex.tools.retrieval.logic_rerank.models import HypothesisBatchScore
from arxitex.tools.retrieval.logic_rerank.prompt import LogicBatchScorePromptGenerator
from arxitex.tools.retrieval.msc2020 import MSCMatch, context_similarity
from arxitex.tools.retrieval.normalization import normalize_text

_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*|\d+|[=+\-*/^(){}\[\]]")
_VAR_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\b")
_RESERVED_VARS = {
    "sin",
    "cos",
    "tan",
    "log",
    "exp",
    "lim",
    "sum",
    "prod",
    "int",
}

_LOGIC_BATCH_TIMEOUT_SECONDS = float(
    os.getenv("ARXITEX_LOGIC_BATCH_TIMEOUT_SECONDS", "60")
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenize_goal(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").strip())


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _debruijnize(expr: str) -> str:
    mapping: Dict[str, str] = {}
    order = 1

    def repl(match: re.Match) -> str:
        nonlocal order
        tok = match.group(1)
        if tok in _RESERVED_VARS:
            return tok
        if tok not in mapping:
            mapping[tok] = f"var_{order}"
            order += 1
        return mapping[tok]

    return _VAR_RE.sub(repl, expr or "")


def compute_goal_score(
    query_goal: str,
    result_goal: str,
    *,
    use_debruijn: bool = True,
) -> float:
    q = normalize_text(query_goal, use_math_verify=True) or query_goal or ""
    r = normalize_text(result_goal, use_math_verify=True) or result_goal or ""
    if q and q == r:
        return 1.0

    if use_debruijn:
        qd = _debruijnize(q)
        rd = _debruijnize(r)
        if qd and qd == rd:
            return 1.0

    return _jaccard(_tokenize_goal(q), _tokenize_goal(r))


class LogicReranker:
    def __init__(
        self,
        *,
        model: str,
        top_n: int = 20,
        concurrency: int = 6,
        use_debruijn: bool = True,
    ) -> None:
        self.model = model
        self.top_n = max(1, int(top_n))
        self.concurrency = max(1, int(concurrency))
        self.use_debruijn = use_debruijn
        self.prompt_generator = LogicBatchScorePromptGenerator()

    async def _score_hypotheses(
        self,
        *,
        query_decomposition: LogicDecomposition,
        result_decomposition: LogicDecomposition,
    ) -> Tuple[float, bool, str]:
        if not result_decomposition.hypotheses or not query_decomposition.hypotheses:
            return 0.0, False, ""

        pid = (
            "logic-batch-"
            + _hash(
                "|".join(h.model_dump_json() for h in result_decomposition.hypotheses)
                + "||"
                + "|".join(h.model_dump_json() for h in query_decomposition.hypotheses)
            )[:16]
        )
        prompt = self.prompt_generator.make_prompt(
            result_hypotheses=result_decomposition.hypotheses,
            query_hypotheses=query_decomposition.hypotheses,
            prompt_id=pid,
        )
        try:
            out = await asyncio.wait_for(
                aexecute_prompt(prompt, HypothesisBatchScore, model=self.model),
                timeout=_LOGIC_BATCH_TIMEOUT_SECONDS,
            )
            shyp = max(0.0, min(1.0, float(out.shyp)))
            return shyp, bool(out.contradiction), out.rationale or ""
        except asyncio.TimeoutError:
            logger.warning(
                "Logic batch call failed: timeout after {}s",
                _LOGIC_BATCH_TIMEOUT_SECONDS,
            )
            return 0.0, False, "timeout"
        except Exception as exc:
            logger.warning("Logic batch call failed: {}", exc)
            return 0.0, False, "batch_call_failed"

    async def rerank_candidates_async(
        self,
        *,
        query_id: str,
        candidate_ids: List[str],
        base_scores: List[float],
        query_logic: Dict[str, LogicDecomposition],
        artifact_logic: Dict[str, LogicDecomposition],
        query_msc: Dict[str, MSCMatch],
        artifact_msc: Dict[str, MSCMatch],
    ) -> Tuple[List[str], List[float], List[Dict]]:
        if not candidate_ids:
            return [], [], []

        query_dec = query_logic.get(query_id)
        if not query_dec:
            return candidate_ids, base_scores, []

        head_n = min(self.top_n, len(candidate_ids))
        head_ids = candidate_ids[:head_n]
        head_scores = base_scores[:head_n] if base_scores else [0.0] * head_n
        tail_ids = candidate_ids[head_n:]
        tail_scores = base_scores[head_n:] if base_scores else []

        min_s = min(head_scores) if head_scores else 0.0
        max_s = max(head_scores) if head_scores else 1.0

        def normalize(s: float) -> float:
            if max_s <= min_s:
                return 1.0
            return (s - min_s) / (max_s - min_s)

        sem = asyncio.Semaphore(self.concurrency)
        q_code = (query_msc.get(query_id) or MSCMatch(None, None)).code

        async def score_one(idx: int, aid: str):
            async with sem:
                result_dec = artifact_logic.get(aid)
                if not result_dec:
                    return None

                shyp, contradiction, rationale = await self._score_hypotheses(
                    query_decomposition=query_dec,
                    result_decomposition=result_dec,
                )
                if contradiction:
                    return None

                sgoal = compute_goal_score(
                    query_dec.goal.canonical_latex or query_dec.goal.raw,
                    result_dec.goal.canonical_latex or result_dec.goal.raw,
                    use_debruijn=self.use_debruijn,
                )
                a_code = (artifact_msc.get(aid) or MSCMatch(None, None)).code
                sctx = context_similarity(q_code, a_code)
                base_norm = normalize(head_scores[idx])
                slogic = 0.60 * shyp + 0.25 * sgoal + 0.15 * sctx
                sfinal = 0.40 * base_norm + 0.60 * slogic
                if rationale:
                    logger.info(
                        "Logic rationale qid={} aid={} shyp={:.3f}: {}",
                        query_id,
                        aid,
                        shyp,
                        rationale,
                    )
                return (
                    aid,
                    sfinal,
                    {
                        "artifact_id": aid,
                        "Sbase": base_norm,
                        "Shyp": shyp,
                        "Sgoal": sgoal,
                        "Sctx": sctx,
                        "Sfinal": sfinal,
                    },
                )

        results = await asyncio.gather(
            *(score_one(idx, aid) for idx, aid in enumerate(head_ids))
        )
        ranked = []
        breakdown = []
        for item in results:
            if not item:
                continue
            aid, sfinal, detail = item
            ranked.append((aid, sfinal))
            breakdown.append(detail)

        ranked.sort(key=lambda x: (-x[1], x[0]))
        breakdown.sort(key=lambda x: -x["Sfinal"])

        ranked_ids = [aid for aid, _ in ranked] + tail_ids
        ranked_scores = [score for _, score in ranked] + tail_scores
        return ranked_ids, ranked_scores, breakdown

    def rerank_candidates(
        self,
        *,
        query_id: str,
        candidate_ids: List[str],
        base_scores: List[float],
        query_logic: Dict[str, LogicDecomposition],
        artifact_logic: Dict[str, LogicDecomposition],
        query_msc: Dict[str, MSCMatch],
        artifact_msc: Dict[str, MSCMatch],
    ) -> Tuple[List[str], List[float], List[Dict]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.rerank_candidates_async(
                    query_id=query_id,
                    candidate_ids=candidate_ids,
                    base_scores=base_scores,
                    query_logic=query_logic,
                    artifact_logic=artifact_logic,
                    query_msc=query_msc,
                    artifact_msc=artifact_msc,
                )
            )
        raise RuntimeError(
            "rerank_candidates cannot be called from an active event loop; "
            "use rerank_candidates_async instead."
        )


async def apply_logic_rerank_async(
    *,
    results: Dict[str, Dict],
    query_ids: List[str],
    id_lookup: Dict[int, str],
    query_logic: Dict[str, LogicDecomposition],
    artifact_logic: Dict[str, LogicDecomposition],
    query_msc: Dict[str, MSCMatch],
    artifact_msc: Dict[str, MSCMatch],
    reranker: Optional[LogicReranker],
) -> Dict[str, Dict]:
    if reranker is None:
        return results

    reverse_lookup = {v: k for k, v in id_lookup.items()}
    total = len(query_ids)
    completed = 0
    skipped = 0
    report_every = max(1, total // 20)
    logger.info(
        "Logic rerank: total_queries={} top_n={} concurrency={}",
        total,
        reranker.top_n,
        reranker.concurrency,
    )

    for qid in query_ids:
        row = results.get(qid)
        if not row:
            skipped += 1
            continue

        indices = row.get("indices", [])
        scores = row.get("scores", [])
        if not indices:
            skipped += 1
            continue
        if not scores or len(scores) != len(indices):
            scores = [0.0] * len(indices)

        candidate_ids = []
        for idx in indices:
            if isinstance(idx, int):
                aid = id_lookup.get(idx)
            else:
                aid = idx
            if aid:
                candidate_ids.append(aid)

        reranked_ids, reranked_scores, breakdown = (
            await reranker.rerank_candidates_async(
                query_id=qid,
                candidate_ids=candidate_ids,
                base_scores=scores,
                query_logic=query_logic,
                artifact_logic=artifact_logic,
                query_msc=query_msc,
                artifact_msc=artifact_msc,
            )
        )

        new_indices = [reverse_lookup.get(aid, aid) for aid in reranked_ids]
        row["indices"] = new_indices
        row["scores"] = reranked_scores
        row["artifact_ids"] = reranked_ids
        row["logic_rerank"] = breakdown
        completed += 1
        done = completed + skipped
        if done == total or done % report_every == 0:
            pct = (done / total) * 100 if total else 100.0
            logger.info(
                "Logic rerank progress: {}/{} ({:.1f}%) processed={} skipped={}",
                done,
                total,
                pct,
                completed,
                skipped,
            )

    return results


def apply_logic_rerank(
    *,
    results: Dict[str, Dict],
    query_ids: List[str],
    id_lookup: Dict[int, str],
    query_logic: Dict[str, LogicDecomposition],
    artifact_logic: Dict[str, LogicDecomposition],
    query_msc: Dict[str, MSCMatch],
    artifact_msc: Dict[str, MSCMatch],
    reranker: Optional[LogicReranker],
) -> Dict[str, Dict]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            apply_logic_rerank_async(
                results=results,
                query_ids=query_ids,
                id_lookup=id_lookup,
                query_logic=query_logic,
                artifact_logic=artifact_logic,
                query_msc=query_msc,
                artifact_msc=artifact_msc,
                reranker=reranker,
            )
        )
    raise RuntimeError(
        "apply_logic_rerank cannot be called from an active event loop; "
        "use apply_logic_rerank_async instead."
    )
