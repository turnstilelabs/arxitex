"""Agentic loop for ColGREP-backed retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List, Optional

from loguru import logger

from arxitex.llms.llms import execute_prompt
from arxitex.tools.retrieval.agentic.models import AgentDecision
from arxitex.tools.retrieval.agentic.prompt import AgenticColGrepPrompt


@dataclass
class AgentTraceStep:
    step: int
    query: str
    candidates: List[dict]
    decision: dict


@dataclass
class AgentResult:
    selected_id: Optional[str]
    candidates: List[str]
    scores: List[float]
    trace: List[AgentTraceStep]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _format_candidates(candidates: List[dict], max_items: int = 10) -> str:
    lines = []
    for c in candidates[:max_items]:
        lines.append(
            " | ".join(
                [
                    f"ID: {c.get('statement_id')}",
                    f"TYPE: {c.get('type')}",
                    f"NUMBER: {c.get('number')}",
                    f"SCORE: {c.get('score')}",
                    f"TEXT: {c.get('text_preview')}",
                ]
            )
        )
    return "\n".join(lines) if lines else "(no candidates)"


def _format_history(trace: List[AgentTraceStep]) -> str:
    if not trace:
        return "(none)"
    lines = []
    for step in trace:
        decision = step.decision
        lines.append(
            f"Step {step.step}: query={step.query} action={decision.get('action')} selected={decision.get('selected_id')}"
        )
    return "\n".join(lines)


def run_agentic_search(
    *,
    mention: str,
    search_fn: Callable[[str, int], List[dict]],
    model: str,
    max_steps: int = 3,
    top_k: int = 10,
    llm_executor: Callable = execute_prompt,
) -> AgentResult:
    trace: List[AgentTraceStep] = []
    query = mention
    selected_id = None
    candidates: List[dict] = []

    for step in range(1, max_steps + 1):
        candidates = search_fn(query, top_k) or []
        prompt_id = f"agentic-colgrep-{_hash(mention + query)[:12]}"
        prompt = AgenticColGrepPrompt().make_prompt(
            mention=mention,
            candidates=_format_candidates(candidates, max_items=top_k),
            history=_format_history(trace),
            prompt_id=prompt_id,
        )
        try:
            decision = llm_executor(prompt, AgentDecision, model=model)
            if isinstance(decision, AgentDecision):
                decision_dict = decision.model_dump()
            else:
                decision_dict = AgentDecision.model_validate(decision).model_dump()
        except Exception as exc:
            logger.warning("Agentic decision failed: {}", exc)
            decision_dict = {
                "action": "final",
                "selected_id": None,
                "confidence": 0.0,
                "rationale": "llm_error",
            }

        trace.append(
            AgentTraceStep(
                step=step,
                query=query,
                candidates=[
                    {
                        "statement_id": c.get("statement_id"),
                        "score": c.get("score"),
                    }
                    for c in candidates
                ],
                decision=decision_dict,
            )
        )

        action = (decision_dict.get("action") or "").lower()
        if action in {"search", "refine"}:
            new_query = (decision_dict.get("query") or "").strip()
            if new_query:
                query = new_query
                continue

        if action == "final":
            selected_id = decision_dict.get("selected_id")
            if selected_id:
                break

        # fallback: stop if we have candidates
        if candidates:
            selected_id = candidates[0].get("statement_id")
            break

    if not selected_id and candidates:
        selected_id = candidates[0].get("statement_id")

    ordered_ids = []
    scores = []
    if selected_id:
        ordered_ids.append(selected_id)
        sel_score = next(
            (
                c.get("score")
                for c in candidates
                if c.get("statement_id") == selected_id
            ),
            None,
        )
        scores.append(float(sel_score) if sel_score is not None else 0.0)

    for c in candidates:
        cid = c.get("statement_id")
        if not cid or cid in ordered_ids:
            continue
        ordered_ids.append(cid)
        score = c.get("score")
        scores.append(float(score) if score is not None else 0.0)

    return AgentResult(
        selected_id=selected_id,
        candidates=ordered_ids,
        scores=scores,
        trace=trace,
    )
