"""Prompt builders for batched hypothesis scoring."""

from __future__ import annotations

from arxitex.llms.prompt import Prompt
from arxitex.tools.retrieval.logic_decomposition.models import LogicHypothesis


class LogicBatchScorePromptGenerator:
    def make_prompt(
        self,
        *,
        result_hypotheses: list[LogicHypothesis],
        query_hypotheses: list[LogicHypothesis],
        prompt_id: str,
    ) -> Prompt:
        system = """You are a mathematical logic verifier.
Given two lists of hypotheses (result vs query), compute a single score shyp in [0,1]
that simulates max-similarity aggregation across hypotheses.

Scoring rule:
- For each result hypothesis hr, find the best-matching query hypothesis hq.
- Assign the per-hr score:
  1.0 = hq entails hr or is equivalent (query is at least as strong).
  0.7 = hr entails hq (result is strictly stronger/specializes).
  0.4 = partial overlap (same predicate/topic but different scope/args).
  0.0 = unrelated.
- shyp = average of per-hr best scores across all hr in the result list.
- contradiction=true if any hr contradicts any hq (mutually exclusive).

Use raw text plus predicate/args/scope to judge equivalence; treat symbol renaming
as equivalent. If either list is empty, return shyp=0 and contradiction=false.

Return JSON only with keys: shyp, contradiction, rationale.
The rationale must be brief (1-2 sentences). No extra keys."""

        def fmt(h: LogicHypothesis) -> str:
            return (
                f"id={h.id}; predicate={h.predicate}; args={h.args}; "
                f"polarity={h.polarity}; quantifier={h.quantifier}; "
                f"scope={h.scope}; raw={h.raw}"
            )

        result_block = "\n".join(f"- {fmt(h)}" for h in result_hypotheses)
        query_block = "\n".join(f"- {fmt(h)}" for h in query_hypotheses)

        user = f"""Result hypotheses:
{result_block}

Query hypotheses:
{query_block}

Return JSON only."""
        return Prompt(id=prompt_id, system=system, user=user)
