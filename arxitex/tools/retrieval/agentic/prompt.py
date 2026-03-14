"""Prompt template for agentic ColGREP retrieval."""

from __future__ import annotations

from arxitex.llms.prompt import Prompt


class AgenticColGrepPrompt:
    def make_prompt(
        self,
        *,
        mention: str,
        candidates: str,
        history: str,
        prompt_id: str,
    ) -> Prompt:
        system = """You are a retrieval agent for mathematical statements.
Decide the next action to find the single most relevant statement for the mention.
Return JSON only with keys: action, query, selected_id, confidence, rationale.
- action: one of "search", "refine", "final".
- query: new query string if action is search/refine, else null.
- selected_id: choose from candidate IDs if action is final.
- confidence: 0..1.
- rationale: one short sentence.

Rules:
- Prefer a precise statement match (theorem/lemma/definition) over vague text.
- If no candidate is clearly correct, refine with a more specific query.
- Only choose selected_id from the provided candidates.
"""
        user = f"""Mention context:
{mention}

Candidates:
{candidates}

History:
{history}

Return JSON only."""
        return Prompt(id=prompt_id, system=system, user=user)
