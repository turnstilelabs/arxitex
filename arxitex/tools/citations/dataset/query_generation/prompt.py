from __future__ import annotations

import re
from typing import Optional

from arxitex.llms.prompt import Prompt
from arxitex.tools.citations.dataset.query_generation.models import MentionContext


def sanitize_prompt_context(text: str) -> str:
    if not text:
        return ""
    s = text
    # Drop bracketed citation labels like [Sch12], [KS1]
    s = re.sub(r"\[[A-Za-z]{2,}\d{2,}[^\]]*\]", " ", s)
    # Drop explicit numbered refs like Theorem 1.3, Def. 2.6(ii), §3.2
    s = re.sub(
        r"\b(?:Theorem|Thm\.?|Lemma|Lem\.?|Proposition|Prop\.?|Corollary|Cor\.?|Definition|Def\.?|Example|Ex\.?|Remark|Rem\.?)\s*"
        r"\d+(?:\.\d+)*\s*(?:\([ivxIVX]+\))?",
        " ",
        s,
    )
    s = re.sub(r"\b(?:Section|Sec\.?)\s*\d+(?:\.\d+)*\b", " ", s)
    s = re.sub(r"§\s*\d+(?:\.\d+)*", " ", s)
    s = re.sub(r"\(\s*\d+(?:\.\d+)*\s*\)", " ", s)
    # Drop common bibliographic metadata patterns
    s = re.sub(r"\bMR\s*\d+\b", " ", s)
    s = re.sub(r"\bDOI:\s*\S+\b", " ", s, flags=re.I)
    s = re.sub(r"\b10\.\d{4,9}/\S+\b", " ", s)
    s = re.sub(r"\bPubl\.\s*Math\.\s*IH[ÉE]S\b", " ", s, flags=re.I)
    s = re.sub(r"\bInvent\.\s*Math\.\b", " ", s, flags=re.I)
    s = re.sub(r"\bMath\.\s*Ann\.\b", " ", s, flags=re.I)
    s = re.sub(r"\bJ\.\s*\w+\.\s*Math\.\b", " ", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


class QueryPromptGenerator:
    def make_prompt(
        self,
        ctx: MentionContext,
        *,
        style: str,
        target_name: str,
        prompt_id: str,
    ) -> Prompt:
        context_prev = ctx.context_prev or ""
        context_sentence = ctx.context_sentence or ""
        context_next = ctx.context_next or ""
        stitched = " ".join(
            [s for s in [context_prev, context_sentence, context_next] if s]
        ).strip()
        raw_paragraph = stitched
        if not raw_paragraph and ctx.context_html:
            raw_paragraph = self._html_to_text(ctx.context_html or "")
        full_paragraph = self._truncate(raw_paragraph or "", 1500) or ""
        sanitized = sanitize_prompt_context(full_paragraph)

        system = f"""You are a mathematician generating a realistic search query.
You must simulate a researcher writing the paper who knows the needed
result is in {target_name} but does NOT know the exact location or name.
Return only the query text as a question sentence. Do not answer."""

        user = f"""
Context:
{sanitized}

Task:
- Generate exactly 1 question-sentence query.
- The query should be a realistic question a researcher would ask
  to find where in {target_name} this result appears.
- The researcher knows it's in {target_name} but not the name or location.
- Do NOT include answers.
- Style: {style}. If style is "precise", use technical hints from context.
  If style is "vague", be less specific but still technical.
"""

        return Prompt(id=prompt_id, system=system, user=user)

    @staticmethod
    def _truncate(text: Optional[str], max_len: int = 800) -> Optional[str]:
        if text is None:
            return None
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _html_to_text(html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html)
        return " ".join(text.split())
