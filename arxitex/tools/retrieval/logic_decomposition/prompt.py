"""Prompt builders for logic decomposition."""

from __future__ import annotations

from arxitex.llms.prompt import Prompt


class LogicDecompositionPromptGenerator:
    def make_prompt(self, statement: str, prompt_id: str) -> Prompt:
        system = """You are a mathematical logic parser.
Return JSON only with keys: context, hypotheses, goal.
- context: short domain phrase. If unclear, return "".
- hypotheses: atomic predicates with id, predicate, args, polarity, quantifier, scope, raw.
- goal: object with raw, canonical_latex, ops.

Important:
- If the input is a location request (e.g. "where", "locate", "find a proof/reference"),
  strip the request and parse the underlying mathematical statement.
- Split hypotheses vs goal: conditions introduced by "for all/for every/let/given/assume/if/when"
  go into hypotheses; the main assertion goes into goal.
- Keep predicates short and canonical (snake_case). Use args if symbols are present.
- Treat symbol renaming as equivalent; do not invent missing symbols.
- If the input is a fragment or keyword query with no clear statement,
  output empty hypotheses [] and goal with empty fields, and only fill context if possible.
- Do not invent missing structure.
- Do not add any extra keys.

Example:
Input: "Where does Scholze prove that for a perfectoid K-algebra C, Frobenius induces an isomorphism C^\\circ/\\varpi^{1/p} \\cong C^\\circ/\\varpi?"
Output: context="perfectoid K-algebra Frobenius", hypotheses=[{"predicate":"perfectoid_k_algebra","args":["C"],"raw":"C is a perfectoid K-algebra"}],
goal.raw="Frobenius induces an isomorphism C^\\circ/\\varpi^{1/p} \\cong C^\\circ/\\varpi", goal.canonical_latex="Frob: C^\\circ/\\varpi^{1/p} \\cong C^\\circ/\\varpi"
"""

        user = f"""Statement:
{statement}

Return JSON only."""
        return Prompt(id=prompt_id, system=system, user=user)
