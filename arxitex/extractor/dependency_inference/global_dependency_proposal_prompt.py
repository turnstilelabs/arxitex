from __future__ import annotations

from typing import Iterable

from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
)
from arxitex.extractor.models import ArtifactNode
from arxitex.llms.prompt import Prompt


def _truncate_proof(proof: str | None, cfg: DependencyInferenceConfig) -> str:
    if not cfg.global_include_proofs:
        return "[omitted]"
    if not proof:
        return "No proof provided"
    if len(proof) <= cfg.global_proof_char_budget:
        return proof
    return proof[: cfg.global_proof_char_budget] + "\n[...truncated...]"


class GlobalDependencyProposalPromptGenerator:
    def make_prompt(
        self, artifacts: Iterable[ArtifactNode], cfg: DependencyInferenceConfig
    ) -> Prompt:
        system = """
        You are an expert mathematician.
        Your task is to PROPOSE likely prerequisite dependencies between artifacts from a single paper.

        Output must be a single JSON object {"edges": [...] }.
        Each edge must have `source_id` and `target_id`.

        Sparsity rule:
        - Prefer earlier artifacts as prerequisites.
        - Avoid redundant edges.
        - Do not output self-loops.

        This is a proposal stage. Do not be overly verbose.
        """

        chunks = []
        for a in artifacts:
            chunks.append(
                f"""
                ## Artifact
                id: {a.id}
                type: {a.type.value}
                label: {a.label or 'N/A'}
                statement:
                ```latex
                {a.content}
                ```
                proof (may be truncated):
                ```latex
                {_truncate_proof(a.proof, cfg)}
                ```
                """.strip()
            )
        user = "\n\n".join(chunks) + "\n\nReturn only JSON."
        return Prompt(id="global_dependency_proposal", system=system, user=user)
