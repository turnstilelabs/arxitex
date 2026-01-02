from __future__ import annotations

from typing import Iterable

from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
)
from arxitex.extractor.models import ArtifactNode, DependencyType
from arxitex.llms.prompt import Prompt


def _truncate_proof(proof: str | None, cfg: DependencyInferenceConfig) -> str:
    if not cfg.global_include_proofs:
        return "[omitted]"
    if not proof:
        return "No proof provided"
    if len(proof) <= cfg.global_proof_char_budget:
        return proof
    return proof[: cfg.global_proof_char_budget] + "\n[...truncated...]"


class GlobalDependencyPromptGenerator:
    def make_prompt(
        self, artifacts: Iterable[ArtifactNode], cfg: DependencyInferenceConfig
    ) -> Prompt:
        dependency_options = ", ".join([f"`{dtype.value}`" for dtype in DependencyType])

        system = f"""
        You are an expert mathematician and logician.
        Your task: infer a dependency graph between mathematical artifacts from a single paper.

        A directed edge means: the Source artifact depends on the Target artifact (Target is a prerequisite).

        You MUST output a single JSON object with an `edges` array.
        Each edge must include:
          - `source_id`
          - `target_id`
          - `dependency_type` (one of: {dependency_options})
          - optional `justification` (short; ideally quote a phrase from the source)

        Rules:
        - Only create an edge if the dependency is reasonably clear from the provided text.
        - Prefer a sparse graph: avoid redundant edges.
        - Do not create self-loops.
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

        user = "\n\n".join(chunks) + "\n\nReturn only JSON."  # keep strict

        return Prompt(id="global_dependency_graph", system=system, user=user)
