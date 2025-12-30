from __future__ import annotations

from loguru import logger

from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
)
from arxitex.extractor.dependency_inference.global_dependency_proposal_models import (
    ProposedEdges,
)
from arxitex.extractor.dependency_inference.global_dependency_proposal_prompt import (
    GlobalDependencyProposalPromptGenerator,
)
from arxitex.extractor.models import ArtifactNode
from arxitex.llms import llms
from arxitex.llms.usage_context import llm_usage_stage


class GlobalDependencyProposer:
    def __init__(self):
        self.prompt_generator = GlobalDependencyProposalPromptGenerator()

    async def apropose(
        self, artifacts: list[ArtifactNode], cfg: DependencyInferenceConfig
    ) -> ProposedEdges:
        prompt = self.prompt_generator.make_prompt(artifacts, cfg)
        logger.info(
            f"[hybrid] Proposing candidate dependencies for {len(artifacts)} artifacts (topk_per_source={cfg.hybrid_topk_per_source}, proof_char_budget={cfg.global_proof_char_budget})"
        )
        with llm_usage_stage("dependency_proposal"):
            return await llms.aexecute_prompt(
                prompt,
                output_class=ProposedEdges,
                model="gpt-5-mini-2025-08-07",
            )
