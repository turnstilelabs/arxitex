from __future__ import annotations

from loguru import logger

from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
)
from arxitex.extractor.dependency_inference.global_dependency_models import (
    GlobalDependencyGraph,
)
from arxitex.extractor.dependency_inference.global_dependency_prompt import (
    GlobalDependencyPromptGenerator,
)
from arxitex.extractor.models import ArtifactNode
from arxitex.llms import llms
from arxitex.llms.usage_context import llm_usage_stage


class GlobalGraphDependencyInference:
    def __init__(self):
        self.prompt_generator = GlobalDependencyPromptGenerator()

    async def ainfer_dependencies(
        self, artifacts: list[ArtifactNode], cfg: DependencyInferenceConfig
    ) -> GlobalDependencyGraph:
        prompt = self.prompt_generator.make_prompt(artifacts, cfg)
        logger.info(
            f"[global] Inferring dependencies for {len(artifacts)} artifacts (proof_char_budget={cfg.global_proof_char_budget})"
        )
        with llm_usage_stage("dependency_global"):
            return await llms.aexecute_prompt(
                prompt,
                output_class=GlobalDependencyGraph,
                model="gpt-5-mini-2025-08-07",
            )
