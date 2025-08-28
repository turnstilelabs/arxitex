from loguru import logger
from arxitex.llms import llms

from arxitex.extractor.dependency_inference.dependency_prompt import DependencyInferencePromptGenerator
from arxitex.extractor.dependency_inference.dependency_models import PairwiseDependencyCheck

class GraphDependencyInference:
    def __init__(self):
        self.prompt_generator = DependencyInferencePromptGenerator()

    def infer_dependency(self, source_artifact: dict, target_artifact: dict) -> PairwiseDependencyCheck:
        """
        Infers if a dependency exists between two artifacts using an LLM.
        """
        prompt = self.prompt_generator.make_dependency_prompt(source_artifact, target_artifact)
        
        try:
            return llms.execute_prompt(
                prompt,
                output_class=PairwiseDependencyCheck,
                model="gpt-5-mini-2025-08-07"
            )
        except Exception as e:
            logger.error(f"Error during dependency inference: {e}")
            raise RuntimeError(f"Failed to infer dependency between artifacts: {source_artifact['id']} and {target_artifact['id']}")

    async def ainfer_dependency(self, source_artifact: dict, target_artifact: dict) -> PairwiseDependencyCheck:
        prompt = self.prompt_generator.make_dependency_prompt(source_artifact, target_artifact)
        
        try:
            return await llms.aexecute_prompt(
                prompt,
                output_class=PairwiseDependencyCheck,
                model="gpt-5-mini-2025-08-07"
            )
        except Exception as e:
            logger.error(f"Error during async dependency inference: {e}")
            raise RuntimeError(f"Failed to infer dependency between artifacts: {source_artifact['id']} and {target_artifact['id']}") from e