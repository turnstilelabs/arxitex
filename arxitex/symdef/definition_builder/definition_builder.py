from typing import List, Optional
from loguru import logger

from arxitex.symdef.definition_builder.definition_prompts import SymbolEnhancementPromptGenerator
from arxitex.symdef.definition_builder.definition_models import TermExtractionResult, DefinitionSynthesisResult, ExtractedDefinition
from arxitex.symdef.utils import Definition
from arxitex.llms import llms


class DefinitionBuilder:
    def __init__(self):
        self.prompt_generator = SymbolEnhancementPromptGenerator()
        
    def extract_terms(self, artifact_content: str) -> List[str]:
        """Extracts terms from an artifact using a structured LLM call."""
        prompt = self.prompt_generator.make_term_extraction_prompt(artifact_content)

        try:
            result = llms.execute_prompt(
                prompt,
                output_class=TermExtractionResult,
                model="gpt-4o-2024-08-06"
            )
            logger.info(f"LLM extracted terms: {result.terms}")
            return result.terms
        except Exception as e:
            logger.error(f"Error during term extraction: {e}")
            raise RuntimeError("Failed to extract terms from artifact content") from e

    def extract_definition(self, artifact_content: str) -> Definition:
        """Extracts a definition from an artifact that is itself a definition."""
        prompt = self.prompt_generator.make_definition_extraction_prompt(artifact_content)

        try:
            result = llms.execute_prompt(
                prompt,
                output_class=ExtractedDefinition,
                model="gpt-4o-2024-08-06"
            )
            logger.info(f"LLM extracted definition: {result.defined_term} - {result.definition_text}")
            return result
        except Exception as e:
            logger.error(f"Error during definition extraction: {e}")
            raise RuntimeError("Failed to extract definition from artifact content") from e

    def synthesize_definition(self, term: str, context_snippets: str, base_definition: Optional[Definition]) -> Optional[str]:
        """Synthesizes a definition using a structured LLM call."""
        prompt = self.prompt_generator.make_definition_synthesis_prompt(term, context_snippets, base_definition)
        try:
            result = llms.execute_prompt(
                prompt,
                output_class=DefinitionSynthesisResult,
                model="gpt-4o-2024-08-06"
            )
            if result.context_was_sufficient:
                return result.definition
            else:
                logger.warning(f"Insufficient context for term '{term}'. No definition synthesized.")
                return None
        except Exception as e:
            logger.error(f"Error during definition synthesis: {e}")
            raise RuntimeError(f"Failed to synthesize definition for term '{term}'") from e

    async def aextract_terms(self, artifact_content: str) -> List[str]:
        """Asynchronously extracts terms from an artifact."""
        prompt = self.prompt_generator.make_term_extraction_prompt(artifact_content)

        try:
            result = await llms.aexecute_prompt(
                prompt,
                output_class=TermExtractionResult,
                model="gpt-4o-2024-08-06"
            )
            logger.info(f"LLM extracted terms: {result.terms}")
            return result.terms
        except Exception as e:
            logger.error(f"Error during async term extraction: {e}")
            raise RuntimeError("Failed to extract terms from artifact content") from e

    async def aextract_definition(self, artifact_content: str) -> Definition:
        """Asynchronously extracts a definition from an artifact that is itself a definition."""
        prompt = self.prompt_generator.make_definition_extraction_prompt(artifact_content)

        try:
            result = await llms.aexecute_prompt(
                prompt,
                output_class=ExtractedDefinition,
                model="gpt-4o-2024-08-06"
            )
            logger.info(f"LLM extracted definition: {result.defined_term} - {result.definition_text}")
            return result
        except Exception as e:
            logger.error(f"Error during async definition extraction: {e}")
            raise RuntimeError("Failed to extract definition from artifact content") from e

    async def asynthesize_definition(self, term: str, context_snippets: str, base_definition: Optional[Definition]) -> Optional[str]:
        """Asynchronously synthesizes a definition."""
        prompt = self.prompt_generator.make_definition_synthesis_prompt(term, context_snippets, base_definition)
        try:
            result = await llms.aexecute_prompt(
                prompt,
                output_class=DefinitionSynthesisResult,
                model="gpt-4o-2024-08-06"
            )
            if result.context_was_sufficient:
                return result.definition
            else:
                logger.warning(f"Insufficient context for term '{term}'. No definition synthesized.")
                return None
        except Exception as e:
            logger.error(f"Error during async definition synthesis: {e}")
            raise RuntimeError(f"Failed to synthesize definition for term '{term}'") from e
    async def asynthesize_definition(self, term: str, context_snippets: str, base_definition: Optional[Definition]) -> Optional[str]:
        """Asynchronously synthesizes a definition."""
        prompt = self.prompt_generator.make_definition_synthesis_prompt(term, context_snippets, base_definition)
        try:
            result = await llms.aexecute_prompt(
                prompt,
                output_class=DefinitionSynthesisResult,
                model="gpt-4o-2024-08-06"
            )
            if result.context_was_sufficient:
                return result.definition
            else:
                logger.warning(f"Insufficient context for term '{term}'. No definition synthesized.")
                return None
        except Exception as e:
            logger.error(f"Error during async definition synthesis: {e}")
            raise RuntimeError(f"Failed to synthesize definition for term '{term}'") from e
