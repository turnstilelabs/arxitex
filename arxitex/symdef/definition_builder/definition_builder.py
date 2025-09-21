from typing import List, Optional

from loguru import logger

from arxitex.llms import llms
from arxitex.symdef.definition_builder.definition_models import (
    DefinitionSynthesisResult,
    DocumentTermExtractionResult,
    ExtractedDefinition,
    TermExtractionResult,
)
from arxitex.symdef.definition_builder.definition_prompts import (
    SymbolEnhancementPromptGenerator,
)
from arxitex.symdef.utils import Definition


class DefinitionBuilder:
    def __init__(self):
        self.prompt_generator = SymbolEnhancementPromptGenerator()

    async def aextract_single_artifact_terms(self, artifact_content: str) -> List[str]:
        """Asynchronously extracts terms from an artifact."""
        prompt = self.prompt_generator.make_term_extraction_prompt(artifact_content)

        try:
            result = await llms.aexecute_prompt(
                prompt, output_class=TermExtractionResult, model="gpt-4.1-2025-04-14"
            )
            logger.info(f"LLM extracted terms: {result.terms}")
            return result.terms
        except Exception as e:
            logger.error(f"Error during async term extraction: {e}")
            raise RuntimeError("Failed to extract terms from artifact content") from e

    async def aextract_document_terms(self, full_document_content: str) -> List[str]:
        """
        Asynchronously extracts all significant terms from the full document content in a single call.
        """
        prompt = self.prompt_generator.make_document_term_extraction_prompt(
            full_document_content
        )
        logger.debug(f"Document-wide term extraction prompt: {prompt}")
        try:
            result = await llms.aexecute_prompt(
                prompt,
                output_class=DocumentTermExtractionResult,
                model="gpt-4.1-2025-04-14",  # "deepseek-ai/DeepSeek-V3.1"
            )
            logger.info(
                f"LLM extracted {len(result.terms)} unique terms from the entire document."
            )
            return result.terms
        except Exception as e:
            logger.error(f"Error during async document-wide term extraction: {e}")
            raise RuntimeError(
                "Failed to extract terms from the full document content"
            ) from e

    async def aextract_definition(self, artifact_content: str) -> Definition:
        """Asynchronously extracts a definition from an artifact that is itself a definition."""
        prompt = self.prompt_generator.make_definition_extraction_prompt(
            artifact_content
        )

        try:
            result = await llms.aexecute_prompt(
                prompt, output_class=ExtractedDefinition, model="gpt-4.1-2025-04-14"
            )
            logger.info(
                f"LLM extracted definition: {result.defined_term} - {result.definition_text}"
            )
            return result
        except Exception as e:
            logger.error(f"Error during async definition extraction: {e}")
            raise RuntimeError(
                "Failed to extract definition from artifact content"
            ) from e

    async def asynthesize_definition(
        self, term: str, context_snippets: str, base_definition: Optional[Definition]
    ) -> Optional[str]:
        """Asynchronously synthesizes a definition."""
        prompt = self.prompt_generator.make_definition_synthesis_prompt(
            term, context_snippets, base_definition
        )
        try:
            result = await llms.aexecute_prompt(
                prompt,
                output_class=DefinitionSynthesisResult,
                model="gpt-4.1-2025-04-14",
            )
            if result.context_was_sufficient:
                return result.definition
            else:
                logger.warning(
                    f"Insufficient context for term '{term}'. No definition synthesized."
                )
                return None
        except Exception as e:
            logger.error(f"Error during async definition synthesis: {e}")
            raise RuntimeError(
                f"Failed to synthesize definition for term '{term}'"
            ) from e
