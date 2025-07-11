from typing import List, Optional
from pydantic import BaseModel, Field

class TermExtractionResult(BaseModel):
    """
    The structured response for extracting terms that need definition from an artifact.
    """
    terms: List[str] = Field(
        ...,
        description="A list of non-trivial mathematical symbols and specialized concepts found in the text."
    )

class DefinitionSynthesisResult(BaseModel):
    """
    The structured response for synthesizing a definition for a given term.
    """
    context_was_sufficient: bool = Field(
        ...,
        description="True if the provided context was sufficient to create a confident definition, otherwise false."
    )
    definition: Optional[str] = Field(
        None,
        description="The synthesized definition text. This should be null if the context was insufficient."
    )

class ExtractedDefinition(BaseModel):
    """
    The structured data extracted from a formal definition artifact.
    This includes the primary term being defined, its definition text, and any aliases.
    """
    defined_term: str
    definition_text: str
    aliases: List[str] = Field(default_factory=list)