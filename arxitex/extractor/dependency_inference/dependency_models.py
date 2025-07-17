from typing import Optional
from pydantic import BaseModel, Field

from arxitex.extractor.utils import DependencyType
    
class PairwiseDependencyCheck(BaseModel):
    """
    The structured response from the LLM for a single pair of artifacts.
    """
    has_dependency: bool = Field(..., description="A boolean flag that is true if any dependency exists, and false otherwise.")
    dependency_type: Optional[DependencyType] = Field(None, description="The specific type of dependency, if one exists.")
    justification: Optional[str] = Field(None, description="A detailed justification, quoting from the source artifact's content to prove the dependency link. Required if has_dependency is true.")

