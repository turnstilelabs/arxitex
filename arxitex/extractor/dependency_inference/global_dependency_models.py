from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from arxitex.extractor.models import DependencyType


class GlobalDependencyEdge(BaseModel):
    """A single dependency edge output by global mode.

    Convention:
        - source_id depends on target_id (target is prerequisite).
    """

    source_id: str
    target_id: str
    dependency_type: DependencyType
    justification: Optional[str] = Field(
        None,
        description="Optional short justification. Prefer quoting a phrase from the source.",
    )


class GlobalDependencyGraph(BaseModel):
    edges: List[GlobalDependencyEdge] = Field(default_factory=list)
