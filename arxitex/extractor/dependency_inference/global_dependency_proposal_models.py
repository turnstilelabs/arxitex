from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ProposedEdge(BaseModel):
    source_id: str
    target_id: str
    rationale: Optional[str] = Field(
        None, description="Optional short rationale (not used as ground truth)."
    )


class ProposedEdges(BaseModel):
    edges: List[ProposedEdge] = Field(default_factory=list)
