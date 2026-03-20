from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class AgentDecision(BaseModel):
    action: Literal["search", "refine", "final"]
    query: Optional[str] = None
    selected_id: Optional[str] = None
    confidence: Optional[float] = 0.0
    rationale: Optional[str] = ""
