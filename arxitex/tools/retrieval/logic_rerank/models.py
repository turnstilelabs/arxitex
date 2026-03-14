"""Structured outputs and helper models for logic reranking."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HypothesisBatchScore(BaseModel):
    shyp: float
    contradiction: bool = False
    rationale: Optional[str] = None
