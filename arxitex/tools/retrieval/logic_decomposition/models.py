"""Structured models for logic decomposition outputs."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class LogicHypothesis(BaseModel):
    id: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    args: List[str] = Field(default_factory=list)
    polarity: Literal["pos", "neg"] = "pos"
    quantifier: Literal["forall", "exists", "none"] = "none"
    scope: str = "global"
    raw: str = ""


class LogicGoal(BaseModel):
    raw: str = ""
    canonical_latex: str = ""
    ops: List[str] = Field(default_factory=list)


class LogicDecomposition(BaseModel):
    context: str = ""
    hypotheses: List[LogicHypothesis] = Field(default_factory=list)
    goal: LogicGoal = Field(default_factory=LogicGoal)
