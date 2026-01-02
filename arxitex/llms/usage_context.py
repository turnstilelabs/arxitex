from __future__ import annotations

"""Async-safe context for attributing LLM usage to a paper.

We use `contextvars` so that concurrent asyncio tasks (one per paper) can set
paper_id/mode/stage once and have *all* downstream LLM calls automatically
include those fields in token accounting.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

paper_id_var: ContextVar[Optional[str]] = ContextVar("arxitex_paper_id", default=None)
mode_var: ContextVar[Optional[str]] = ContextVar("arxitex_mode", default=None)
stage_var: ContextVar[Optional[str]] = ContextVar("arxitex_stage", default=None)


def get_usage_context() -> dict:
    return {
        "paper_id": paper_id_var.get(),
        "mode": mode_var.get(),
        "stage": stage_var.get(),
    }


@contextmanager
def llm_usage_context(
    *,
    paper_id: str,
    mode: Optional[str] = None,
    stage: Optional[str] = None,
) -> Iterator[None]:
    """Set the current paper context for downstream LLM calls."""

    t1 = paper_id_var.set(paper_id)
    t2 = mode_var.set(mode)
    t3 = stage_var.set(stage)
    try:
        yield
    finally:
        paper_id_var.reset(t1)
        mode_var.reset(t2)
        stage_var.reset(t3)


@contextmanager
def llm_usage_stage(stage: str) -> Iterator[None]:
    """Temporarily override the stage for downstream LLM calls."""

    tok = stage_var.set(stage)
    try:
        yield
    finally:
        stage_var.reset(tok)
