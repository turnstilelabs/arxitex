from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

_USAGE_SINKS: list[callable] = []


def register_usage_sink(fn) -> None:
    """Register a callable sink that will receive every TokenUsage event.

    A sink should be fast and best-effort. Any exceptions raised by sinks are
    swallowed so LLM calls never fail due to telemetry.
    """

    _USAGE_SINKS.append(fn)


@dataclass
class TokenUsage:
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    model: str
    provider: str
    cached: bool = False
    context: Optional[str] = None


def _read_usage_fields(
    usage: Any,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Best-effort extraction of token counts
    """
    if usage is None:
        return None, None, None

    # Dict-like
    if isinstance(usage, dict):
        return (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

    # Object with attributes
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    return pt, ct, tt


def log_usage(u: TokenUsage) -> None:
    logger.info(
        "LLM usage | provider={provider} model={model} cached={cached} "
        "prompt_tokens={pt} completion_tokens={ct} total_tokens={tt}{extra}".format(
            provider=u.provider,
            model=u.model,
            cached=u.cached,
            pt=u.prompt_tokens,
            ct=u.completion_tokens,
            tt=u.total_tokens,
            extra=f" context={u.context}" if u.context else "",
        )
    )

    # Best-effort sinks
    for sink in list(_USAGE_SINKS):
        try:
            sink(u)
        except Exception:
            # Never fail core execution due to metrics.
            pass


def log_response_usage(
    response: Any,
    *,
    model: str,
    provider: str,
    context: Optional[str] = None,
    cached: bool = False,
) -> None:
    """
    Convenience helper to log usage directly from a response object when it exposes `.usage`.
    Silently no-ops if usage is not available.
    """
    usage = getattr(response, "usage", None)
    pt, ct, tt = _read_usage_fields(usage)
    log_usage(
        TokenUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            model=model,
            provider=provider,
            cached=cached,
            context=context,
        )
    )
