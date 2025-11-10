from __future__ import annotations

import asyncio
import os

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Best-effort import of OpenAI SDK exceptions (available in openai >=1.x)
try:
    from openai import (  # type: ignore
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )

    OPENAI_EXC = (APIError, APIConnectionError, RateLimitError, APITimeoutError)
except Exception:  # pragma: no cover - if SDK changes or not installed in tests
    OPENAI_EXC = tuple()

# Together SDK typically wraps http errors; we rely on httpx and timeouts
RETRYABLE_EXC = (httpx.HTTPError, TimeoutError, asyncio.TimeoutError) + OPENAI_EXC

MAX_RETRIES = int(os.getenv("ARXITEX_LLM_MAX_RETRIES", "4"))


def _before_sleep(retry_state):
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(f"LLM retry #{retry_state.attempt_number} after error: {exc}")


retry_sync = retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXC),
    before_sleep=_before_sleep,
)

retry_async = retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXC),
    before_sleep=_before_sleep,
)
