from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from arxitex.downloaders.async_downloader import (
    ArxivExtractorError as DownloaderArxivExtractorError,
)
from arxitex.extractor.models import ArxivExtractorError as ModelArxivExtractorError

# Best-effort import of OpenAI SDK exceptions (available in openai >=1.x).
# If the SDK is not installed or changes, we simply won't classify those
# errors specially and will fall back to a generic bucket.
try:  # pragma: no cover - defensive import
    from openai import (  # type: ignore
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )

    OPENAI_EXC: tuple[type[BaseException], ...] = (
        APIConnectionError,
        APIError,
        RateLimitError,
        APITimeoutError,
    )
except Exception:  # pragma: no cover
    OPENAI_EXC = tuple()


@dataclass
class ErrorInfo:
    """Normalized information about a processing failure.

    This is what we store into processed_papers.details and return in
    workflow summaries so failures are easy to aggregate and inspect.
    """

    code: str
    message: str
    stage: str
    exception_type: str

    def to_details_dict(self) -> dict[str, Any]:
        """Convert to a dict that can be merged into ProcessedIndex.details."""

        return {
            "reason": self.message,
            "reason_code": self.code,
            "error_stage": self.stage,
            "exception_type": self.exception_type,
        }


def classify_processing_error(exc: Exception) -> ErrorInfo:
    """Map a raw exception to a structured ErrorInfo.

    We intentionally keep this logic conservative and based on
    exception type + well-known message fragments so that it is
    resilient to internal refactors.
    """

    msg = str(exc) or exc.__class__.__name__
    etype = exc.__class__.__name__
    lower_msg = msg.lower()

    # --- Extractor / downloader errors ---
    if isinstance(exc, (ModelArxivExtractorError, DownloaderArxivExtractorError)):
        # PDF-only: no LaTeX source available.
        if "pdf-only" in lower_msg or "pdf only" in lower_msg:
            return ErrorInfo(
                code="no_latex_source",
                message="Paper is PDF-only; LaTeX source is required to build a graph.",
                stage="extract",
                exception_type=etype,
            )

        # Download exhausted retries.
        if "failed to download source" in lower_msg:
            return ErrorInfo(
                code="source_download_failed",
                message="Failed to download LaTeX source from arXiv after multiple retries.",
                stage="download",
                exception_type=etype,
            )

        # Withdrawn papers: no source is available.
        if "withdrawn" in lower_msg:
            return ErrorInfo(
                code="paper_withdrawn",
                message="Paper is withdrawn on arXiv; source archive is unavailable.",
                stage="download",
                exception_type=etype,
            )

        # Blocked by arXiv anti-bot.
        if "recaptcha" in lower_msg:
            return ErrorInfo(
                code="source_blocked_by_recaptcha",
                message=(
                    "arXiv returned a reCAPTCHA challenge page instead of the source archive. "
                    "Try again later, reduce request rate, or download manually."
                ),
                stage="download",
                exception_type=etype,
            )

        # Archive looks corrupt or unknown.
        if "gzip archive is corrupted" in lower_msg:
            return ErrorInfo(
                code="source_gzip_corrupt",
                message="Downloaded gzip archive is corrupted and cannot be decompressed.",
                stage="extract",
                exception_type=etype,
            )

        if "tar archive is corrupted" in lower_msg:
            return ErrorInfo(
                code="source_tar_corrupt",
                message="Downloaded tar archive is corrupted and cannot be read.",
                stage="extract",
                exception_type=etype,
            )

        if "zip archive is corrupted" in lower_msg:
            return ErrorInfo(
                code="source_zip_corrupt",
                message="Downloaded ZIP archive is corrupted and cannot be read.",
                stage="extract",
                exception_type=etype,
            )

        if "unable to extract or identify downloaded file format" in lower_msg:
            return ErrorInfo(
                code="source_extract_failed",
                message="Unable to extract or identify the downloaded source archive.",
                stage="extract",
                exception_type=etype,
            )

        # Fallback for other extractor errors.
        return ErrorInfo(
            code="extractor_error",
            message=msg,
            stage="extract",
            exception_type=etype,
        )

    # --- Bad arXiv IDs ---
    if isinstance(exc, ValueError) and "invalid arxiv id format" in lower_msg:
        return ErrorInfo(
            code="invalid_arxiv_id",
            message=msg,
            stage="download",
            exception_type=etype,
        )

    # --- Empty or invalid graphs ---
    if isinstance(exc, ValueError) and (
        "empty graph" in lower_msg or "empty or invalid graph" in lower_msg
    ):
        return ErrorInfo(
            code="graph_empty",
            message="Graph generation resulted in no artifacts; the LaTeX source contained no detectable statements.",
            stage="graph_build",
            exception_type=etype,
        )

    # --- LLM / API related errors ---
    if OPENAI_EXC and isinstance(exc, OPENAI_EXC):
        if "rate limit" in lower_msg:
            return ErrorInfo(
                code="llm_rate_limited",
                message="LLM call was rate-limited by the provider.",
                stage="llm",
                exception_type=etype,
            )
        if "timeout" in lower_msg:
            return ErrorInfo(
                code="llm_timeout",
                message="LLM call timed out while waiting for a response.",
                stage="llm",
                exception_type=etype,
            )
        return ErrorInfo(
            code="llm_api_error",
            message="LLM provider returned an API error: " + msg,
            stage="llm",
            exception_type=etype,
        )

    # httpx / timeout-based errors (Together or generic HTTP failures).
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)):
        return ErrorInfo(
            code="llm_timeout",
            message="LLM or HTTP call timed out while waiting for a response.",
            stage="llm",
            exception_type=etype,
        )

    if isinstance(exc, httpx.HTTPError):
        return ErrorInfo(
            code="llm_connection_error",
            message="HTTP error while calling LLM or external service: " + msg,
            stage="llm",
            exception_type=etype,
        )

    # --- Fallback ---
    return ErrorInfo(
        code="unexpected_error",
        message=msg,
        stage="unknown",
        exception_type=etype,
    )
