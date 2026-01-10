from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from arxitex.extractor.models import ArxivExtractorError
from arxitex.extractor.streaming import astream_artifact_graph


class ProcessPaperRequest(BaseModel):
    arxivUrl: str = Field(
        ..., description="Full arXiv URL, e.g. https://arxiv.org/abs/2211.11689"
    )
    inferDependencies: bool = Field(
        True, description="Enable LLM dependency inference between artifacts"
    )
    enrichContent: bool = Field(
        True, description="Enable LLM definition/symbol extraction and enrichment"
    )


def _extract_arxiv_id(arxiv_url: str) -> str:
    # Supports:
    # - https://arxiv.org/abs/2211.11689
    # - https://arxiv.org/abs/2211.11689v1
    # - https://arxiv.org/abs/math.AG/0601001
    # - https://arxiv.org/pdf/2211.11689.pdf
    match = re.search(
        r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?)",
        arxiv_url,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not extract arXiv ID from URL")
    return match.group(1)


def _require_openai_key_if_needed(
    *, infer_dependencies: bool, enrich_content: bool
) -> None:
    if not (infer_dependencies or enrich_content):
        return

    if os.getenv("OPENAI_API_KEY"):
        return

    # The repo supports other providers in some places, but for this webapp we
    # standardize on OpenAI (per project requirement).
    raise HTTPException(
        status_code=400,
        detail=(
            "Enhancements requested but OPENAI_API_KEY is not set. "
            "Either disable enhancements or export OPENAI_API_KEY."
        ),
    )


def _sse_data(payload: dict[str, Any]) -> bytes:
    # Only using `data:` lines because the frontend expects to parse `data: {...}`.
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


app = FastAPI(title="ArxiTex Backend", version="0.1.0")

# ---------------------------------------------------------------------------
# CORS
#
# We need CORS for the Next.js frontend to call this backend directly.
#
# Goals:
# - Dev: allow any localhost port (so you can run Next on 3000/3001/3006/etc).
# - Prod: allow explicit origins via env vars (so we don't accidentally ship
#   a too-permissive policy).
#
# Env vars (preferred in prod):
# - ARXITEX_CORS_ALLOW_ORIGINS="https://app.example.com,https://staging.example.com"
# - ARXITEX_CORS_ALLOW_ORIGIN_REGEX="https://.*\\.example\\.com"
# ---------------------------------------------------------------------------


def _cors_allow_origins_from_env() -> list[str] | None:
    raw = (os.getenv("ARXITEX_CORS_ALLOW_ORIGINS") or "").strip()
    if not raw:
        return None
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or None


def _cors_allow_origin_regex_from_env() -> str | None:
    raw = (os.getenv("ARXITEX_CORS_ALLOW_ORIGIN_REGEX") or "").strip()
    return raw or None


cors_allow_origins = _cors_allow_origins_from_env()
cors_allow_origin_regex = _cors_allow_origin_regex_from_env()

if cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Safe dev default: allow any localhost origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=cors_allow_origin_regex
        # IMPORTANT: this is a *raw regex string*; do not over-escape.
        # We want to match e.g. "http://localhost:3000".
        or r"https?://(localhost|127\.0\.0\.1):\d+",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process-paper")
async def process_paper(req: ProcessPaperRequest):
    try:
        arxiv_id = _extract_arxiv_id(req.arxivUrl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    _require_openai_key_if_needed(
        infer_dependencies=req.inferDependencies,
        enrich_content=req.enrichContent,
    )

    async def event_stream() -> AsyncIterator[bytes]:
        # Initial status
        yield _sse_data(
            {"type": "status", "data": f"Starting processing for {arxiv_id}..."}
        )

        queue: asyncio.Queue[bytes] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for ev in astream_artifact_graph(
                    arxiv_id=arxiv_id,
                    infer_dependencies=req.inferDependencies,
                    enrich_content=req.enrichContent,
                    source_dir=None,
                ):
                    queue.put_nowait(_sse_data(ev))
            except (ArxivExtractorError, FileNotFoundError, ValueError) as e:
                queue.put_nowait(
                    _sse_data({"type": "status", "data": f"Processing error: {e}"})
                )
            except Exception as e:  # pragma: no cover
                queue.put_nowait(
                    _sse_data({"type": "status", "data": f"Unexpected error: {e}"})
                )

        task = asyncio.create_task(produce())

        while True:
            if task.done() and queue.empty():
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.25)
                yield chunk
            except asyncio.TimeoutError:
                yield b": keep-alive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _extractor_mode(*, infer_dependencies: bool, enrich_content: bool) -> str:
    if infer_dependencies and enrich_content:
        return "full-hybrid (deps + content)"
    if infer_dependencies:
        return "hybrid (deps-only)"
    if enrich_content:
        return "hybrid (content-only)"
    return "regex-only"
