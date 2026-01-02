from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from loguru import logger

from arxitex.db.error_utils import classify_processing_error
from arxitex.extractor.models import ArxivExtractorError
from arxitex.extractor.pipeline import agenerate_artifact_graph

SseEvent = Dict[str, Any]


async def astream_artifact_graph(
    *,
    arxiv_id: str,
    infer_dependencies: bool,
    enrich_content: bool,
    source_dir: Optional[Path] = None,
) -> AsyncIterator[SseEvent]:
    """Stream graph construction events for a single arXiv paper.

    This implementation now delegates to ``agenerate_artifact_graph`` which
    orchestrates all passes (regex base graph, enrichment, dependency
    inference) via ``GraphEnhancer.build_graph``.

    We preserve the SSE vocabulary expected by the frontend:

    - {"type": "node", "data": <node dict>}  (upsert)
    - {"type": "link", "data": <edge dict>}  (add)
    - {"type": "status", "data": <string>}   (status/progress message)
    - {"type": "error", "data": <error dict>} (terminal error with details)
    - {"type": "done"}                         (stream finished)

    Note: Unlike the earlier version, we no longer stream *intermediate*
    LLM updates during enrichment/dependency inference. Instead, we run the
    high-level pipeline once and then stream the resulting nodes and edges.
    """

    yield {
        "type": "status",
        "data": (
            f"Starting extraction for {arxiv_id} | "
            f"infer_deps={infer_dependencies}, enrich_content={enrich_content}"
        ),
    }

    try:
        results = await agenerate_artifact_graph(
            arxiv_id=arxiv_id,
            infer_dependencies=infer_dependencies,
            enrich_content=enrich_content,
            # Use the new mode-aware dependency inference by default.
            dependency_mode="auto",
            dependency_config=None,
            source_dir=source_dir,
        )
    except ArxivExtractorError as e:
        # Expected, user-facing extractor failure (e.g. PDF-only, bad archive).
        logger.error(f"A processing error occurred while building graph: {e}")
        err = classify_processing_error(e)
        yield {"type": "error", "data": err.to_details_dict()}
        yield {"type": "done"}
        return
    except Exception as e:  # pragma: no cover - defensive
        # Unexpected failure; still surface a structured error so the UI can
        # show a friendly message while logs retain full details.
        logger.error(
            f"An unexpected error occurred while building graph for {arxiv_id}: {e}",
            exc_info=True,
        )
        err = classify_processing_error(e)
        yield {"type": "error", "data": err.to_details_dict()}
        yield {"type": "done"}
        return

    graph = results.get("graph")

    if not graph or not graph.nodes:
        # Normalize "no artifacts" into a structured error using the same
        # taxonomy as other pipeline failures.
        err = classify_processing_error(ValueError("empty graph"))
        yield {"type": "error", "data": err.to_details_dict()}
        yield {"type": "done"}
        return

    # Stream all nodes first
    for node in graph.nodes:
        yield {"type": "node", "data": node.to_dict()}

    # Then stream all edges (both reference-based and dependency-based)
    for edge in graph.edges:
        yield {"type": "link", "data": edge.to_dict()}

    yield {"type": "status", "data": "Graph extraction complete."}
    yield {"type": "done"}
