from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from loguru import logger

from arxitex.db.error_utils import classify_processing_error
from arxitex.extractor.models import (
    ArtifactNode,
    ArxivExtractorError,
    DocumentGraph,
    Edge,
)
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

    Streaming is driven by callbacks from GraphEnhancer.build_graph:
    - base graph (regex-only) is sent first (all nodes, then reference edges)
    - enrichment can emit updated nodes
    - dependency inference can emit new dependency edges
    """

    yield {
        "type": "status",
        "data": (
            f"Starting extraction for {arxiv_id} | "
            f"infer_deps={infer_dependencies}, enrich_content={enrich_content}"
        ),
    }

    # Use a queue so events can be emitted as they become available while
    # the high-level pipeline is running.
    queue: asyncio.Queue[SseEvent] = asyncio.Queue()

    async def on_base_graph(graph: DocumentGraph) -> None:
        """Emit the initial regex-only graph snapshot.

        We send all nodes, then all reference/internal edges. Dependency
        edges will be streamed later via on_dependency_edge.
        """

        for node in graph.nodes:
            await queue.put({"type": "node", "data": node.to_dict()})
        for edge in graph.edges:
            await queue.put({"type": "link", "data": edge.to_dict()})

    async def on_enriched_node(node: ArtifactNode) -> None:
        """Emit updated node content as enrichment adds prerequisite defs."""

        await queue.put({"type": "node", "data": node.to_dict()})

    async def on_dependency_edge(edge: Edge) -> None:
        """Emit dependency edges incrementally during inference."""

        await queue.put({"type": "link", "data": edge.to_dict()})

    async def on_status(message: str) -> None:
        """Emit high-level stage status updates (base graph, enrichment, deps)."""

        await queue.put({"type": "status", "data": message})

    async def produce() -> None:
        try:
            results = await agenerate_artifact_graph(
                arxiv_id=arxiv_id,
                infer_dependencies=infer_dependencies,
                enrich_content=enrich_content,
                # Use the new mode-aware dependency inference by default.
                dependency_mode="auto",
                dependency_config=None,
                source_dir=source_dir,
                on_base_graph=on_base_graph,
                on_enriched_node=on_enriched_node,
                on_dependency_edge=on_dependency_edge,
                on_status=on_status,
            )
            # Optionally emit the definition bank at the end of the stream,
            # if enrichment was enabled and a non-empty bank is available.
            bank = results.get("bank")
            if bank is not None and enrich_content:
                try:
                    bank_dict = await bank.to_dict()
                except Exception as e:  # pragma: no cover - defensive
                    logger.error(
                        "Failed to serialize definition bank for %s: %s",
                        arxiv_id,
                        e,
                    )
                    bank_dict = None

                if bank_dict:
                    await queue.put(
                        {
                            "type": "definition_bank",
                            "data": bank_dict,
                        }
                    )
        except ArxivExtractorError as e:
            logger.error(f"A processing error occurred while building graph: {e}")
            err = classify_processing_error(e)
            await queue.put({"type": "error", "data": err.to_details_dict()})
            await queue.put({"type": "done"})
            return
        except Exception as e:  # pragma: no cover - defensive
            logger.error(
                f"An unexpected error occurred while building graph for {arxiv_id}: {e}",
                exc_info=True,
            )
            err = classify_processing_error(e)
            await queue.put({"type": "error", "data": err.to_details_dict()})
            await queue.put({"type": "done"})
            return

        graph = results.get("graph")

        if not graph or not graph.nodes:
            err = classify_processing_error(ValueError("empty graph"))
            await queue.put({"type": "error", "data": err.to_details_dict()})
            await queue.put({"type": "done"})
            return

        await queue.put({"type": "status", "data": "Graph extraction complete."})
        await queue.put({"type": "done"})

    # Launch producer in the background.
    task = asyncio.create_task(produce())

    # Consume from the queue as events become available.
    while True:
        if task.done() and queue.empty():
            break
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.25)
            yield event
        except asyncio.TimeoutError:
            # The FastAPI layer already emits an SSE comment keep-alive.
            # Avoid pushing extra JSON status events to reduce client churn.
            continue
