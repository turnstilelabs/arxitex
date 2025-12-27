from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from loguru import logger

from arxitex.downloaders.async_downloader import AsyncSourceDownloader
from arxitex.downloaders.utils import read_and_combine_tex_files
from arxitex.extractor.graph_building.graph_enhancer import GraphEnhancer

SseEvent = Dict[str, Any]


async def astream_artifact_graph(
    *,
    arxiv_id: str,
    infer_dependencies: bool,
    enrich_content: bool,
    source_dir: Optional[Path] = None,
) -> AsyncIterator[SseEvent]:
    """Stream graph construction in strict phases.

    Phase 1: regex base nodes/edges
    Phase 2: enrichment -> node upserts
    Phase 3: dependency inference -> edge additions

    Events use the same frontend vocabulary:
    - {type: 'node', data: <node dict>} (upsert)
    - {type: 'link', data: <edge dict>} (add)
    - {type: 'status', data: <string>}
    - {type: 'done'}
    """

    log_location = source_dir if source_dir else "system's default temp directory"
    logger.debug(f"[{arxiv_id}] Creating temporary directory inside: {log_location}")

    with tempfile.TemporaryDirectory(
        prefix=f"{arxiv_id.replace('/', '_')}_", dir=source_dir
    ) as temp_dir:
        temp_path = Path(temp_dir)

        async with AsyncSourceDownloader(cache_dir=temp_path) as downloader:
            project_dir = await downloader.download_and_extract_source(arxiv_id)

            if not project_dir:
                yield {
                    "type": "status",
                    "data": f"Failed to retrieve LaTeX content for {arxiv_id}",
                }
                return

            latex_content = read_and_combine_tex_files(project_dir)

            enhancer = GraphEnhancer()

            # --- Phase 1: base graph ---
            yield {
                "type": "status",
                "data": "Pass 1: Extracting base artifacts (regex)",
            }
            base_graph = enhancer.regex_builder.build_graph(
                project_dir, source_file=f"arxiv:{arxiv_id}"
            )

            if not base_graph.nodes:
                yield {
                    "type": "status",
                    "data": "Regex pass found no artifacts. Aborting.",
                }
                yield {"type": "done"}
                return

            for n in base_graph.nodes:
                yield {"type": "node", "data": n.to_dict()}
            for e in base_graph.edges:
                yield {"type": "link", "data": e.to_dict()}

            # --- Phase 2: enrichment ---
            bank = None
            artifact_to_terms_map: dict[str, list[str]] = {}

            should_enrich = enrich_content or infer_dependencies
            if should_enrich:
                yield {
                    "type": "status",
                    "data": "Pass 2: Enriching artifact content (LLM)",
                }

                yield_queue: asyncio.Queue[SseEvent] = asyncio.Queue()

                async def on_artifact_enriched(
                    artifact_id: str, prerequisite_defs: dict[str, str]
                ):
                    node = base_graph.get_node_by_id(artifact_id)
                    if not node:
                        return
                    node.prerequisite_defs = prerequisite_defs
                    yield_queue.put_nowait({"type": "node", "data": node.to_dict()})

                enrichment_task = asyncio.create_task(
                    enhancer.document_enhancer.enhance_document(
                        [n for n in base_graph.nodes if not n.is_external],
                        latex_content,
                        use_global_extraction=True,
                        on_artifact_enhanced=on_artifact_enriched,
                    )
                )

                while True:
                    if enrichment_task.done() and yield_queue.empty():
                        break
                    try:
                        ev = await asyncio.wait_for(yield_queue.get(), timeout=0.25)
                        yield ev
                    except asyncio.TimeoutError:
                        # keep-alive is handled by the FastAPI stream wrapper
                        continue

                enrichment_results = await enrichment_task
                bank = enrichment_results.get("definition_bank")
                artifact_to_terms_map = enrichment_results.get(
                    "artifact_to_terms_map", {}
                )

                # Ensure final node states are emitted (in case some callback events were missed)
                defs_map = enrichment_results.get("definitions_map", {})
                for node_id, prereqs in defs_map.items():
                    node = base_graph.get_node_by_id(node_id)
                    if not node:
                        continue
                    node.prerequisite_defs = prereqs
                    yield {"type": "node", "data": node.to_dict()}
            else:
                yield {
                    "type": "status",
                    "data": "Pass 2: Skipped (enrichment disabled)",
                }

            # --- Phase 3: dependency inference ---
            if infer_dependencies:
                if not artifact_to_terms_map:
                    yield {
                        "type": "status",
                        "data": "Pass 3: Skipped (no term map available)",
                    }
                else:
                    yield {
                        "type": "status",
                        "data": "Pass 3: Inferring dependencies (LLM)",
                    }

                    async for edge_dict in enhancer.astream_dependency_edges(
                        graph=base_graph,
                        artifact_to_terms_map=artifact_to_terms_map,
                        bank=bank,
                    ):
                        yield {"type": "link", "data": edge_dict}

            yield {"type": "done"}
