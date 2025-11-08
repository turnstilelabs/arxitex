import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Literal, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from arxitex.arxiv_api import ArxivAPI
from arxitex.downloaders.async_downloader import AsyncSourceDownloader
from arxitex.downloaders.utils import read_and_combine_tex_files

# Streaming/build helpers
from arxitex.extractor.graph_building.base_builder import BaseGraphBuilder
from arxitex.extractor.graph_building.graph_enhancer import GraphEnhancer

# Reuse your pipeline
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.definition_builder.definition_builder import DefinitionBuilder
from arxitex.symdef.document_enhancer import DocumentEnhancer
from arxitex.symdef.utils import ContextFinder, Definition
from arxitex.workflows.runner import ArxivPipelineComponents
from arxitex.workflows.utils import save_graph_data

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", PROJECT_ROOT / "pipeline_output")
).resolve()
GRAPHS_DIR = DEFAULT_OUTPUT_DIR / "graphs"
BANKS_DIR = DEFAULT_OUTPUT_DIR / "definition_banks"
LOGS_DIR = DEFAULT_OUTPUT_DIR / "logs"

for d in (GRAPHS_DIR, BANKS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class IngestRequest(BaseModel):
    arxiv_url_or_id: str = Field(
        ...,
        description="ArXiv URL or ID (e.g. https://arxiv.org/abs/2211.11689 or 2211.11689)",
    )
    infer_dependencies: bool = Field(
        False, description="Enable LLM dependency inference"
    )
    enrich_content: bool = Field(False, description="Enable LLM content enrichment")
    force: bool = Field(False, description="Force re-processing even if cached")


JobStatus = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    job_id: str
    arxiv_id: str
    status: JobStatus
    progress: int = 0
    stage: str = "queued"
    error: Optional[str] = None
    extractor_mode: Optional[str] = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def extract_arxiv_id(id_or_url: str) -> str:
    """
    Robustly extract an arXiv ID from a variety of inputs:
    - Bare IDs: "2211.11689", "2211.11689v1", "math.AG/0601001"
    - ABS URLs: "https://arxiv.org/abs/2211.11689v1"
    - PDF URLs: "https://arxiv.org/pdf/2211.11689.pdf", ".../2211.11689v1.pdf"
    - Fallback: last path segment with optional '.pdf' stripped.
    """
    s = id_or_url.strip()
    # Fast path for plain IDs (no URL scheme and no slashes)
    if "://" not in s and "/" not in s:
        return s

    try:
        parsed = urlparse(s)
        path = (parsed.path or "").strip()

        # Standard patterns
        if "/abs/" in path:
            return path.split("/abs/")[-1].strip("/")

        if "/pdf/" in path:
            candidate = path.split("/pdf/")[-1].strip("/")
            if candidate.endswith(".pdf"):
                candidate = candidate[:-4]
            return candidate

        # Fallback: last segment, strip .pdf if present
        seg = path.strip("/").split("/")[-1]
        return seg[:-4] if seg.endswith(".pdf") else seg
    except Exception:
        # Last resort: naive split
        tail = s.rsplit("/", 1)[-1]
        return tail[:-4] if tail.endswith(".pdf") else tail


def compute_extractor_mode(infer_dependencies: bool, enrich_content: bool) -> str:
    if infer_dependencies and enrich_content:
        return "full-hybrid (deps + content)"
    if infer_dependencies:
        return "hybrid (deps-only)"
    if enrich_content:
        return "hybrid (content-only)"
    return "regex-only"


# -----------------------------------------------------------------------------
# Job Store (in-memory for dev; switch to Redis in prod)
# -----------------------------------------------------------------------------
class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job

    async def update(self, job_id: str, **fields) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            self._jobs[job_id] = job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)


JOB_STORE = InMemoryJobStore()


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="ArxiTex API", version="1.0.0")

# CORS (open for dev; tighten for prod)
# CORS (dev-friendly defaults). If ALLOWED_ORIGINS is set, use it (CSV of origins).
# Otherwise, allow typical local dev hosts via a regex while keeping credentials support.
origins_env = os.getenv("ALLOWED_ORIGINS")
default_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
if origins_env:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins_env.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],  # when credentials=True, '*' is not permitted; use regex instead
        allow_origin_regex=default_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# -----------------------------------------------------------------------------
# Core worker
# -----------------------------------------------------------------------------
async def _run_ingest_job(
    job_id: str,
    arxiv_id: str,
    output_dir: Path,
    enrich_content: bool,
    infer_dependencies: bool,
    force: bool,
) -> None:
    components = ArxivPipelineComponents(output_dir=str(output_dir))

    # Short-circuit if already processed and present on disk, unless forced.
    if components.processing_index.is_successfully_processed(arxiv_id) and not force:
        await JOB_STORE.update(job_id, status="done", progress=100, stage="cached")
        return

    await JOB_STORE.update(job_id, status="running", stage="downloading", progress=5)

    try:
        # run pipeline
        await JOB_STORE.update(job_id, stage="building_graph", progress=35)
        results = await agenerate_artifact_graph(
            arxiv_id=arxiv_id,
            enrich_content=enrich_content,
            infer_dependencies=infer_dependencies,
            source_dir=output_dir / "temp_processing",
        )

        graph = results.get("graph")
        bank = results.get("bank")

        if not graph or not graph.nodes:
            raise ValueError("Graph generation resulted in an empty or invalid graph.")

        extractor_mode = compute_extractor_mode(
            infer_dependencies=infer_dependencies, enrich_content=enrich_content
        )

        await JOB_STORE.update(job_id, stage="serializing", progress=70)

        # Serialize graph
        graph_data = graph.to_dict(arxiv_id=arxiv_id, extractor_mode=extractor_mode)
        graph_filepath = save_graph_data(arxiv_id, str(GRAPHS_DIR), graph_data)

        # Serialize definition bank if available
        if bank:
            bank_data = await bank.to_dict()
            bank_path = BANKS_DIR / f"{arxiv_id.replace('/', '_')}_bank.json"
            bank_path.write_text(
                json.dumps(bank_data, ensure_ascii=False), encoding="utf-8"
            )

        # Update processing index
        components.processing_index.update_processed_papers_status(
            arxiv_id,
            status="success",
            output_path=str(graph_filepath),
            stats=graph_data.get("stats", {}),
        )

        await JOB_STORE.update(
            job_id,
            status="done",
            progress=100,
            stage="completed",
            extractor_mode=extractor_mode,
        )

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        components.processing_index.update_processed_papers_status(
            arxiv_id, status="failure", reason=str(e)
        )
        await JOB_STORE.update(
            job_id, status="failed", progress=100, stage="failed", error=str(e)
        )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/api/v1/papers/ingest", response_model=Job, status_code=202)
async def ingest(req: IngestRequest) -> Job:
    arxiv_id = extract_arxiv_id(req.arxiv_url_or_id)
    job_id = uuid.uuid4().hex

    components = ArxivPipelineComponents(output_dir=str(DEFAULT_OUTPUT_DIR))

    # Validate that server-side LLM credentials exist when requested from the client.
    # This prevents firing an async job that will fail later due to missing API keys.
    if (req.enrich_content or req.infer_dependencies) and not (
        os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
    ):
        raise HTTPException(
            status_code=400,
            detail="LLM features require a server-side API key. Set OPENAI_API_KEY or TOGETHER_API_KEY.",
        )

    # If cached and not forced, return a "done" job immediately.
    # Verify the graph file actually exists; if the processing index claims success but the file is missing,
    # fall through and re-run the pipeline (this covers cases where files were deleted but the index wasn't).
    graph_path = GRAPHS_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if (
        not req.force
        and components.processing_index.is_successfully_processed(arxiv_id)
        and graph_path.exists()
    ):
        job = Job(
            job_id=job_id,
            arxiv_id=arxiv_id,
            status="done",
            progress=100,
            stage="cached",
            extractor_mode=compute_extractor_mode(
                req.infer_dependencies, req.enrich_content
            ),
        )
        await JOB_STORE.create(job)
        return job

    job = Job(
        job_id=job_id, arxiv_id=arxiv_id, status="queued", progress=0, stage="queued"
    )
    await JOB_STORE.create(job)

    # Fire and forget the async worker
    asyncio.create_task(
        _run_ingest_job(
            job_id=job_id,
            arxiv_id=arxiv_id,
            output_dir=DEFAULT_OUTPUT_DIR,
            enrich_content=req.enrich_content,
            infer_dependencies=req.infer_dependencies,
            force=req.force,
        )
    )

    return job


@app.get("/api/v1/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str) -> Job:
    job = await JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/papers/{arxiv_id}")
async def get_paper(arxiv_id: str):
    graph_path = GRAPHS_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not graph_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Paper not processed yet. POST /api/v1/papers/ingest to start.",
        )

    try:
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception(f"Failed to read graph for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read graph file")

    bank_path = BANKS_DIR / f"{arxiv_id.replace('/', '_')}_bank.json"
    bank_data = None
    if bank_path.exists():
        try:
            bank_data = json.loads(bank_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read definition bank for {arxiv_id}: {e}")

    # Fetch basic metadata (title, authors) from arXiv
    title = None
    authors = []
    try:
        api = ArxivAPI()
        resp = api.session.get(api.base_url, params={"id_list": arxiv_id}, timeout=30)
        resp.raise_for_status()
        _, _, entries = api.parse_response(resp.text)
        if entries:
            paper_meta = api.entry_to_paper(entries[0])
            if paper_meta:
                title = paper_meta.get("title")
                authors = paper_meta.get("authors", []) or []
        api.close()
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {arxiv_id}: {e}")

    return {
        "arxiv_id": arxiv_id,
        "graph": graph_data,
        "definition_bank": bank_data,
        "title": title,
        "authors": authors,
    }


@app.get("/api/v1/papers/{arxiv_id}/graph")
async def get_paper_graph(arxiv_id: str):
    graph_path = GRAPHS_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    try:
        return json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception(f"Failed to read graph for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read graph file")


@app.get("/api/v1/papers/{arxiv_id}/definitions")
async def get_paper_definitions(arxiv_id: str):
    bank_path = BANKS_DIR / f"{arxiv_id.replace('/', '_')}_bank.json"
    if not bank_path.exists():
        raise HTTPException(status_code=404, detail="Definition bank not found")
    try:
        return json.loads(bank_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception(f"Failed to read definition bank for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read definition bank")


@app.get("/api/v1/papers/{arxiv_id}/stream-build")
async def stream_build(
    arxiv_id: str,
    infer_dependencies: bool = False,
    enrich_content: bool = False,
    force: bool = False,
):
    """
    Streams the build process via Server-Sent Events (SSE).
    Events:
      - nodes_seeded: { graph }
      - term_inferred: { term, aliases, definition_text, source_artifact_id, dependencies }
      - prerequisite_link: { artifact_id, term }
      - progress: { stage }
      - error: { message }
      - done: { status }
    """

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(event_type: str, payload: dict):
            await queue.put((event_type, payload))

        async def sse(event_type: str, payload: dict) -> bytes:
            return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            )

        async def worker():
            components = ArxivPipelineComponents(output_dir=str(DEFAULT_OUTPUT_DIR))
            try:
                # Download source
                await emit("progress", {"stage": "downloading"})
                async with AsyncSourceDownloader(
                    cache_dir=DEFAULT_OUTPUT_DIR / "temp_stream"
                ) as downloader:
                    project_dir = await downloader.download_and_extract_source(arxiv_id)
                    if not project_dir:
                        raise RuntimeError(
                            f"Failed to retrieve LaTeX content for {arxiv_id}"
                        )

                    # Fast regex/base graph
                    builder = BaseGraphBuilder()
                    graph = builder.build_graph(
                        project_dir=project_dir, source_file=f"arxiv:{arxiv_id}"
                    )
                    extractor_mode = compute_extractor_mode(
                        infer_dependencies=infer_dependencies,
                        enrich_content=enrich_content,
                    )
                    graph_data = graph.to_dict(
                        arxiv_id=arxiv_id, extractor_mode=extractor_mode
                    )
                    await emit("nodes_seeded", {"graph": graph_data})

                    has_llm = bool(
                        os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
                    )
                    if (infer_dependencies or enrich_content) and not has_llm:
                        await emit(
                            "error",
                            {
                                "message": "LLM features requested but no server-side API key is configured."
                            },
                        )
                        # Save regex-only graph and finish
                        save_graph_data(arxiv_id, str(GRAPHS_DIR), graph_data)
                        await emit("done", {"status": "regex-only"})
                        return

                    # Build definition bank and enriched content with streaming callbacks
                    latex_content = read_and_combine_tex_files(project_dir)

                    async def on_term_inferred(defn: Definition):
                        try:
                            await emit(
                                "term_inferred",
                                {
                                    "term": defn.term,
                                    "aliases": defn.aliases,
                                    "definition_text": getattr(
                                        defn, "definition_text", "N/A"
                                    ),
                                    "source_artifact_id": defn.source_artifact_id,
                                    "dependencies": getattr(defn, "dependencies", []),
                                },
                            )
                        except Exception as e:
                            logger.warning(f"term_inferred emit failed: {e}")

                    async def on_prereq_link(artifact_id: str, term: str):
                        try:
                            await emit(
                                "prerequisite_link",
                                {"artifact_id": artifact_id, "term": term},
                            )
                        except Exception as e:
                            logger.warning(f"prerequisite_link emit failed: {e}")

                    definition_builder = DefinitionBuilder()
                    context_finder = ContextFinder()
                    bank = DefinitionBank()
                    enhancer = DocumentEnhancer(
                        llm_enhancer=definition_builder,
                        context_finder=context_finder,
                        definition_bank=bank,
                        on_term_inferred=on_term_inferred,
                        on_prerequisite_link=on_prereq_link,
                    )

                    await emit("progress", {"stage": "enriching_content"})
                    nodes_to_enhance = [n for n in graph.nodes if not n.is_external]
                    enhanced = await enhancer.enhance_document(
                        nodes_to_enhance, latex_content
                    )
                    definitions_map = enhanced.get("definitions_map", {})
                    artifact_to_terms_map = enhanced.get("artifact_to_terms_map", {})
                    bank = enhanced.get("definition_bank", bank)

                    # Put prerequisite_defs back into nodes for serialization
                    for node in graph.nodes:
                        if node.id in definitions_map:
                            node.prerequisite_defs = definitions_map[node.id]

                    if infer_dependencies:
                        await emit("progress", {"stage": "dependency_inference"})
                        ge = GraphEnhancer()

                        async def _on_dep_edge(e):
                            try:
                                await emit(
                                    "dependency_edge",
                                    {
                                        "source_id": e.source_id,
                                        "target_id": e.target_id,
                                        "dependency_type": e.dependency_type,
                                        "dependency": getattr(e, "dependency", None),
                                        "context": getattr(e, "context", None),
                                    },
                                )
                            except Exception as ex:
                                logger.warning(f"dependency_edge emit failed: {ex}")

                        async def _on_dep_progress(done: int, total: int):
                            try:
                                await emit(
                                    "dependency_progress",
                                    {"processed": done, "total": total},
                                )
                            except Exception as ex:
                                logger.warning(f"dependency_progress emit failed: {ex}")

                        graph = await ge.ainfer_dependencies_streaming(
                            graph,
                            artifact_to_terms_map,
                            bank,
                            on_edge=_on_dep_edge,
                            on_progress=_on_dep_progress,
                        )

                    await emit("progress", {"stage": "serializing"})
                    # Serialize graph
                    graph_data = graph.to_dict(
                        arxiv_id=arxiv_id, extractor_mode=extractor_mode
                    )
                    graph_filepath = save_graph_data(
                        arxiv_id, str(GRAPHS_DIR), graph_data
                    )

                    # Serialize definition bank
                    bank_dict = await bank.to_dict()
                    if bank_dict:
                        bank_path = (
                            BANKS_DIR / f"{arxiv_id.replace('/', '_')}_bank.json"
                        )
                        bank_path.write_text(
                            json.dumps(bank_dict, ensure_ascii=False),
                            encoding="utf-8",
                        )

                    # Update processing index
                    components.processing_index.update_processed_papers_status(
                        arxiv_id,
                        status="success",
                        output_path=str(graph_filepath),
                        stats=graph_data.get("stats", {}),
                    )

                    await emit("done", {"status": "completed"})
            except Exception as e:
                logger.exception(f"SSE build failed for {arxiv_id}: {e}")
                try:
                    components.processing_index.update_processed_papers_status(
                        arxiv_id, status="failure", reason=str(e)
                    )
                except Exception:
                    pass
                await emit("error", {"message": str(e)})
                await emit("done", {"status": "failed"})

        task = asyncio.create_task(worker())

        try:
            while True:
                evt, data = await queue.get()
                yield await sse(evt, data)
                if evt == "done":
                    break
        finally:
            task.cancel()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )


@app.get("/api/v1/llm/status")
async def llm_status():
    """
    Returns whether LLM features are available on the server and which providers are configured.
    Example response: {"available": true, "providers": ["openai"]}
    """
    available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY"))
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("TOGETHER_API_KEY"):
        providers.append("together")
    return {"available": available, "providers": providers}


@app.get("/healthz")
async def health():
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Dev entrypoint helper (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Run directly with the instantiated app to avoid import path issues in dev.
    # Reload is disabled to keep this simple; use an external watcher if desired.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
