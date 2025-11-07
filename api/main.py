import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Literal, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Reuse your pipeline
from arxitex.extractor.pipeline import agenerate_artifact_graph
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
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

    return {
        "arxiv_id": arxiv_id,
        "graph": graph_data,
        "definition_bank": bank_data,
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

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
