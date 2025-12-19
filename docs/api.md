# API server

The API is implemented with **FastAPI** in [`api/main.py`](../api/main.py).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Environment variables

- `OUTPUT_DIR` (default: `./pipeline_output`)
  - The API writes graphs / definition banks / PDFs / anchors under this directory.
- `ALLOWED_ORIGINS` (optional)
  - Comma-separated list of allowed origins for CORS (e.g. `http://localhost:3000`).
  - If unset, the server allows typical localhost origins via a regex.

LLM:
- `OPENAI_API_KEY` or `TOGETHER_API_KEY`
  - Required only for LLM-powered features.

## Key endpoints

- `POST /api/v1/papers/ingest`
  - Starts a background job to process a paper.
- `GET /api/v1/jobs/{job_id}`
  - Poll job status.
- `GET /api/v1/papers/{arxiv_id}`
  - Returns graph, optional definition bank, and basic metadata.
- `GET /api/v1/papers/{arxiv_id}/stream-build`
  - Server-Sent Events stream for a live build.
- `GET /api/v1/papers/{arxiv_id}/pdf`
  - PDF proxy + disk cache.
- `GET /api/v1/llm/status`
  - Returns whether LLM features are available.
