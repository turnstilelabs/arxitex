# Web app

The web UI lives in `web/` and is a Next.js (App Router) application.

## Run locally

In one terminal:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd web
npm install
npm run dev
```

Then open: http://localhost:3000

## Configuration

The UI talks to the API using:
- `NEXT_PUBLIC_API_BASE` (default: `http://127.0.0.1:8000`)

Create `web/.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

## Notes
- The paper view uses **SSE** to stream builds (`/stream-build`).
- The PDF viewer uses the API PDF proxy endpoint to avoid CORS issues.
