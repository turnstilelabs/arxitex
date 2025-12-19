# ArxiTex Web

This folder contains the **Next.js** frontend for ArxiTex.

It connects to the ArxiTex API (FastAPI) and provides:
- paper ingestion + live streaming build (SSE)
- PDF reader + artifact selection
- graph visualization + artifact details

## Prerequisites
- Node.js 18+ recommended
- The API server running (see `../docs/api.md`)

## Configure

Create a local env file:

```bash
cp .env.example .env.local
```

By default it points to `http://127.0.0.1:8000`.

## Run

```bash
npm install
npm run dev
```

Open: http://localhost:3000

## Lint / build

```bash
npm run lint
npm run build
```
