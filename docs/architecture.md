# Architecture

ArxiTex turns an arXiv paper’s LaTeX source into a structured, machine-readable representation of:
- **Artifacts** (definitions, theorems, lemmas, proofs, …)
- **References** (explicit `\\ref{}` / bibliography `\\cite{}`)
- **Dependencies** (explicit and optionally inferred)
- **Definition bank** (optionally enriched with LLM-extracted / synthesized definitions)

At a high level, the system is:

```
(arXiv id) -> download LaTeX -> parse artifacts -> link proofs/refs
                             -> (optional) build definition bank
                             -> (optional) infer missing dependencies
                             -> serialize graph (+ optional bank/anchors)
```

## Main Python modules

### 1) Downloading (`arxitex/downloaders`)
- `AsyncSourceDownloader` retrieves and extracts the arXiv source.
- Utilities combine TeX files into a single string when needed.

### 2) Graph extraction (`arxitex/extractor`)
The extractor builds a `DocumentGraph`:
- **Pass 1**: regex-based artifact detection.
- Proof linking and reference resolution.
- **Optional pass**: LLM-powered dependency inference.

Key components:
- `extractor/graph_building/BaseGraphBuilder` – initial artifact and reference extraction.
- `extractor/graph_building/GraphEnhancer` – orchestrates enhancements.
- `extractor/dependency_inference/*` – candidate generation + LLM inference.

### 3) Symbol definition enrichment (`arxitex/symdef`)
A `DefinitionBank` maps symbols/terms to:
- canonical term + aliases
- definition text
- provenance (artifact id)
- optional dependency list

This can be used to enrich artifacts so they’re more self-contained.

### 4) Workflows (`arxitex/workflows`)
Workflows implement a discover/process loop:
- `discover` queues new arXiv IDs
- `process` consumes the queue and generates graphs in batches

### 5) API (`api/main.py`)
A FastAPI service exposes:
- ingestion endpoint (async job)
- SSE build endpoint for live streaming
- endpoints to fetch graph / definition bank / anchors / PDF proxy

### 6) Web app (`web/`)
The Next.js app provides:
- “enter arXiv id” home page
- paper view: PDF reader + graph viewer + artifact details
- live streaming UI using SSE

## Data model
The frontend expects these JSON shapes (see `web/src/lib/types.ts`):
- `DocumentGraph` with `nodes` and `edges`
- optional `DefinitionBank`
- optional `ArtifactAnchorIndex` for PDF deep-linking

## Extension points
- Add new artifact types and extraction rules in the graph builder
- Add new LLM providers or prompting strategies in `arxitex/llms`
- Add new indices/export formats in `arxitex/indices` and `web/src/lib/export.ts`
