# Retrieval Viewer (Standalone)

This is a static HTML viewer to compare per‑query retrieval results across methods.

## Usage

1. Start a local server from the repo root:
   - `python -m http.server 8000`
2. Open `http://localhost:8000/arxitex/tools/retrieval/viewer/index.html` in a browser.
3. The viewer auto‑loads the default paths (hardcoded in `viewer.js`).
   - Graph JSON: `data/graphs/perfectoid.json`
   - Queries JSONL: `data/citation_dataset/perfectoid_queries.jsonl`
   - Results JSONL for each method: `e1_results.jsonl`, `e2_results.jsonl`, `e3_results.jsonl`, `e4_results.jsonl`, `e5_results.jsonl`
4. Pick a query from the dropdown to inspect results.

## Math Rendering

The viewer uses KaTeX auto-render from a CDN. If you are offline, math will render as plain text.

The viewer shows:
- Query text and explicit references
- Computed qrels (from graph + explicit refs)
- Top‑K results for the selected method (with type, label, and snippets)
- Relevant artifacts highlighted in green with underline
