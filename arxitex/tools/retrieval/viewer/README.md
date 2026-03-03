# Retrieval Viewer (Standalone)

This is a static HTML viewer to compare per‑query retrieval results across methods.

## Usage

1. Start a local server from the repo root:
   - `python -m http.server 8000`
2. Open `http://localhost:8000/arxitex/tools/retrieval/viewer/index.html` in a browser.
3. Add URL parameters to point to your files:
   - `?graph=/data/graphs/your_target.json`
   - `&queries=/data/citation_dataset/your_target_queries.jsonl`
   - Optional: `&e1=...&e2=...&e3=...&e4=...&e5=...`
4. Pick a query from the dropdown to inspect results.

## Math Rendering

The viewer uses KaTeX auto-render from a CDN. If you are offline, math will render as plain text.

The viewer shows:
- Query text and explicit references
- Computed qrels (from graph + explicit refs)
- Top‑K results for the selected method (with type, label, and snippets)
- Relevant artifacts highlighted in green with underline
