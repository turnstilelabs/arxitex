# ArxiTex: Building Large-Scale Searchable Knowledge Graph

Our goal is to build a structured, machine-readable knowledge graph representing the logical dependencies and symbolic definitions within a paper.

Quick summary
- Parse LaTeX sources to discover mathematical artifacts (theorems, lemmas, definitions, proofs).
- Build a dependency graph linking statements using explicit references and inferred semantic dependencies.
- Optionally enrich artifacts with synthesized definitions using an LLM.
- Export graphs and optional search indices / visualizations.

Getting started (short)
Prerequisites
- Python 3.11+ recommended
- A virtual environment (venv / .venv) is recommended
- If you want LLM-based features, set at least one provider credential (see Configuration)

Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or for editable install:
pip install -e .
```

Run a single-paper extraction (example)
This repository provides two entry points:

- Lightweight pipeline script (module): arxitex.extractor.pipeline
- Workflow CLI: arxitex.workflows.cli

Using the pipeline module to extract one paper:
```bash
# Run regex-only extraction (no LLM)
python -m arxitex.extractor.pipeline 2211.11689

# Run with enrichment + dependency inference (requires OPENAI_API_KEY or other LLM keys)
python -m arxitex.extractor.pipeline 2211.11689 --infer-deps --enrich-content --pretty
```

Using the workflow CLI:
```bash
# Process a single paper (temporary download + processing)
python -m arxitex.workflows.cli single 2211.11689

# Discover papers from arXiv matching a query and add to queue
python -m arxitex.workflows.cli discover --query "cat:math.GR" --max-papers 50

# Process queued papers (concurrent workers)
python -m arxitex.workflows.cli process --max-papers 20 --workers 4 --enrich-content --infer-dependencies
```

Where outputs go
- By default outputs are written to `pipeline_output/` in the project root (can be overridden with CLI `-o/--output-dir`).
- Example saved graphs, banks and visualizations may appear under `data/` for examples. For normal runs, generated data should be stored outside version control (see .gitignore).

Configuration
Environment variables
- OPENAI_API_KEY — required for OpenAI-based LLM features (used when running with --infer-deps or --enrich-content).
- TOGETHER_API_KEY — optional, used by Together client if you choose Together as a provider (check provider docs).
- RICH_QUIET and TQDM_DISABLE are set by the CLI to reduce interactive output, but you can control progress/spinner behavior via standard env/config.

CLI flags and paths
- `-o / --output-dir`: sets the base directory where pipeline output and indices are stored (default: `pipeline_output/`).
- Database / index path: workflows create and manage an sqlite DB under `pipeline_output/` by default. See `arxitex/indices` for index classes.
- LLM model/provider selection is controlled in code (arxitex/llms/llms.py). For production, consider setting a small config file or environment-driven provider selection.

Example .env (create at project root)
```
# Example .env file
OPENAI_API_KEY=sk_...
# If using Together:
# TOGETHER_API_KEY=...
```
Tip: never commit secrets. Use a secrets manager or repository variables for CI.

Running tests
The repo includes a test suite. To run tests locally:
```bash
# with virtualenv active
pip install -r requirements.txt
pytest -q
```

Architecture (short overview)
- Downloaders: arxitex/downloaders — fetch and extract LaTeX projects
- Extraction & graph building: arxitex/extractor and arxitex/extractor/graph_building
- Symbol definition synthesis: arxitex/symdef
- LLM integration + prompt caching: arxitex/llms
- Indices & persistence: arxitex/indices
- Orchestration CLI/workflows: arxitex/workflows

Notes on publishing
- The repo intentionally avoids committing large generated artifacts. The .gitignore now excludes `pipeline_output/` and generated `data/*` so users can run the pipeline locally to reproduce outputs.
- A LICENSE (MIT) has been added. If you include third-party content (paper TeX sources), verify redistribution rights.

Examples and visualizations
- The project contains small example graphs in `data/` for demonstration. For reproducible runs, use the CLI with a temporary output dir or configure an output path.

Where to go next
- If you plan to use LLM features, create an .env file with your provider keys and follow the Configuration section above.
- To help contributors, consider adding small example inputs under `data/examples/` (kept intentionally small).

If something in this quickstart doesn't work on your machine, tell me which step failed and I will help fix it or adapt the instructions.
