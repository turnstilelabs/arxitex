# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest -q
```

Notes:
- Tests should be network-free by default.
- Live OpenAI search tests are gated behind a marker (see `docs/USAGE_OPENAI_SEARCH.md`).

## Pre-commit
This repo includes `.pre-commit-config.yaml`.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Web

```bash
cd web
npm install
npm run lint
npm run build
```
