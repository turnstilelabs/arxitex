# Retrieval Tools

This directory contains retrieval baselines and benchmarking utilities for the
mentions dataset produced by `arxitex.tools.mentions.dataset.build_dataset`.

## Quick start with mentions dataset

1) Build the mentions dataset and emit queries/qrels:

```bash
python -m arxitex.tools.mentions.dataset.build_dataset \
  --targets 2211.11689 \
  --out-dir data/mentions_dataset \
  --emit-queries
```

2) Run the retrieval benchmark:

```bash
python -m arxitex.tools.retrieval.retrieval_benchmark \
  --graph data/mentions_dataset/combined_statements.json \
  --queries data/mentions_dataset/queries.jsonl \
  --qrels data/mentions_dataset/qrels.json \
  --out-dir data/retrieval \
  --experiment e1
```

Notes:
- `--graph` expects a JSON file with `nodes` (edges optional). The combined
  statements output from Stage 5 is sufficient.
- `--qrels` accepts either JSONL (`{"query_id":..., "relevant_ids":[...]}` per line)
  or a JSON dict (`{"qid": ["id1", "id2"]}`).

## Experiments

The `--experiment` flag supports:
- `e1`: BM25
- `e2`: Dense embeddings (OpenAI)
- `e3`: PyLate / PLAID (late interaction dense retrieval)
- `e4`: Hybrid RRF (reciprocal-rank fusion of multiple rankers)
- `e5`: Agentic ColGREP (LLM-guided search refinement)

Run `--experiment all` to evaluate all baselines.

## Training (biencoder)

Biencoder training scripts live under `tools/retrieval/training/biencoder`:

```bash
python -m arxitex.tools.retrieval.training.biencoder.train_biencoder
python -m arxitex.tools.retrieval.training.biencoder.evaluate_biencoder
```
