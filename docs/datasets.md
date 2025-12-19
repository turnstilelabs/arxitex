# Datasets & generated artifacts

ArxiTex can produce a variety of artifacts while processing papers:
- extracted graphs (`*.json`)
- definition banks (`*_bank.json`)
- anchor indices (`anchors/*.json`)
- PDFs cached on disk
- search indices (`*.jsonl`)

## Repository policy

In this open-source repository:
- `data/` and `pipeline_output/` are treated as **generated build artifacts**.
- They should **not** be committed.

Why?
- Graphs and PDFs are derived from third-party papers and may have redistribution constraints.
- Outputs can be large and churn frequently.

## Samples

We keep a **tiny sample dataset** under `samples/` to:
- provide a stable example for docs/tests
- avoid bundling large or copyrighted artifacts

The sample graph is synthetic (hand-written) and is not derived from any paper.
