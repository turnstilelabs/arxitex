# SGA 4.5 experiment (ArxiTex)

This folder contains a reproducible experiment for running **artifact extraction** on the
SGA 4.5 LaTeX sources (Deligne, *SGA 4 1/2*) and producing a **pre-run estimate** of the
LLM cost/runtime for:

1. symbol/term enhancement (definition bank + term extraction/synthesis)
2. dependency-graph inference

The estimate is produced **without running dependency inference**.

## Quickstart

From the repo root:

```bash
# 1) (already done by the assistant in this workspace)
# git clone https://github.com/NomiL/sga4.5.git experiment/sga45/src/sga4.5

# 2) Run the estimator
python experiment/sga45/scripts/estimate_cost.py \
  --repo-dir experiment/sga45/src/sga4.5 \
  --output-dir experiment/sga45/output
```

## Tuning dependency strategy + runtime assumptions

The estimator reports **side-by-side** dependency inference estimates for:

- `global` (1 call)
- `hybrid` (1 proposer call + up to N pairwise verifications)
- `pairwise` (up to N pairwise verifications)
- `auto` (the mode ArxiTex would pick given `--dependency-auto-max-*`)

You can control the caps/heuristics:

```bash
python experiment/sga45/scripts/estimate_cost.py \
  --repo-dir experiment/sga45/src/sga4.5 \
  --output-dir experiment/sga45/output \
  --max-total-pairs 100 \
  --dependency-auto-max-tokens 12000
```

And you can change the *runtime projection* assumptions (purely heuristic):

```bash
python experiment/sga45/scripts/estimate_cost.py \
  --repo-dir experiment/sga45/src/sga4.5 \
  --output-dir experiment/sga45/output \
  --assumed-latency-seconds 2.0 \
  --assumed-tokens-per-second 2000 \
  --assumed-total-tokens-multiplier 1.15
```

Outputs:

- `experiment/sga45/output/graphs_raw/` : regex-only graphs (artifacts + explicit references)
- `experiment/sga45/output/report.json` : machine-readable estimate summary
- `experiment/sga45/output/report.md` : human-readable estimate summary

## Notes

- Token/call estimates are based on prompt-size heuristics and ArxiTex's existing
  auto-mode token estimator for global dependency inference.
- **No dependency inference LLM calls are made** by the estimator.
