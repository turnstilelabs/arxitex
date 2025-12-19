# CLI

ArxiTex ships a Python entrypoint:

```bash
arxitex --help
```

Internally this maps to `arxitex.workflows.cli:cli_main`.

## Workflow commands

### Discover
Find new papers matching an arXiv query and enqueue them:

```bash
arxitex discover --query 'cat:math.GR' --max-papers 10
```

### Process
Download and process papers from the queue in parallel:

```bash
arxitex process --max-papers 20 --workers 8 --enrich-content --infer-dependencies
```

### Single
Process one paper and write output under the configured output directory:

```bash
arxitex single 2211.11689 --enrich-content --infer-dependencies
```

## Legacy / direct pipeline runner
There is also a script-style entrypoint:

```bash
python -m arxitex.extractor.pipeline 2211.11689 --pretty
```

This writes outputs under `data/` by default. In the open-source repo, `data/` is treated as *generated* (see [`datasets.md`](./datasets.md)).
