# Citation Dataset Pipeline

Build a dataset of works that cite a target work, extract mention contexts from arXiv/ar5iv, and generate synthetic researcher queries. Then map references to TeX artifacts.

## Setup

Use the arxitex environment and install missing deps:

```bash
pip install -r requirements.txt
```

Additional deps for this pipeline:
- `beautifulsoup4`, `lxml`, `pdfminer.six`

## Stage 1: OpenAlex citing works

```bash
python -m arxitex.tools.citations.dataset.arxiv_identification \
  --target-id my-target \
  --target-ids https://openalex.org/W123... \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache
```

Outputs:
- `{target}_target_ids.json`
- `{target}_works.jsonl`

## Stage 2: arXiv mention extraction

Stage 2 matches the target's bibliography entry by title and then extracts in-text
citations for the corresponding label(s).

```bash
python -m arxitex.tools.citations.dataset.get_citations \
  --target-title "Target Work Title" \
  --target-id my-target \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache
```

Outputs:
- `{target}_arxiv_works.jsonl`
- `{target}_mentions.jsonl`
- `{target}_failures.jsonl`

## Stage 3: synthetic queries

```bash
python -m arxitex.tools.citations.dataset.query_generation \
  --target-id my-target \
  --target-name "Target Work Title" \
  --out-dir data/citation_dataset
```

Outputs:
- `{target}_queries.jsonl`

Each query row also stores:
- `source_refs`, `source_named_refs`, `source_ref_text`
- `explicit_refs` (structured refs extracted from context)

## Stage 4: graph extraction for retrieval

Build a graph from the local TeX source tree (with PDF labels) before running Stage 3:

```bash
OPENAI_API_KEY=... \
python -m arxitex.extractor.pipeline \
  --source-dir /path/to/tex \
  --source-id my-target \
  --pdf-path /path/to/main.pdf \
  --all-enhancements \
  -o data/graphs/{target}.json
```

## Notes

- Reference extraction is generic and works best for typical math formats (Theorem/Lemma/Definition with numbers).

## Review viewer

Open `qrels_min_viewer.html` in a browser and load the `{target}_queries.jsonl`.
