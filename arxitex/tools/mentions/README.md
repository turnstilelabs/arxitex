# Mentions Dataset Pipeline

Build a dataset of works that cite a target work, extract mention contexts from arXiv/ar5iv, generate synthetic researcher queries, and align those queries to statements from the target's TeX sources.

## Quick start

1) Acquire network-dependent inputs (citing works, mentions, statements):

```bash
python -m arxitex.tools.mentions.dataset.acquire_inputs \
  --targets 2211.11689 \
  --out-dir data/mentions_dataset \
  --statements-dir data/statements/mentions \
  --cache-dir data/cache
```

2) Build deterministic local dataset (contexts + gold links):

```bash
python -m arxitex.tools.mentions.dataset.build_dataset \
  --targets 2211.11689 \
  --statements-dir data/statements/mentions \
  --out-dir data/mentions_dataset
```

`build_dataset` is now local-only: it does not call OpenAlex/arXiv/ar5iv.

## Setup

Use the arxitex environment and install missing deps:

```bash
pip install -r requirements.txt
pip install -e ".[mentions]"
```

The mentions pipeline extras are provided via the `mentions` extra in `setup.py`.

## High-level flow
- **Resolve target**: arXiv URL/id → title/authors → OpenAlex Work ID.
- **Stage 1**: fetch works that cite the target (OpenAlex).
- **Stage 2**: extract mention contexts from ar5iv/PDF for arXiv-available works.
- **Stage 3**: generate synthetic researcher queries from mentions.
- **Stage 4**: extract statements from the target TeX source tree (labels + text).
- **Stage 5**: build a retrieval dataset by mapping mention refs to statements.

## Stage 1: OpenAlex citing works

```bash
python -m arxitex.tools.mentions.acquisition.citing_works_cli \
  --target-id my-target \
  --target-ids https://openalex.org/W123... \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache
```

Or resolve from an arXiv URL/id:

```bash
python -m arxitex.tools.mentions.acquisition.citing_works_cli \
  --target-arxiv https://arxiv.org/abs/2211.11689 \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache
```

If `--target-id` is omitted, it is derived from the arXiv id (e.g., `arxiv_2211.11689`).

Outputs:
- `{target}_target_ids.json`
- `{target}_works.jsonl`

## Stage 2: arXiv mention extraction

Stage 2 matches the target's bibliography entry by title and then extracts in-text
citations for the corresponding label(s).

```bash
python -m arxitex.tools.mentions.extraction.extract_mentions_cli \
  --target-title "Target Work Title" \
  --target-id my-target \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache
```

Or derive the target title from an arXiv URL/id:

```bash
python -m arxitex.tools.mentions.extraction.extract_mentions_cli \
  --target-arxiv https://arxiv.org/abs/2211.11689 \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache
```

Offline (cache-only) mode:

```bash
python -m arxitex.tools.mentions.extraction.extract_mentions_cli \
  --target-title "Target Work Title" \
  --target-id my-target \
  --out-dir data/citation_dataset \
  --cache-dir data/citation_dataset/cache \
  --offline
```

Outputs:
- `{target}_arxiv_works.jsonl`
- `{target}_mentions.jsonl`
- `{target}_failures.jsonl`

## Stage 3: synthetic queries

```bash
python -m arxitex.tools.mentions.generation \
  --target-id my-target \
  --target-name "Target Work Title" \
  --out-dir data/citation_dataset
```

Or derive the target name from an arXiv URL/id:

```bash
python -m arxitex.tools.mentions.generation \
  --target-arxiv https://arxiv.org/abs/2211.11689 \
  --out-dir data/citation_dataset
```

Outputs:
- `{target}_queries.jsonl`

Each query row also stores:
- `source_refs`, `source_named_refs`, `source_ref_text`
- `explicit_refs` (structured refs extracted from context)

### Query Quality Filters (default)
The generator applies strict filters to reduce leakage and overlong queries:
- Rejects queries that mention the target title (case-insensitive).
- Rejects queries containing theorem/lemma/etc with explicit numbers.
- Rejects bracketed citations (e.g., `[Sch12]`) and section markers.
- Rejects queries longer than 30 words.

## Stage 4: statement extraction for retrieval

Extract statements from the local TeX source tree (with PDF labels):

```bash
OPENAI_API_KEY=... \
python -m arxitex.extractor.pipeline \
  --source-dir /path/to/tex \
  --source-id my-target \
  --pdf-path /path/to/main.pdf \
  --all-enhancements \
  -o data/statements/{target}.json
```

## Stage 5: mention supervision build

Build clean mention contexts and gold links against target statements:

```bash
python -m arxitex.tools.mentions.dataset.build_dataset \
  --targets-json data/mentions/targets.json \
  --statements-dir data/statements/mentions \
  --out-dir data/mentions
```

If inputs are missing, run:

```bash
python -m arxitex.tools.mentions.dataset.acquire_inputs \
  --targets-json data/mentions/targets.json \
  --statements-dir data/statements/mentions \
  --out-dir data/mentions \
  --cache-dir data/cache
```

Outputs:
- `combined_statements.json`
- `statements.jsonl`
- `mention_contexts.jsonl`
- `mention_gold_links.json`
- `splits/*_mention_contexts.jsonl`
- `splits/*_mention_gold_links.json`

### Multi-target source prep (optional)

If you are building a dataset with many different target papers, use the source
helpers under `dataset/sources` to fetch and expand target source trees:

```bash
python -m arxitex.tools.mentions.dataset.sources.fetch_arxiv_sources \
  --categories math.AP \
  --out-dir data/sources \
  --targets-json data/mentions/targets.json
```

```bash
python -m arxitex.tools.mentions.dataset.sources.expand_targets_from_works \
  --works-dir data/mentions \
  --out-dir data/sources \
  --targets-json data/mentions/targets.json
```

## Notes

- Reference extraction is generic and works best for typical math formats (Theorem/Lemma/Definition with numbers).

## Review viewer

Open `qrels_min_viewer.html` in a browser and load the stage output JSON (queries or gold links).
