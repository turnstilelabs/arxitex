# ArxiTex: Building Large-Scale Searchable Knowledge Graph
Our goal is to build a structured, machine-readable knowledge graph representing the logical dependencies and symbolic definitions within a paper.

## Quick summary
- Parse LaTeX sources to discover mathematical artifacts (theorems, lemmas, definitions, proofs,...).
- Build a dependency graph linking statements using explicit references and inferred semantic dependencies.
- Optionally enrich artifacts with synthesized definitions using an LLM.
- Export graphs and optional search indices / visualizations.

## Getting started
### Prerequisites
- Python 3.11+ recommended
- If you want LLM-based features, set at least one provider credential (`OPENAI_API_KEY` or `TOGETHER_API_KEY`).

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or for editable install:
pip install -e .
```

## Exporting processed papers to a Hugging Face dataset

Once papers have been processed and persisted into the SQLite DB (via the
`workflows.cli process` commands with `--persist-db`), you can export each
paper's graph + definition bank into a single JSON file suitable for a
Hugging Face dataset and the ArxiGraph webapp.

Use the `export_hf_dataset` tool:

```bash
python -m arxitex.tools.export_hf_dataset \
  --db-path path/to/arxitex.sqlite \
  --output-dir /path/to/hf-dataset/data
```

Key points:

- No re-processing: this tool only reads from the existing SQLite DB; it
  does not call LLMs or re-run extraction.
- It discovers successfully processed papers via the `processed_papers`
  table (managed by `ProcessedIndex`).
- For each paper it reconstructs the graph and definition data and
  writes:

  ```jsonc
  {
    "graph": { /* DocumentGraph.to_dict(...) */ },
    "definition_bank": { /* term -> definition */ } | null,
    "artifact_to_terms_map": { "artifact_id": ["term1", ...], ... }
  }
  ```

  to `--output-dir` using the filename convention:

  ```text
  arxiv_{arxiv_id.replace('/', '_')}.json
  ```

You can restrict export to a subset of arXiv IDs with:

```bash
python -m arxitex.tools.export_hf_dataset \
  --db-path path/to/arxitex.sqlite \
  --output-dir /path/to/hf-dataset/data \
  --only-arxiv-id 2103.14030 --only-arxiv-id 2211.11689
```

In a Hugging Face `datasets` repo these files typically live under
`data/`, and can be fetched from URLs of the form:

```text
https://huggingface.co/datasets/<org>/<repo>/resolve/<ref>/data/arxiv_2211.11689.json
```

where `<ref>` is either a branch name like `main` or an immutable
commit hash used for version pinning.

# 1. Building a graph from an ArXiv paper
## 1.1 Initial Graph Construction (`extractor/graph_building`)
We collect all artifacts (definition, proposition, claim, theorem,...) with regular expressions as well as the explicit dependencies between thoses (through the use of `\ref{...}`).

For instance, for the `2506.14029v1` paper, an example node artifact is:

```json
{
    "id": "corollary-7-d53a3c16",
    "type": "corollary",
    "content": "Let $G$ be a countable discrete group with an ICC quotient. For every candidate boundary $B$, there is a compatible probability measure $\\mu$ such that $B$ equipped with the hitting measure is not the Poisson boundary of $(G,\\mu).$",
    "content_preview": "Let G be a countable discrete group with an ICC quotient. For every candidate boundary B, there is a compatible probability measure such that B...",
    "display_name": "Corollary",
    "label": null,
    "position": {
    "line_start": 225,
    "line_end": 226
      }
}
```

with a reference to a previous theorem in its proof:

```json
{
"references": [
    {
        "target_id": "thm:always-bigger",
        "reference_type": "internal",
        "context": "let $\\mu_\\tau$ be the measure coming from Theorem \\ref{thm:always-bigger}. Since $\\mu_\\tau$ is a randomized stopping time t",
        "position": {
        "line_start": 0,
        "line_end": null
        }
    }
    ]
}
```

The `BaseGraphBuilder` is the primary engine for parsing raw LaTeX source code into a structured dependency graph. It operates in a sophisticated, multi-pass pipeline to ensure accuracy and handle the complexities of real-world academic papers.

Instead of a single monolithic class, it acts as an orchestrator, delegating specialized tasks to helper classes for a clean and maintainable design. The core process first parses all artifact and proof environments, then links detached proofs using a semantic-first strategy. Finally, it enriches the artifacts by parsing the bibliography (from embedded content or separate .bbl/.bib files) and extracting all internal (\ref) and external (\cite) references.

### Custom theorem-like environments (\newtheorem)
Many papers define custom theorem-like environments (e.g. `thmA`, `mainthm`, `prop*`) using `\newtheorem{...}{...}`. During Pass 1 we scan the LaTeX source for these declarations and automatically treat them as first-class artifacts.

The `NewTheoremScanner` maps each declared environment to a canonical artifact type based on the human-facing title ("Theorem", "Lemma", "Definition", …) and merges these dynamic aliases with a static set of common abbreviations (e.g. `thm` → `theorem`, `defn` → `definition`).

## 1.2 LLM-Powered Symbol Definition Enhancement (`symdef`)
A major challenge in understanding a paper is tracking the meaning of its specialized symbols and terms (e.g., $h(x)$, union-closed family). This sub-system is dedicated to creating a comprehensive definition bank for every symbol and concept within the paper to make artifacts self-contained. This is crucial for statement search as well. It is organised as follows.

We build for each paper its `DefinitionBank` as a central repository for all discovered definitions of a paper.

First, for all definition artifacts, an LLM (`aextract_definition`) extracts the defined term, its aliases, and the full definition text.

```json
{
  "union closed set system": {
    "term": "union closed set system",
    "aliases": [
      "$\\F$"
    ],
    "definition_text": "A set system $\\F$ is \\emph{union closed} if for all $A,B \\in \\F$ we have $A \\cup B \\in \\F$",
    "source_artifact_id": "definition-1-33775d",
    "dependencies": []
  }
}
```

Then for each artifact, we extract all the list of its non-trivial mathematical terms with another LLM call (`aextract_terms`). If it's not yet in the paper's definition bank, we synthetize its definition (`asynthesize_definition`).

```json
{
"f": {
    "term": "f",
    "aliases": [],
    "definition_text": "Let $f:[0,1]^2 \\to \\mathbb{R}_{\\ge 0}$ be defined as $$ f(x,y) := \\frac{h(xy)}{h(x)y + h(y)x} $$ for $(x,y) \\in (0,1)^2$ and extended (continuously) to $[0,1]^2$ by setting $f(x,y) = 1$ if $x \\in \\{0,1\\}$ or $y \\in \\{0,1\\}$. ",
    "source_artifact_id": "synthesized_from_context_for_claim-6-5aac5c",
    "dependencies": []
  }
}
```

Last but not least, we enhanced each artifact with the definition of all its terms

```json
{
      "content": "--- Prerequisite Definitions ---\n**\\varphi**: Let \\varphi = 1-\\psi =\\frac{\\sqrt{5}-1}{2} be the positive root of x^2+x-1=0.\n\n---\n\n\\label{f_min}\nThe function $f$ is minimized at $(\\varphi,\\varphi)$. At this point $f(\\varphi,\\varphi)=\\frac{1}{2 \\varphi}$.",
}
```

## 1.3 LLM-Powered Semantic Tags (`semantic_tagger`)
Semantic tags are short, plain-English summaries attached to artifacts.

**Single paper:**

```bash
OPENAI_API_KEY=... \
python -m arxitex.extractor.pipeline 2211.11689 --enrich-content --semantic-tags -o data/graphs/2211.11689.json
```

This produces a standard graph JSON with an added `semantic_tag` field per node. This can also be used directly:

```bash
python -m arxitex.extractor.semantic_tagger \
  --artifacts data/citation_dataset/{target}_artifacts.jsonl \
  --out data/citation_dataset/{target}_artifacts_tagged.jsonl
```

This input JSONL file must have a `text` field with the artifact content.

## 1.4 LLM-Powered Dependency Inference (`extractor/dependency_inference`)
The initial regex-based graph is often incomplete, as many dependencies are often implied rather than explicitly referenced. We can enhance the graph by inferring these missing logical links.

For each artifact, we construct a "conceptual footprint" by combining the terms directly used in the artifact with the known dependencies of those terms from the `DefinitionBank`. We then generate a list of high-potential candidate pairs by identifying artifacts that share concepts, either through direct term overlap or through a hierarchical "subword" relationship (e.g., linking "approximate union closed set system" to "union closed set system")

Finally, we send those pairs to an LLM to establish whether there is a dependency relationship between the two, and if yes what type of dependency this is.

### Dependency inference modes (pairwise / global / hybrid / auto)
The dependency inference stage supports multiple strategies. These can be selected via `--dependency-mode`.

- **`pairwise`** (original behavior):
  - Generate candidate pairs using term overlap + subword overlap.
  - Run the pairwise verifier LLM once per candidate pair.

- **`global`** (one-shot):
  - Run a single LLM call that outputs the final dependency edges for the whole paper.
  - Uses **statements + truncated proofs** (Option B) to stay within a bounded prompt size.
  - Controlled by: `--dependency-global-proof-char-budget`.

- **`hybrid`** (propose + verify):
  - Run one global *proposal* LLM call to propose candidate edges (sparse).
  - Then run the pairwise verifier only on the proposed candidates.
  - Candidate explosion is controlled via caps (wired through the CLI into a single
    per-paper pair cap):
    - `--dependency-max-pairs` (global cap on the number of LLM-verified pairs per paper)

- **`auto`** (recommended default):
  - Chooses between `global`, `hybrid`, and `pairwise` based on artifact count and an estimated token budget.

## 1.4 Paper Processing Pipeline
Examples:

```bash
  # Fast regex-only extraction, output to stdout
  python pipeline.py 2211.11689

  # Regex + enrich artifact content with definitions + output to a specific JSON file
  python pipeline.py 2211.11689 --enrich-content -o enriched.json

  # Regex + infer dependency links
  python pipeline.py 2211.11689 --infer-deps

  # Regex + infer dependency links + enrich artifact
  python pipeline.py 2211.11689 --all-enhancements --pretty

  # Local TeX source extraction (no arXiv download)
  python -m arxitex.extractor.pipeline --source-dir data/perfectoid_tex --source-id perfectoid --all-enhancements --pretty -o data/graphs/perfectoid.json
```

## 1.5 Graph Visualization
Just for fun, we propose a rudimentary visualization of the output graph

```bash
python pipeline.py 2211.11689 --infer-deps --visualize -p
```

# 2. Workflow Orchestration
The core workflow is designed around a simple two-step loop: Discover and Process. This allows  to first build a large queue of relevant papers and then process them efficiently in batches.

## 2.1 Discover: Finding and Queuing Relevant Papers

The `discover` command is the entry point for finding papers. It automatically finds all matching papers that haven't been seen before, adding them to a processing queue.

```bash
python -m arxitex.workflows.cli discover  --query cat:math.GR  --max-papers 10
```

# 2.2 Process: Analyzing Papers in Parallel Batches

The `process` command is the workhorse of the pipeline. It takes papers from the queue, downloads their LaTeX source, and runs the full analysis pipeline as explained above on them concurrently to generate their knowledge graphs.

```bash
OPENAI_API_KEY=... \
python -m arxitex.workflows.cli process \
  --mode full \
  --persist-db \
  --max-papers 20 \
  --workers 8 \
  --dependency-mode auto
```

To add semantic tags during batch processing, include `--semantic-tags` (requires `--mode defs` or `--mode full`).

You can force a specific dependency inference strategy:

```bash
# One-shot global dependency inference (statements + truncated proofs)
python -m arxitex.workflows.cli process --mode full --persist-db \
  --dependency-mode global \
  --dependency-global-proof-char-budget 1200

# Hybrid: global proposal -> pairwise verification
  python -m arxitex.workflows.cli process --mode full --persist-db \
  --dependency-mode hybrid \
  --dependency-max-pairs 250

# Original behavior
python -m arxitex.workflows.cli process --mode full --persist-db \
  --dependency-mode pairwise
```

## 2.4 (Optional) Build a "VIP" subset using citation counts (OpenAlex)

You can enrich the pipeline DB with **total citation counts** from OpenAlex and use this
signal to select a smaller, high-value subset of papers to run expensive LLM enhancements on.

### Backfill citation counts

This reads arXiv IDs from your existing pipeline DB (`pipeline_output/arxitex_indices.db`) and
stores results into a new table `paper_citations`.

```bash
python -m arxitex.workflows.cli -o pipeline_output backfill-citations \
  --workers 8 \
  --refresh-days 30 \
  --mailto "you@domain.com"
```

For a quick test run:

```bash
python -m arxitex.workflows.cli -o pipeline_output backfill-citations --max-papers 50 --workers 4 --refresh-days 0
```

### Query top-cited papers

```bash
sqlite3 pipeline_output/arxitex_indices.db "
select paper_id, citation_count
from paper_citations
where citation_count is not null
order by citation_count desc
limit 50;"
```

## 2.5 Resolve external bibliography entries to arXiv IDs
The extracted graph includes `external_reference` nodes for citations (from `\cite{...}` and bracket-style citations) with the bibliography entry text attached when available. Bibliography entries do not always contain an explicit arXiv identifier, so ArxiTex provides an optional backfill step that attempts to match each external reference to an arXiv ID.

This uses a two-stage strategy:
- fast-path: detect explicit arXiv IDs in the reference string (e.g. `arXiv:1234.5678v2`, `abs/1234.5678`)
- otherwise: heuristically extract a candidate title/authors span from the bibliography entry and query the arXiv API with fuzzy title + author matching.

Run the backfill tool:

```bash
python -m arxitex.tools.citations.arxiv_backfill \
  --db-path pipeline_output/arxitex_indices.db \
  --qps 1.0 \
  --refresh-days 30
```

Match decisions are stored in the `external_reference_arxiv_matches` table (and a cache table is maintained to avoid repeatedly querying the arXiv API for the same normalized title/authors).

# 2.3 Search format
We can convert the graph data to a better format for search witht the `--format-for-search` flag. Each artifact is saved with its extracted `terms` and the paper's `title`, `authors`, `abstract`.

```bash
python workflows/cli.py process -n 1 --enrich-content --format-for-search
```

This saves all processed papers into `search_index.jsonl`

```jsonl
{"title": "On the conjugacy problem for subdirect products of hyperbolic groups", "authors": ["Martin R. Bridson"], "arxiv_id": "2507.05087v1", "abstract": "If $G_1$ and $G_2$ are torsion-free hyperbolic groups and $P<G_1\\times G_2$ is a finitely generated subdirect product, then the conjugacy problem in $P$ is solvable if and only if there is a uniform algorithm to decide membership of the cyclic subgroups in the finitely presented group $G_1/(P\\cap G_1)$. The proof of this result relies on a new technique for perturbing elements in a hyperbolic group to ensure that they are not proper powers.", "artifact_id": "lemma-6-2c644b", "artifact_type": "lemma", "content": "**C_G(w)**: we write $C_G(w)$ to denote the centraliser in $G$ of the image of $w\\in F(X)$. \n\n**G**: The rel-cyclics Dehn function of a finitely presented group $\\G=\\<X\\mid R\\>$ is $$ \\d^{c}(n) : = \\max_{w,u} \\{ {\\rm{Area}}(w\\,u^p) +|pn| \\colon |w|+|u|\\le n,\\ w=_\\G u^{-p},\\ |p|\\le  o(u)/2\\}, $$ where $o(u)\\in\\N\\cup\\{\\infty\\}$ is the order of $u$ in $\\G$.\n\n---\n\n\\label{l:find-root}\nIf $G$ is torsion-free and hyperbolic, then there is an algorithm that, given a word $w$ in the generators, will decide if \n$w$ is non-trivial in $G$ and, if it is,  will produce a word $w_0$ such that $C_G(w) = \\<w_0\\>$.", "terms": ["<w_0>", "C_G(w)", "G", "generators", "hyperbolic", "non-trivial", "torsion-free", "word"]}
```
