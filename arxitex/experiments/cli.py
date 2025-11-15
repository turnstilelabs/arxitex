"""
CLI to run LLM retrieval experiments (exact paper retrieval) without Jupyter.

Implements the same pipeline as the notebook:
- Random ID-based discovery and stratified sampling by time buckets (older, recent, latest_pre_cutoff, post_cutoff)
- Artifact extraction with enrich_content=False and infer_dependencies=False
- Query generation in 3 categories: verbatim, reformulated, vague
- Retrieval: closed-book and optional web-assisted (arXiv candidates)
- Evaluation: Strong Match = (ID match) AND (Title match, normalized)

Usage examples:
  python -m arxitex.experiments.cli sample --sample-per-bucket 5
  python -m arxitex.experiments.cli extract --run-dir experiments/math-llm-retrieval-20250101-120000
  python -m arxitex.experiments.cli gen-queries --run-dir <run_dir> --model gpt-5-2025-08-07
  python -m arxitex.experiments.cli retrieve --run-dir <run_dir> --model gpt-5-2025-08-07
  python -m arxitex.experiments.cli evaluate --run-dir <run_dir>
  python -m arxitex.experiments.cli all --sample-per-bucket 3 --model gpt-5-2025-08-07

Notes:
- OpenAI only. Ensure OPENAI_API_KEY is set.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import typer
from loguru import logger
from pydantic import BaseModel, Field

from arxitex.arxiv_api import ArxivAPI
from arxitex.experiments.generate_then_verify import (
    run_in_memory_experiment,
    sample_by_generating_and_verifying,
)
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.llms.llms import aexecute_prompt
from arxitex.llms.prompt import Prompt

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Default constants
EXPERIMENTS_ROOT = Path("experiments")
DEFAULT_RUN_NAME = "math-llm-retrieval"
DEFAULT_CUTOFF = datetime(2024, 11, 1, tzinfo=timezone.utc)
ARTIFACT_TYPES = {
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "claim",
    "result",
    "conclusion",
}

# Precomputed constants and Typer option objects to avoid function calls in defaults (B008)
DEFAULT_CUTOFF_ISO = DEFAULT_CUTOFF.isoformat()

# Common options
OPT_SAMPLE_SAMPLE_PER_BUCKET = typer.Option(
    10, "--sample-per-bucket", "-n", help="Number of papers per bucket."
)
OPT_SEED = typer.Option(42, "--seed", help="Random seed.")
OPT_RUN_NAME = typer.Option(
    DEFAULT_RUN_NAME, "--run-name", help="Name prefix for experiment run folder."
)
OPT_CUTOFF_ISO = typer.Option(
    DEFAULT_CUTOFF_ISO,
    "--cutoff-iso",
    help="Cutoff ISO date for post_cutoff bucket (e.g., 2024-01-01T00:00:00+00:00).",
)
OPT_MAX_TO_COLLECT = typer.Option(
    1000,
    "--max-collect",
    help="Upper bound on random ID sampling attempts (approx. batches = max-collect/batch-size).",
)
OPT_BATCH_SIZE = typer.Option(
    100, "--batch-size", help="Random ID batch size per API call (<=100)."
)

OPT_RUN_DIR = typer.Option(
    ..., "--run-dir", exists=True, file_okay=False, dir_okay=True, readable=True
)
OPT_MAX_ARTIFACTS_PER_PAPER = typer.Option(
    10, "--max-artifacts-per-paper", help="Cap artifacts per paper."
)

OPT_QUERIES_PER_CATEGORY = typer.Option(
    3, "--k", help="Number of queries per category."
)
OPT_MODEL_OPENAI = typer.Option(
    "gpt-5-2025-08-07", "--model", "-m", help="OpenAI model name (OpenAI only)."
)

OPT_WEB_ASSIST_BOOL = typer.Option(
    False,
    "--web-assist/--no-web-assist",
    help="Enable arXiv-candidate-assisted mode.",
)
OPT_ARXIV_WEB_TOP_K = typer.Option(
    5, "--web-top-k", help="Number of arXiv candidates for web-assisted mode."
)

# Variants for 'all' command
OPT_SAMPLE_PER_BUCKET_E2E = typer.Option(3, "--sample-per-bucket", "-n")
OPT_MAX_TO_COLLECT_E2E = typer.Option(400, "--max-collect")
OPT_MAX_ARTIFACTS_PER_PAPER_E2E = typer.Option(5, "--max-artifacts-per-paper")
OPT_QUERIES_PER_CATEGORY_E2E = typer.Option(2, "--k")

# sample_ids specific
OPT_ID_FILE = typer.Option(
    None, "--id-file", help="Path to a text file with one arXiv ID per line."
)
OPT_IDS_CSV = typer.Option(None, "--ids", help="Comma-separated arXiv IDs.")

# sample_and_run specific
OPT_N_PER_BUCKET = typer.Option(10, "--n-per-bucket", "-n")
OPT_RUN_NAME_GNV = typer.Option("math-llm-retrieval-gnv", "--run-name")
OPT_MIN_OK_PER_BUCKET = typer.Option(5, "--min-ok-per-bucket")
OPT_MODEL_GPT4_TURBO = typer.Option("gpt-4-turbo", "--model", "-m")
OPT_CONCURRENCY = typer.Option(8, "--concurrency")


# ------------- Helpers -------------


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(run_name: str) -> Path:
    run_dir = EXPERIMENTS_ROOT / f"{run_name}-{now_timestamp()}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    # Lowercase, strip punctuation, collapse whitespace
    s = s.lower().strip()
    table = str.maketrans({c: " " for c in string.punctuation})
    s = s.translate(table)
    s = " ".join(s.split())
    return s


def strong_match(
    pred_id: Optional[str], pred_title: Optional[str], true_id: str, true_title: str
) -> Dict[str, int]:
    id_match = 1 if (pred_id is not None and pred_id.strip() == true_id.strip()) else 0
    title_match = 0
    if pred_title is not None:
        title_match = (
            1 if normalize_text(pred_title) == normalize_text(true_title) else 0
        )
    return {
        "id_match": id_match,
        "title_match": title_match,
        "strong_match": 1 if (id_match and title_match) else 0,
    }


def parse_iso8601(ts: str) -> datetime:
    # Atom uses ISO 8601, usually with 'Z'. Convert to +00:00 for Python
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def bucket_for_date(dt: datetime, cutoff: datetime) -> str:
    older_end = datetime(2014, 12, 31, tzinfo=timezone.utc)
    recent_start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    recent_end = datetime(2020, 12, 31, tzinfo=timezone.utc)
    latest_start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    latest_end = datetime(2023, 12, 31, tzinfo=timezone.utc)

    if dt <= older_end:
        return "older"
    if recent_start <= dt <= recent_end:
        return "recent"
    if latest_start <= dt <= latest_end:
        return "latest_pre_cutoff"
    if dt >= cutoff:
        return "post_cutoff"
    # Between end of 2023 and cutoff if cutoff moves in the future
    return "latest_pre_cutoff"


def entries_to_papers(
    entries: List[Any], api: Optional[ArxivAPI] = None
) -> List[Dict[str, Any]]:
    # Reuse ArxivAPI.entry_to_paper
    local_api = api or ArxivAPI()
    papers: List[Dict[str, Any]] = []
    for e in entries:
        p = local_api.entry_to_paper(e)
        if p is None:
            continue
        pub_elem = e.find("atom:published", local_api.ns)
        published = pub_elem.text if (pub_elem is not None and pub_elem.text) else None
        p["published"] = published
        papers.append(p)
    if api is None:
        local_api.close()
    return papers


def stratified_sample(
    papers: List[Dict[str, Any]], per_bucket: int, seed: int, cutoff: datetime
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    random.seed(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "older": [],
        "recent": [],
        "latest_pre_cutoff": [],
        "post_cutoff": [],
    }
    for p in papers:
        if not p.get("published"):
            continue
        dt = parse_iso8601(p["published"])  # type: ignore
        b = bucket_for_date(dt, cutoff)
        buckets[b].append(p)

    sampled: List[Dict[str, Any]] = []
    for bname, blist in buckets.items():
        if len(blist) == 0:
            continue
        k = min(per_bucket, len(blist))
        sampled_bucket = random.sample(blist, k)
        for s in sampled_bucket:
            s["bucket"] = bname
        sampled.extend(sampled_bucket)

    return sampled, buckets


# ------------- Query generation schemas and prompts -------------


class QueryGenOutput(BaseModel):
    queries: List[str] = Field(default_factory=list)


def prompt_for_queries(
    category: str,
    title: str,
    abstract: str,
    artifact_type: str,
    artifact_text: str,
    k: int,
) -> Prompt:
    rules_common = (
        "Write like a mathematician searching arXiv/Google Scholar."
        " Use 4–16 tokens typical."
        " Avoid instruction phrasing (no 'please' or 'you are')."
        " Do not include LaTeX labels or internal artifact IDs."
    )
    if category == "verbatim":
        style = (
            "Generate queries that would retrieve the artifact nearly word-for-word."
            " Keep close to the original phrasing, including key terms/symbols if helpful."
        )
    elif category == "reformulated":
        style = (
            "Generate paraphrased queries that preserve technical specificity."
            " Include salient terms, named objects or concepts from title/abstract."
        )
    else:  # vague
        style = (
            "Generate short, high-level queries capturing only the gist someone might recall."
            " Prefer brevity and general terms."
        )

    sys = "You generate realistic search queries used by mathematicians. Output strictly JSON matching the provided schema."
    usr = (
        f"Paper title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Artifact type: {artifact_type}\n"
        f"Artifact text: {artifact_text}\n\n"
        f"Task: {style} {rules_common} Generate exactly {k} queries."
    )
    return Prompt(system=sys, user=usr)


# ------------- Retrieval schemas and prompts -------------


class Ref(BaseModel):
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class RetrievalOutput(BaseModel):
    reference: Ref
    confidence: Optional[float] = 0.0
    reasoning: Optional[str] = None


class CandidatePick(BaseModel):
    reference: Ref
    confidence: Optional[float] = 0.0
    reasoning: Optional[str] = None


def prompt_closed_book(query: str) -> Prompt:
    sys = (
        "You retrieve references of math papers. You MUST answer strictly in JSON matching the schema."
        " If uncertain, still pick your single best guess with low confidence."
    )
    usr = (
        f"Query: {query}\n\n"
        "Task: Return a single reference you believe matches best."
        " If you think you know the exact arXiv id and title, fill both; otherwise leave fields null but still give one best guess."
    )
    return Prompt(system=sys, user=usr)


def arxiv_search_candidates(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    # Build a simple arXiv 'all:' query joining terms
    terms = [t for t in query.split() if t]
    search = " ".join([f"all:{t}" for t in terms]) or "all:math"
    api = ArxivAPI()
    resp = api.fetch_papers(search, start=0, batch_size=max(10, top_k * 3))
    n, _, entries = api.parse_response(resp)
    cands: List[Dict[str, Any]] = []
    if n and entries:
        papers = entries_to_papers(entries, api=api)
        for p in papers[:top_k]:
            cands.append(
                {
                    "arxiv_id": p["arxiv_id"],
                    "title": p["title"],
                    "url": f"https://arxiv.org/abs/{p['arxiv_id']}",
                    "abstract": p.get("abstract", "")[:400],
                }
            )
    api.close()
    return cands


def prompt_web_assist(query: str, candidates: List[Dict[str, Any]]) -> Prompt:
    sys = (
        "Select exactly one candidate from the provided list that best matches the user's query."
        " Output strictly the JSON schema. Do not create new references."
    )
    usr = f"Query: {query}\n\nCandidates (each with arxiv_id, title, url, abstract_snippet):\n{json.dumps(candidates, ensure_ascii=False)}\n\nPick exactly one from this list."
    return Prompt(system=sys, user=usr)


# ------------- Commands -------------


@app.command(
    help="Random ID-based sampling by time buckets. Save selected papers and config."
)
def sample(
    sample_per_bucket: int = OPT_SAMPLE_SAMPLE_PER_BUCKET,
    seed: int = OPT_SEED,
    run_name: str = OPT_RUN_NAME,
    cutoff_iso: str = OPT_CUTOFF_ISO,
    max_to_collect: int = OPT_MAX_TO_COLLECT,
    batch_size: int = OPT_BATCH_SIZE,
):
    cutoff = datetime.fromisoformat(cutoff_iso)
    logger.info(
        f"Sampling randomly by arXiv ID per time bucket. per_bucket={sample_per_bucket} cutoff={cutoff.isoformat()}"
    )
    max_batches = max(1, max_to_collect // max(1, batch_size))
    sampled = sample_random_math_by_id_buckets(
        per_bucket=sample_per_bucket,
        seed=seed,
        cutoff=cutoff,
        ids_batch_size=batch_size,
        max_batches=max_batches,
    )

    # Enforce desired number of papers per bucket with adaptive retries
    req = sample_per_bucket
    req_buckets = ["older", "recent", "latest_pre_cutoff", "post_cutoff"]

    def _counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Dict[str, int] = {b: 0 for b in req_buckets}
        for p in rows:
            b = p.get("bucket", "latest_pre_cutoff")
            c[b] = c.get(b, 0) + 1
        return c

    counts = _counts(sampled)
    missing = {b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req}

    tries = 0
    while missing and tries < 5:
        tries += 1
        more_batches = max_batches * (2**tries)
        sampled = sample_random_math_by_id_buckets(
            per_bucket=sample_per_bucket,
            seed=seed + tries,
            cutoff=cutoff,
            ids_batch_size=min(100, max(1, batch_size)),
            max_batches=more_batches,
            target_buckets=list(missing.keys()),
            existing=sampled,
        )
        counts = _counts(sampled)
        missing = {
            b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req
        }

    if missing:
        # Try fallback scan of math feed to fill remaining buckets
        sampled = fallback_fill_missing_with_query(
            sampled=sampled,
            missing=missing,
            cutoff=cutoff,
            seen_ids={p["arxiv_id"] for p in sampled},
            batch_size=batch_size,
            max_scan=max(2000, max_batches * batch_size * 10),
        )
        counts = _counts(sampled)
        missing = {
            b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req
        }
        if missing:
            raise typer.BadParameter(
                f"Could not fill all buckets to {req} each after {tries} adaptive retries and fallback scan. Missing: {missing}. Try increasing --max-collect or --batch-size, or adjust --cutoff-iso."
            )
        # Try fallback scan of math feed to fill remaining buckets
        sampled = fallback_fill_missing_with_query(
            sampled=sampled,
            missing=missing,
            cutoff=cutoff,
            seen_ids={p["arxiv_id"] for p in sampled},
            batch_size=batch_size,
            max_scan=max(2000, max_batches * batch_size * 10),
        )
        counts = _counts(sampled)
        missing = {
            b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req
        }
        if missing:
            raise typer.BadParameter(
                f"Could not fill all buckets to {req} each after {tries} adaptive retries and fallback scan. Missing: {missing}. Try increasing --max-collect or --batch-size, or adjust --cutoff-iso."
            )
        # Try fallback scan of math feed to fill remaining buckets
        sampled = fallback_fill_missing_with_query(
            sampled=sampled,
            missing=missing,
            cutoff=cutoff,
            seen_ids={p["arxiv_id"] for p in sampled},
            batch_size=batch_size,
            max_scan=max(2000, max_batches * batch_size * 10),
        )
        counts = _counts(sampled)
        missing = {
            b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req
        }
        if missing:
            raise typer.BadParameter(
                f"Could not fill all buckets to {req} each after {tries} adaptive retries and fallback scan. Missing: {missing}. Try increasing --max-collect or --batch-size, or adjust --cutoff-iso."
            )
    # Enforce desired number of papers per bucket with adaptive retries
    req = sample_per_bucket
    req_buckets = ["older", "recent", "latest_pre_cutoff", "post_cutoff"]

    def _counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Dict[str, int] = {b: 0 for b in req_buckets}
        for p in rows:
            b = p.get("bucket", "latest_pre_cutoff")
            c[b] = c.get(b, 0) + 1
        return c

    counts = _counts(sampled)
    missing = {b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req}

    tries = 0
    while missing and tries < 5:
        tries += 1
        more_batches = max_batches * (2**tries)
        sampled = sample_random_math_by_id_buckets(
            per_bucket=sample_per_bucket,
            seed=seed + tries,
            cutoff=cutoff,
            ids_batch_size=min(100, max(1, batch_size)),
            max_batches=more_batches,
            target_buckets=list(missing.keys()),
            existing=sampled,
        )
        counts = _counts(sampled)
        missing = {
            b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req
        }

    if missing:
        raise typer.BadParameter(
            f"Could not fill all buckets to {req} each after {tries} adaptive retries. Missing: {missing}. Try increasing --max-collect or --batch-size, or adjust --cutoff-iso."
        )

    run_dir = make_run_dir(run_name)
    config = {
        "run_name": run_name,
        "seed": seed,
        "cutoff_date": cutoff.isoformat(),
        "sample_per_bucket": sample_per_bucket,
        "sampling_mode": "random_by_id",
        "max_artifacts_per_paper": 10,
        "queries_per_category": 3,
        "use_web_assist": False,
        "arxiv_web_top_k": 5,
        "openai_model": "gpt-5-2025-08-07",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_jsonl(run_dir / "samples" / "selected_papers.jsonl", sampled)
    logger.success(f"Run dir: {run_dir}")
    logger.success(
        f"Saved sampled papers: {run_dir / 'samples' / 'selected_papers.jsonl'}"
    )


async def _process_one_paper(
    arxiv_id: str, out_graphs_dir: Path
) -> Optional[Dict[str, Any]]:
    try:
        results = await agenerate_artifact_graph(
            arxiv_id=arxiv_id,
            infer_dependencies=False,
            enrich_content=False,
            source_dir=out_graphs_dir.parent / "temp",
        )
        graph = results.get("graph")
        if not graph or not getattr(graph, "nodes", None):
            return None
        graph_data = graph.to_dict(arxiv_id=arxiv_id)
        out_path = out_graphs_dir / f"{arxiv_id.replace('/', '_')}.json"
        out_path.write_text(
            json.dumps(graph_data, ensure_ascii=False), encoding="utf-8"
        )
        return graph_data
    except Exception as e:
        logger.error(f"Error processing {arxiv_id}: {e}")
        return None


def _flatten_artifacts(
    graph_data: Dict[str, Any], paper_meta: Dict[str, Any], max_artifacts_per_paper: int
) -> List[Dict[str, Any]]:
    nodes = graph_data.get("nodes", [])
    rows: List[Dict[str, Any]] = []
    count = 0
    for n in nodes:
        t = n.get("type")
        if t and t.lower() in ARTIFACT_TYPES:
            rows.append(
                {
                    "arxiv_id": graph_data.get("arxiv_id", paper_meta.get("arxiv_id")),
                    "title": paper_meta.get("title"),
                    "abstract": paper_meta.get("abstract"),
                    "bucket": paper_meta.get("bucket"),
                    "artifact_id": n.get("id"),
                    "artifact_type": t.lower(),
                    "artifact_text": n.get("content") or n.get("content_preview") or "",
                }
            )
            count += 1
            if count >= max_artifacts_per_paper:
                break
    return rows


@app.command(help="Extract artifacts for previously sampled papers.")
def extract(
    run_dir: Path = OPT_RUN_DIR,
    max_artifacts_per_paper: int = OPT_MAX_ARTIFACTS_PER_PAPER,
):
    sampled = read_jsonl(run_dir / "samples" / "selected_papers.jsonl")
    if not sampled:
        raise typer.BadParameter("No sampled papers found. Run 'sample' first.")

    out_graphs_dir = run_dir / "graphs"

    async def _extract_all():
        tasks = []
        seen = set()
        for p in sampled:
            aid = p["arxiv_id"]
            if aid in seen:
                continue
            seen.add(aid)
            tasks.append(_process_one_paper(aid, out_graphs_dir))
        res = await asyncio.gather(*tasks)
        return [r for r in res if r is not None]

    graphs = asyncio.run(_extract_all())
    paper_meta_by_id = {p["arxiv_id"]: p for p in sampled}

    artifact_rows: List[Dict[str, Any]] = []
    for gd in graphs:
        aid = gd.get("arxiv_id")
        meta = paper_meta_by_id.get(aid, {})
        artifact_rows.extend(_flatten_artifacts(gd, meta, max_artifacts_per_paper))

    write_jsonl(run_dir / "artifacts.jsonl", artifact_rows)
    logger.success(
        f"Artifacts saved: {run_dir / 'artifacts.jsonl'} ({len(artifact_rows)} rows)"
    )


@app.command(help="Generate queries in three categories for each artifact.")
def gen_queries(
    run_dir: Path = OPT_RUN_DIR,
    queries_per_category: int = OPT_QUERIES_PER_CATEGORY,
    model: str = OPT_MODEL_OPENAI,
):
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is not set.")

    artifacts = read_jsonl(run_dir / "artifacts.jsonl")
    if not artifacts:
        raise typer.BadParameter("No artifacts found. Run 'extract' first.")

    async def _gen_queries_for_artifact(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for cat in ["verbatim", "reformulated", "vague"]:
            pr = prompt_for_queries(
                cat,
                row["title"],
                row["abstract"],
                row["artifact_type"],
                row["artifact_text"],
                queries_per_category,
            )
            out = await aexecute_prompt(pr, QueryGenOutput, model=model)
            for q in out.queries:
                out_rows.append(
                    {
                        "arxiv_id": row["arxiv_id"],
                        "artifact_id": row["artifact_id"],
                        "category": cat,
                        "query": q.strip(),
                        "bucket": row["bucket"],
                        "artifact_type": row["artifact_type"],
                        "title": row["title"],
                    }
                )
        return out_rows

    async def _run():
        rows: List[Dict[str, Any]] = []
        for r in artifacts:
            rows.extend(await _gen_queries_for_artifact(r))
        return rows

    queries_rows = asyncio.run(_run())
    write_jsonl(run_dir / "queries.jsonl", queries_rows)
    logger.success(
        f"Queries saved: {run_dir / 'queries.jsonl'} ({len(queries_rows)} rows)"
    )


@app.command(help="Run retrieval (closed-book and optional web-assisted).")
def retrieve(
    run_dir: Path = OPT_RUN_DIR,
    model: str = OPT_MODEL_OPENAI,
    use_web_assist: bool = OPT_WEB_ASSIST_BOOL,
    arxiv_web_top_k: int = OPT_ARXIV_WEB_TOP_K,
):
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is not set.")

    queries = read_jsonl(run_dir / "queries.jsonl")
    if not queries:
        raise typer.BadParameter("No queries found. Run 'gen-queries' first.")

    async def _retrieve_closed_book(query: str) -> RetrievalOutput:
        pr = prompt_closed_book(query)
        return await aexecute_prompt(pr, RetrievalOutput, model=model)

    async def _retrieve_web(query: str) -> Tuple[CandidatePick, List[Dict[str, Any]]]:
        cands = arxiv_search_candidates(query, top_k=arxiv_web_top_k)
        pr = prompt_web_assist(query, cands)
        out = await aexecute_prompt(pr, CandidatePick, model=model)
        return out, cands

    async def _run():
        rows_closed: List[Dict[str, Any]] = []
        rows_web: List[Dict[str, Any]] = []
        for q in queries:
            true_id = q["arxiv_id"]
            true_title = q.get("title", "")

            cb = await _retrieve_closed_book(q["query"])
            cb_ref = (
                cb.reference.dict()
                if cb and cb.reference
                else {"arxiv_id": None, "title": None}
            )
            cm = strong_match(
                cb_ref.get("arxiv_id"), cb_ref.get("title"), true_id, true_title
            )
            rows_closed.append(
                {
                    **q,
                    "mode": "closed_book",
                    "pred_arxiv_id": cb_ref.get("arxiv_id"),
                    "pred_title": cb_ref.get("title"),
                    "confidence": cb.confidence,
                    "reasoning": cb.reasoning,
                    **cm,
                }
            )

            if use_web_assist:
                pick, cands = await _retrieve_web(q["query"])
                pk_ref = (
                    pick.reference.dict()
                    if pick and pick.reference
                    else {"arxiv_id": None, "title": None}
                )
                cm2 = strong_match(
                    pk_ref.get("arxiv_id"), pk_ref.get("title"), true_id, true_title
                )
                rows_web.append(
                    {
                        **q,
                        "mode": "web_assist",
                        "pred_arxiv_id": pk_ref.get("arxiv_id"),
                        "pred_title": pk_ref.get("title"),
                        "confidence": pick.confidence,
                        "reasoning": pick.reasoning,
                        "candidates": cands,
                        **cm2,
                    }
                )
        return rows_closed, rows_web

    retrieval_closed_rows, retrieval_web_rows = asyncio.run(_run())
    write_jsonl(run_dir / "retrieval_closed.jsonl", retrieval_closed_rows)
    logger.success(
        f"Closed-book retrieval saved: {run_dir / 'retrieval_closed.jsonl'} ({len(retrieval_closed_rows)} rows)"
    )
    if use_web_assist:
        write_jsonl(run_dir / "retrieval_web.jsonl", retrieval_web_rows)
        logger.success(
            f"Web-assisted retrieval saved: {run_dir / 'retrieval_web.jsonl'} ({len(retrieval_web_rows)} rows)"
        )


@app.command(help="Evaluate results and write summaries.")
def evaluate(
    run_dir: Path = OPT_RUN_DIR,
):
    def df_safe_read_jsonl(path: Path) -> pd.DataFrame:
        rows = read_jsonl(path)
        return pd.DataFrame(rows)

    df_closed = df_safe_read_jsonl(run_dir / "retrieval_closed.jsonl")
    df_web = df_safe_read_jsonl(run_dir / "retrieval_web.jsonl")

    def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        agg = (
            df.groupby(group_cols)
            .agg(
                n=("strong_match", "size"),
                strong_match_rate=("strong_match", "mean"),
                id_match_rate=("id_match", "mean"),
                title_match_rate=("title_match", "mean"),
            )
            .reset_index()
        )
        return agg

    summary_cols = ["bucket", "artifact_type", "category", "mode"]

    if not df_closed.empty:
        df_closed["mode"] = "closed_book"
        df_closed_sum = aggregate(df_closed, summary_cols)
        df_closed_sum.to_csv(run_dir / "eval" / "summary_closed.csv", index=False)
        df_closed.to_csv(run_dir / "eval" / "per_query_closed.csv", index=False)
        logger.success(
            f"Wrote {run_dir / 'eval' / 'summary_closed.csv'} and per_query_closed.csv"
        )
    else:
        logger.warning("No closed-book results to evaluate.")

    if not df_web.empty:
        df_web["mode"] = "web_assist"
        df_web_sum = aggregate(df_web, summary_cols)
        df_web_sum.to_csv(run_dir / "eval" / "summary_web.csv", index=False)
        df_web.to_csv(run_dir / "eval" / "per_query_web.csv", index=False)
        logger.success(
            f"Wrote {run_dir / 'eval' / 'summary_web.csv'} and per_query_web.csv"
        )
    else:
        logger.info("No web-assisted results found (skipping).")


@app.command(
    help="End-to-end: sample -> extract -> gen-queries -> retrieve -> evaluate"
)
def all(
    sample_per_bucket: int = OPT_SAMPLE_PER_BUCKET_E2E,
    seed: int = OPT_SEED,
    run_name: str = OPT_RUN_NAME,
    cutoff_iso: str = OPT_CUTOFF_ISO,
    max_to_collect: int = OPT_MAX_TO_COLLECT_E2E,
    batch_size: int = OPT_BATCH_SIZE,
    max_artifacts_per_paper: int = OPT_MAX_ARTIFACTS_PER_PAPER_E2E,
    queries_per_category: int = OPT_QUERIES_PER_CATEGORY_E2E,
    model: str = OPT_MODEL_OPENAI,
    use_web_assist: bool = OPT_WEB_ASSIST_BOOL,
    arxiv_web_top_k: int = OPT_ARXIV_WEB_TOP_K,
):
    # sample
    cutoff = datetime.fromisoformat(cutoff_iso)
    max_batches = max(1, max_to_collect // max(1, batch_size))
    sampled = sample_random_math_by_id_buckets(
        per_bucket=sample_per_bucket,
        seed=seed,
        cutoff=cutoff,
        ids_batch_size=batch_size,
        max_batches=max_batches,
    )

    # Enforce desired number of papers per bucket with adaptive retries
    req = sample_per_bucket
    req_buckets = ["older", "recent", "latest_pre_cutoff", "post_cutoff"]

    def _counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c: Dict[str, int] = {b: 0 for b in req_buckets}
        for p in rows:
            b = p.get("bucket", "latest_pre_cutoff")
            c[b] = c.get(b, 0) + 1
        return c

    counts = _counts(sampled)
    missing = {b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req}

    tries = 0
    while missing and tries < 5:
        tries += 1
        more_batches = max_batches * (2**tries)
        sampled = sample_random_math_by_id_buckets(
            per_bucket=sample_per_bucket,
            seed=seed + tries,
            cutoff=cutoff,
            ids_batch_size=min(100, max(1, batch_size)),
            max_batches=more_batches,
            target_buckets=list(missing.keys()),
            existing=sampled,
        )
        counts = _counts(sampled)
        missing = {
            b: req - counts.get(b, 0) for b in req_buckets if counts.get(b, 0) < req
        }

    if missing:
        raise typer.BadParameter(
            f"Could not fill all buckets to {req} each after {tries} adaptive retries. Missing: {missing}. Try increasing --max-collect or --batch-size, or adjust --cutoff-iso."
        )

    run_dir = make_run_dir(run_name)
    config = {
        "run_name": run_name,
        "seed": seed,
        "cutoff_date": cutoff.isoformat(),
        "sample_per_bucket": sample_per_bucket,
        "max_artifacts_per_paper": max_artifacts_per_paper,
        "queries_per_category": queries_per_category,
        "use_web_assist": use_web_assist,
        "arxiv_web_top_k": arxiv_web_top_k,
        "openai_model": model,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_jsonl(run_dir / "samples" / "selected_papers.jsonl", sampled)
    logger.success(f"[SAMPLE] Saved to {run_dir}")

    # extract
    async def _extract_all():
        tasks = []
        seen = set()
        for p in sampled:
            aid = p["arxiv_id"]
            if aid in seen:
                continue
            seen.add(aid)
            tasks.append(_process_one_paper(aid, run_dir / "graphs"))
        res = await asyncio.gather(*tasks)
        return [r for r in res if r is not None]

    graphs = asyncio.run(_extract_all())
    paper_meta_by_id = {p["arxiv_id"]: p for p in sampled}
    artifact_rows: List[Dict[str, Any]] = []
    for gd in graphs:
        aid = gd.get("arxiv_id")
        meta = paper_meta_by_id.get(aid, {})
        artifact_rows.extend(_flatten_artifacts(gd, meta, max_artifacts_per_paper))
    write_jsonl(run_dir / "artifacts.jsonl", artifact_rows)
    logger.success(f"[EXTRACT] Artifacts: {len(artifact_rows)}")

    # gen-queries
    async def _gen_queries_for_artifact(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for cat in ["verbatim", "reformulated", "vague"]:
            pr = prompt_for_queries(
                cat,
                row["title"],
                row["abstract"],
                row["artifact_type"],
                row["artifact_text"],
                queries_per_category,
            )
            out = await aexecute_prompt(pr, QueryGenOutput, model=model)
            for q in out.queries:
                out_rows.append(
                    {
                        "arxiv_id": row["arxiv_id"],
                        "artifact_id": row["artifact_id"],
                        "category": cat,
                        "query": q.strip(),
                        "bucket": row["bucket"],
                        "artifact_type": row["artifact_type"],
                        "title": row["title"],
                    }
                )
        return out_rows

    async def _gen_all():
        rows: List[Dict[str, Any]] = []
        for r in artifact_rows:
            rows.extend(await _gen_queries_for_artifact(r))
        return rows

    queries_rows = asyncio.run(_gen_all())
    write_jsonl(run_dir / "queries.jsonl", queries_rows)
    logger.success(f"[QUERIES] {len(queries_rows)}")

    # retrieve
    async def _retrieve_closed_book(query: str) -> RetrievalOutput:
        pr = prompt_closed_book(query)
        return await aexecute_prompt(pr, RetrievalOutput, model=model)

    async def _retrieve_web(query: str) -> Tuple[CandidatePick, List[Dict[str, Any]]]:
        cands = arxiv_search_candidates(query, top_k=arxiv_web_top_k)
        pr = prompt_web_assist(query, cands)
        out = await aexecute_prompt(pr, CandidatePick, model=model)
        return out, cands

    async def _ret_all():
        rows_closed: List[Dict[str, Any]] = []
        rows_web: List[Dict[str, Any]] = []
        for q in queries_rows:
            true_id = q["arxiv_id"]
            true_title = q.get("title", "")

            cb = await _retrieve_closed_book(q["query"])
            cb_ref = (
                cb.reference.dict()
                if cb and cb.reference
                else {"arxiv_id": None, "title": None}
            )
            cm = strong_match(
                cb_ref.get("arxiv_id"), cb_ref.get("title"), true_id, true_title
            )
            rows_closed.append(
                {
                    **q,
                    "mode": "closed_book",
                    "pred_arxiv_id": cb_ref.get("arxiv_id"),
                    "pred_title": cb_ref.get("title"),
                    "confidence": cb.confidence,
                    "reasoning": cb.reasoning,
                    **cm,
                }
            )

            if use_web_assist:
                pick, cands = await _retrieve_web(q["query"])
                pk_ref = (
                    pick.reference.dict()
                    if pick and pick.reference
                    else {"arxiv_id": None, "title": None}
                )
                cm2 = strong_match(
                    pk_ref.get("arxiv_id"), pk_ref.get("title"), true_id, true_title
                )
                rows_web.append(
                    {
                        **q,
                        "mode": "web_assist",
                        "pred_arxiv_id": pk_ref.get("arxiv_id"),
                        "pred_title": pk_ref.get("title"),
                        "confidence": pick.confidence,
                        "reasoning": pick.reasoning,
                        "candidates": cands,
                        **cm2,
                    }
                )
        return rows_closed, rows_web

    retrieval_closed_rows, retrieval_web_rows = asyncio.run(_ret_all())
    write_jsonl(run_dir / "retrieval_closed.jsonl", retrieval_closed_rows)
    if use_web_assist:
        write_jsonl(run_dir / "retrieval_web.jsonl", retrieval_web_rows)
    logger.success(
        f"[RETRIEVE] closed={len(retrieval_closed_rows)} web={len(retrieval_web_rows) if use_web_assist else 0}"
    )

    # evaluate
    def df_safe_read_jsonl(path: Path) -> pd.DataFrame:
        rows = read_jsonl(path)
        return pd.DataFrame(rows)

    df_closed = df_safe_read_jsonl(run_dir / "retrieval_closed.jsonl")
    df_web = df_safe_read_jsonl(run_dir / "retrieval_web.jsonl")

    def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        agg = (
            df.groupby(group_cols)
            .agg(
                n=("strong_match", "size"),
                strong_match_rate=("strong_match", "mean"),
                id_match_rate=("id_match", "mean"),
                title_match_rate=("title_match", "mean"),
            )
            .reset_index()
        )
        return agg

    summary_cols = ["bucket", "artifact_type", "category", "mode"]

    if not df_closed.empty:
        df_closed["mode"] = "closed_book"
        df_closed_sum = aggregate(df_closed, summary_cols)
        df_closed_sum.to_csv(run_dir / "eval" / "summary_closed.csv", index=False)
        df_closed.to_csv(run_dir / "eval" / "per_query_closed.csv", index=False)

    if not df_web.empty:
        df_web["mode"] = "web_assist"
        df_web_sum = aggregate(df_web, summary_cols)
        df_web_sum.to_csv(run_dir / "eval" / "summary_web.csv", index=False)
        df_web.to_csv(run_dir / "eval" / "per_query_web.csv", index=False)

    logger.success(f"[EVAL] Wrote summaries to {run_dir / 'eval'}")
    typer.echo(str(run_dir))


# ----- ID-based sampling (by arXiv identifier) -----


def parse_date_from_arxiv_id(arxiv_id: str) -> Optional[datetime]:
    """
    Parse a naive timestamp from an arXiv identifier.

    Supports:
      - New style: YYMM.NNNNN (e.g., 2507.05087v1) => year=2000+YY, month=MM
      - Old style: cat/YYMMNNN (e.g., math.AG/0601001v2) => year=2000+YY, month=MM

    Returns a timezone-aware datetime in UTC at the 1st of the inferred month,
    or None if parsing fails.
    """
    aid = arxiv_id.strip()
    # Strip version if present
    aid = aid.split("v")[0]

    # New style: e.g., 2507.05087
    m_new = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", aid)
    if m_new:
        yy, mm = int(m_new.group(1)), int(m_new.group(2))
        year = 2000 + yy
        month = max(1, min(12, mm))
        return datetime(year, month, 1, tzinfo=timezone.utc)

    # Old style: e.g., math.AG/0601001
    m_old = re.match(r"^[a-z\-]+(\.[A-Z]{2,})?/(\d{2})(\d{2})\d{3}$", aid)
    if m_old:
        yy, mm = int(m_old.group(2)), int(m_old.group(3))
        year = 2000 + yy
        month = max(1, min(12, mm))
        return datetime(year, month, 1, tzinfo=timezone.utc)

    return None


def stratified_sample_ids(
    ids: List[str], per_bucket: int, seed: int, cutoff: datetime
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Stratify a raw list of arXiv identifiers into the same 4 buckets using dates inferred from IDs.

    Returns (sampled_ids, buckets_map) where buckets_map has keys:
      {"older","recent","latest_pre_cutoff","post_cutoff"} and values are lists of ids.
    """
    random.seed(seed)
    buckets: Dict[str, List[str]] = {
        "older": [],
        "recent": [],
        "latest_pre_cutoff": [],
        "post_cutoff": [],
    }

    for aid in ids:
        dt = parse_date_from_arxiv_id(aid)
        if not dt:
            # Could not parse — skip it
            continue
        b = bucket_for_date(dt, cutoff)
        buckets[b].append(aid)

    sampled: List[str] = []
    for _, blist in buckets.items():
        if not blist:
            continue
        k = min(per_bucket, len(blist))
        sampled_bucket = random.sample(blist, k)
        sampled.extend(sampled_bucket)

    return sampled, buckets


def fetch_papers_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch metadata for a list of arXiv IDs using id_list. Returns 'papers' dicts.
    """
    if not ids:
        return []
    api = ArxivAPI()
    # arXiv API allows many IDs; use conservative batching
    batch_size = 100
    out: List[Dict[str, Any]] = []
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        resp = api.fetch_by_ids(chunk)
        n, _, entries = api.parse_response(resp)
        if n and entries:
            out.extend(entries_to_papers(entries, api=api))
    api.close()
    return out


def fallback_fill_missing_with_query(
    sampled: List[Dict[str, Any]],
    missing: Dict[str, int],
    cutoff: datetime,
    seen_ids: Optional[set] = None,
    batch_size: int = 100,
    max_scan: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Fallback strategy to fill remaining buckets if random ID sampling underfills.
    Scans the math feed (cat:math.*) in submittedDate order and selects items whose
    arXiv ID date falls into the missing buckets.
    """
    seen = set(seen_ids or [])
    to_fill = {b: int(v) for b, v in missing.items() if v > 0}
    if not to_fill:
        return sampled

    api = ArxivAPI()
    start = 0
    while to_fill and start < max_scan:
        resp = api.fetch_papers(
            "cat:math.*", start=start, batch_size=min(100, max(1, batch_size))
        )
        n, _, entries = api.parse_response(resp)
        if n == 0 or not entries:
            break
        papers = entries_to_papers(entries, api=api)
        for p in papers:
            aid = p["arxiv_id"]
            if aid in seen:
                continue
            dt = parse_date_from_arxiv_id(aid)
            if not dt:
                continue
            b = bucket_for_date(dt, cutoff)
            if b in to_fill and to_fill[b] > 0:
                p["bucket"] = b
                sampled.append(p)
                seen.add(aid)
                to_fill[b] -= 1
        start += n
    api.close()
    return sampled


# ----- Random ID sampling by time buckets -----


def month_list(start: datetime, end: datetime) -> List[Tuple[int, int]]:
    months: List[Tuple[int, int]] = []
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y < end_y) or (y == end_y and m <= end_m):
        months.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def bucket_month_ranges(cutoff: datetime) -> Dict[str, Tuple[datetime, datetime]]:
    # Use new-style IDs for random generation.
    # New-style IDs began in 2007; use 2007-04 onward to be safe.
    older_start = datetime(2007, 4, 1, tzinfo=timezone.utc)
    older_end = datetime(2014, 12, 1, tzinfo=timezone.utc)
    recent_start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    recent_end = datetime(2020, 12, 1, tzinfo=timezone.utc)
    latest_start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    latest_end = datetime(2023, 12, 1, tzinfo=timezone.utc)

    post_start = datetime(cutoff.year, cutoff.month, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    post_end = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    return {
        "older": (older_start, older_end),
        "recent": (recent_start, recent_end),
        "latest_pre_cutoff": (latest_start, latest_end),
        "post_cutoff": (post_start, post_end),
    }


def build_random_id_for_month(year: int, month: int, rng: random.Random) -> str:
    return f"{year % 100:02d}{month:02d}.{rng.randint(1, 99999):05d}"


def is_math_paper(p: Dict[str, Any]) -> bool:
    pc = (p.get("primary_category") or "").lower()
    if pc.startswith("math"):
        return True
    for c in p.get("all_categories") or []:
        if str(c).lower().startswith("math"):
            return True
    return False


def sample_random_math_by_id_buckets(
    per_bucket: int,
    seed: int,
    cutoff: datetime,
    ids_batch_size: int = 100,
    max_batches: int = 200,
    target_buckets: Optional[List[str]] = None,
    existing: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Randomly generate arXiv IDs per time bucket (using new-style YYMM.NNNNN),
    fetch existing records, keep only math papers, and collect up to per_bucket per bucket.
    If 'existing' is provided, it will be extended, and only the buckets listed in
    'target_buckets' will be attempted (defaults to all buckets).
    """
    rng = random.Random(seed)
    ranges = bucket_month_ranges(cutoff)

    results: List[Dict[str, Any]] = list(existing) if existing else []
    seen_ids: set[str] = {p["arxiv_id"] for p in results}
    buckets_to_do = target_buckets or [
        "older",
        "recent",
        "latest_pre_cutoff",
        "post_cutoff",
    ]

    # Track how many we already have per bucket from 'existing'
    existing_counts: Dict[str, int] = {
        b: 0 for b in ["older", "recent", "latest_pre_cutoff", "post_cutoff"]
    }
    for _p in results:
        b = _p.get("bucket", "latest_pre_cutoff")
        if b in existing_counts:
            existing_counts[b] += 1

    for bucket in buckets_to_do:
        # Determine remaining needed for this bucket
        need = max(0, per_bucket - existing_counts.get(bucket, 0))
        if need == 0:
            continue
        target = need
        collected = 0
        start_dt, end_dt = ranges[bucket]
        months = month_list(start_dt, end_dt)
        start_idx = rng.randrange(len(months)) if months else 0
        if not months:
            logger.warning(f"No months available for bucket={bucket}")
            continue

        batches = 0
        while collected < target and batches < max_batches:
            y, m = months[(start_idx + batches) % len(months)]

            # Generate a candidate pool of random IDs for this month (cap to API limit 100)
            per_call = min(100, max(1, ids_batch_size))
            cand_ids: set[str] = set()
            while len(cand_ids) < per_call:
                cand_ids.add(build_random_id_for_month(y, m, rng))

            papers = fetch_papers_by_ids(list(cand_ids))

            for p in papers:
                aid = p["arxiv_id"]
                if aid in seen_ids:
                    continue
                if not is_math_paper(p):
                    continue
                dt = parse_date_from_arxiv_id(aid)
                if not dt:
                    continue
                if bucket_for_date(dt, cutoff) != bucket:
                    continue

                p["bucket"] = bucket
                results.append(p)
                seen_ids.add(aid)
                collected += 1
                if collected >= target:
                    break

            batches += 1

        existing_counts[bucket] = existing_counts.get(bucket, 0) + collected
        if collected < target:
            logger.warning(
                f"Bucket '{bucket}': filled {collected}/{target} after {batches} batches."
            )
        else:
            logger.info(f"Bucket '{bucket}': collected {collected} papers.")

    return results


@app.command(
    help="ID-based sampling: provide a list of arXiv IDs, stratify by ID-encoded date, then fetch metadata."
)
def sample_ids(
    id_file: Optional[Path] = OPT_ID_FILE,
    ids_csv: Optional[str] = OPT_IDS_CSV,
    sample_per_bucket: int = OPT_SAMPLE_SAMPLE_PER_BUCKET,
    seed: int = OPT_SEED,
    run_name: str = OPT_RUN_NAME,
    cutoff_iso: str = OPT_CUTOFF_ISO,
):
    cutoff = datetime.fromisoformat(cutoff_iso)

    # Load IDs
    ids: List[str] = []
    if id_file and id_file.exists():
        for line in id_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.append(line)
    if ids_csv:
        ids.extend([x.strip() for x in ids_csv.split(",") if x.strip()])

    if not ids:
        raise typer.BadParameter("No IDs provided. Use --id-file and/or --ids.")

    # Stratified sample purely from IDs
    sampled_ids, buckets = stratified_sample_ids(ids, sample_per_bucket, seed, cutoff)

    # Fetch metadata for sampled IDs to persist title/abstract for downstream steps
    sampled_papers = fetch_papers_by_ids(sampled_ids)
    # Attach bucket to each paper
    bucket_by_id = {}
    for bname, blist in buckets.items():
        for aid in blist:
            bucket_by_id[aid] = bname
    for p in sampled_papers:
        p["bucket"] = bucket_by_id.get(p["arxiv_id"], "latest_pre_cutoff")

    run_dir = make_run_dir(run_name)
    config = {
        "run_name": run_name,
        "seed": seed,
        "cutoff_date": cutoff.isoformat(),
        "arxiv_query": None,
        "sample_per_bucket": sample_per_bucket,
        "sampling_mode": "by_id",
        "max_artifacts_per_paper": 10,
        "queries_per_category": 3,
        "use_web_assist": False,
        "arxiv_web_top_k": 5,
        "openai_model": "gpt-5-2025-08-07",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_jsonl(run_dir / "samples" / "selected_papers.jsonl", sampled_papers)
    logger.success(f"[SAMPLE_IDS] Run dir: {run_dir}")
    logger.success(
        f"[SAMPLE_IDS] Saved sampled papers: {run_dir / 'samples' / 'selected_papers.jsonl'}"
    )


@app.command(
    help="Generate-then-verify sampling and run the pipeline end-to-end in-memory."
)
def sample_and_run(
    n_per_bucket: int = OPT_N_PER_BUCKET,
    run_name: str = OPT_RUN_NAME_GNV,
    max_artifacts_per_paper: int = OPT_MAX_ARTIFACTS_PER_PAPER,
    queries_per_category: int = OPT_QUERIES_PER_CATEGORY,
    n_successful_papers_per_bucket: int = OPT_MIN_OK_PER_BUCKET,
    model: str = OPT_MODEL_GPT4_TURBO,
    concurrency: int = OPT_CONCURRENCY,
):
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is not set.")

    buckets = sample_by_generating_and_verifying(n_per_bucket=n_per_bucket)

    asyncio.run(
        run_in_memory_experiment(
            sampled_id_buckets=buckets,
            run_name=run_name,
            max_artifacts_per_paper=max_artifacts_per_paper,
            queries_per_category=queries_per_category,
            n_successful_papers_per_bucket=n_successful_papers_per_bucket,
            model=model,
            concurrency_limit=concurrency,
        )
    )


def main():
    # Quiet richer logs in non-TUI environments if desired
    # os.environ["RICH_QUIET"] = "True"
    app()


if __name__ == "__main__":
    main()
