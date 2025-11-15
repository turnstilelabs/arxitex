from __future__ import annotations

import asyncio
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from arxitex.arxiv_api import ArxivAPI
from arxitex.experiments.prompts_generate_then_verify import (
    prompt_closed_book,
    prompt_for_queries,
)
from arxitex.extractor.pipeline import agenerate_artifact_graph
from arxitex.llms.llms import aexecute_prompt

EXPERIMENTS_ROOT = Path("experiments")


def make_run_dir(run_name: str) -> Path:
    """Creates a timestamped directory for an experiment run."""
    run_dir = (
        EXPERIMENTS_ROOT / f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "logs" / "experiment.log")
    return run_dir


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_id_cache(path: Path) -> Dict[str, List[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_id_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def strong_match(
    pred_id: Optional[str], pred_title: Optional[str], true_id: str, true_title: str
) -> Dict[str, int]:
    id_match = 1 if pred_id and pred_id.strip() == true_id.strip() else 0
    title_match = (
        1
        if (pred_title and normalize_text(pred_title) == normalize_text(true_title))
        else 0
    )
    return {
        "id_match": id_match,
        "title_match": title_match,
        "strong_match": 1 if (id_match and title_match) else 0,
    }


# --------------------------- ArXiv parsing helpers ---------------------------


def parse_response_for_math_ids(response_text: str, api: ArxivAPI) -> List[str]:
    """Parses the API response and returns a list of valid math paper IDs (with version)."""
    if not response_text:
        return []
    try:
        root = ET.fromstring(response_text)
        ids: List[str] = []
        for entry in root.findall(".//atom:entry", api.ns):
            primary_cat_elem = entry.find("arxiv:primary_category", api.ns)
            category = (
                primary_cat_elem.get("term", "") if primary_cat_elem is not None else ""
            )
            if category.startswith("math"):
                arxiv_id_with_version = entry.find("atom:id", api.ns).text.split("/abs/")[-1]  # type: ignore
                ids.append(arxiv_id_with_version)
        return ids
    except ET.ParseError:
        return []


def parse_full_paper_details(
    response_text: str, api: ArxivAPI, bucket_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Parses an API response and returns full paper metadata with bucket."""
    if not response_text:
        return []
    try:
        root = ET.fromstring(response_text)
        papers: List[Dict[str, Any]] = []
        for entry in root.findall(".//atom:entry", api.ns):
            arxiv_id_v = entry.find("atom:id", api.ns).text.split("/abs/")[-1]  # type: ignore
            title = (entry.find("atom:title", api.ns).text or "").strip().replace("\n", " ")  # type: ignore
            abstract = (entry.find("atom:summary", api.ns).text or "").strip().replace("\n", " ")  # type: ignore
            published = entry.find("atom:published", api.ns).text or ""  # type: ignore
            papers.append(
                {
                    "arxiv_id": arxiv_id_v,
                    "title": title,
                    "abstract": abstract,
                    "published_date": published,
                    "bucket": bucket_map.get(arxiv_id_v, "unknown"),
                }
            )
        return papers
    except ET.ParseError:
        return []


# ------------------------ Synthetic ID generation ------------------------


def generate_synthetic_arxiv_id(submission_date: datetime) -> str:
    """
    Generates a syntactically correct, but FAKE, arXiv ID based on its format rules.
    This correctly handles the 4-digit vs. 5-digit sequence number change in 2015.
    """
    yy = submission_date.year % 100
    mm = submission_date.month

    if submission_date.year < 2015:
        sequence_num = random.randint(1, 9999)
        sequence_str = f"{sequence_num:04d}"
    else:
        # Increase probability of lower sequences to improve hit rate for recent years.
        if random.random() < 0.7:
            sequence_num = random.randint(1, 5000)
        else:
            sequence_num = random.randint(5001, 40000)
        sequence_str = f"{sequence_num:05d}"

    return f"{yy:02d}{mm:02d}.{sequence_str}"  # versionless


def random_date_in_range(start: datetime, end: datetime) -> datetime:
    """Generates a random datetime within a given range."""
    return start + (end - start) * random.random()


def sample_by_generating_and_verifying(
    n_per_bucket: int,
    verify_sleep: float = 3.0,
    cache_path: Optional[Path] = EXPERIMENTS_ROOT / "gnv_id_cache.json",
    use_cache: bool = True,
    persist_cache: bool = True,
) -> Dict[str, List[str]]:
    """
    Implements the 'generate-then-verify' workflow to sample arXiv IDs in 'math'.
    Returns a mapping {bucket_name: [arxiv_id_with_version]}.
    """
    bucket_definitions = {
        "older": (
            datetime(2010, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 12, 31, tzinfo=timezone.utc),
        ),
        "recent": (
            datetime(2021, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 10, 31, tzinfo=timezone.utc),
        ),
        "post": (
            datetime(2024, 11, 1, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
        ),
    }

    final_buckets: Dict[str, List[str]] = {name: [] for name in bucket_definitions}
    api = ArxivAPI()
    BATCH_SIZE = 20

    # Load/prepare cache of validated IDs
    cache_sets: Dict[str, set] = {name: set() for name in bucket_definitions}
    if use_cache and cache_path and Path(cache_path).exists():
        cached = load_id_cache(Path(cache_path))
        for _name in bucket_definitions:
            for _pid in cached.get(_name, []):
                cache_sets[_name].add(_pid)

    # Prefill from cache to avoid unnecessary verification
    if use_cache:
        for _name in bucket_definitions:
            if len(final_buckets[_name]) < n_per_bucket and cache_sets.get(_name):
                prefill = list(cache_sets[_name])
                random.shuffle(prefill)
                for _pid in prefill:
                    if len(final_buckets[_name]) >= n_per_bucket:
                        break
                    # ensure uniqueness
                    if _pid not in final_buckets[_name]:
                        final_buckets[_name].append(_pid)

    try:
        for name, (start_dt, end_dt) in bucket_definitions.items():
            logger.info(
                f"--- Populating bucket: '{name}' ({start_dt.date()} to {end_dt.date()}) ---"
            )

            while len(final_buckets[name]) < n_per_bucket:
                generated_ids_batch = {
                    generate_synthetic_arxiv_id(random_date_in_range(start_dt, end_dt))
                    for _ in range(BATCH_SIZE)
                }

                response_text = api.fetch_by_ids(list(generated_ids_batch))
                found_math_ids = parse_response_for_math_ids(response_text, api)

                if found_math_ids:
                    for paper_id in found_math_ids:
                        if (
                            len(final_buckets[name]) < n_per_bucket
                            and paper_id not in final_buckets[name]
                        ):
                            logger.success(f"Found valid math paper: {paper_id}")
                            final_buckets[name].append(paper_id)
                            cache_sets[name].add(paper_id)

                logger.info(
                    f"Bucket '{name}' status: {len(final_buckets[name])} / {n_per_bucket}"
                )
                logger.trace(f"Waiting {verify_sleep} seconds before next API call...")
                time.sleep(verify_sleep)
    finally:
        api.close()

    if persist_cache and cache_path:
        # Merge and persist cache to disk
        to_save: Dict[str, List[str]] = {}
        for _name in bucket_definitions:
            merged = set(cache_sets.get(_name, set())) | set(
                final_buckets.get(_name, [])
            )
            to_save[_name] = sorted(list(merged))
        save_id_cache(Path(cache_path), to_save)

    return final_buckets


# --------------------------- Pydantic models ---------------------------


class QueryGenOutput(BaseModel):
    queries: List[str] = Field(default_factory=list)


class PredictedReference(BaseModel):
    title: Optional[str] = None


class RetrievalCandidate(BaseModel):
    reference: PredictedReference
    confidence: float
    reasoning: str


class MultiRetrievalOutput(BaseModel):
    candidates: List[RetrievalCandidate] = Field(default_factory=list)


# ------------------------------ Evaluation ------------------------------


def evaluate_title_match(pred_title: Optional[str], true_title: str) -> Dict[str, int]:
    if not pred_title:
        return {"exact_title_match": 0}
    return {
        "exact_title_match": int(
            normalize_text(pred_title) == normalize_text(true_title)
        )
    }


def evaluate_retrieval_list(
    predicted_candidates: List[RetrievalCandidate], true_title: str
) -> Dict[str, Any]:
    norm_true = normalize_text(true_title)
    for i, candidate in enumerate(predicted_candidates):
        pred_title = candidate.reference.title
        if not pred_title:
            continue
        if normalize_text(pred_title) == norm_true:
            return {"is_in_top_k": 1, "rank_if_found": i + 1}
    return {"is_in_top_k": 0, "rank_if_found": -1}


# ------------------------------ Main pipeline ------------------------------


async def run_in_memory_experiment(
    sampled_id_buckets: Dict[str, List[str]],
    run_name: str = "math-llm-retrieval",
    max_artifacts_per_paper: int = 10,
    queries_per_category: int = 3,
    queries_concurrency: int = 8,
    n_successful_papers_per_bucket: int = 5,
    model: str = "gpt-5-mini-2025-08-07",
    concurrency_limit: int = 8,
):
    """
    Runs the experiment pipeline ensuring that each bucket has at least
    `n_successful_papers_per_bucket` successfully processed papers.
    """
    run_dir = make_run_dir(run_name)
    logger.info(f"Starting run: {run_name} | Saving to {run_dir}")

    detailed_logs: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency_limit)
    queries_semaphore = asyncio.Semaphore(queries_concurrency)
    api = ArxivAPI()

    # Flatten all IDs for initial metadata fetch
    all_ids = [pid for pids in sampled_id_buckets.values() for pid in pids]
    bucket_map = {
        pid: name for name, pids in sampled_id_buckets.items() for pid in pids
    }
    logger.info(f"Fetching metadata for {len(all_ids)} papers...")
    response_text = api.fetch_by_ids(all_ids)
    papers_with_metadata = parse_full_paper_details(response_text, api, bucket_map)
    api.close()
    if not papers_with_metadata:
        logger.error("Could not fetch metadata. Aborting.")
        return

    logger.success(f"Got metadata for {len(papers_with_metadata)} papers.")

    # Group by bucket
    papers_by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for paper in papers_with_metadata:
        bucket = paper.get("bucket", "unknown")
        papers_by_bucket.setdefault(bucket, []).append(paper)

    final_results: List[Dict[str, Any]] = []

    # --- Loop per bucket and enforce minimum successful count ---
    for bucket_name, papers in papers_by_bucket.items():
        logger.info(
            f"Processing bucket '{bucket_name}' ({len(papers)} papers total)..."
        )
        successful_papers = 0
        processed_results: List[Dict[str, Any]] = []

        for paper_meta in papers:
            if successful_papers >= n_successful_papers_per_bucket:
                break

            try:
                # --- Step 2: Extract Artifacts ---
                results = await agenerate_artifact_graph(
                    arxiv_id=paper_meta["arxiv_id"],
                    infer_dependencies=False,
                    enrich_content=False,
                )
                graph = results.get("graph")
                if not graph:
                    continue
                graph_data = graph.to_dict(arxiv_id=paper_meta["arxiv_id"])
                (
                    run_dir
                    / "graphs"
                    / f"{paper_meta['arxiv_id'].replace('/', '_')}.json"
                ).write_text(
                    json.dumps(graph_data, ensure_ascii=False), encoding="utf-8"
                )

                # Build artifact rows: focus on theorems primarily
                artifact_rows: List[Dict[str, Any]] = []
                for n in graph_data.get("nodes", [])[:max_artifacts_per_paper]:
                    t = (n.get("type") or "").lower()
                    if t == "theorem":
                        artifact_rows.append(
                            {
                                **paper_meta,
                                "artifact_id": n.get("id"),
                                "artifact_type": t,
                                "artifact_text": n.get("content", ""),
                            }
                        )
                if not artifact_rows:
                    continue

                # --- Step 3: Query Generation (concurrent) ---
                query_rows: List[Dict[str, Any]] = []

                async def _gen_queries_task(
                    art: Dict[str, Any], cat: str
                ) -> List[Dict[str, Any]]:
                    async with queries_semaphore:
                        pr = prompt_for_queries(
                            cat,
                            art["artifact_type"],
                            art["artifact_text"],
                            queries_per_category,
                        )
                        out = await aexecute_prompt(pr, QueryGenOutput, model=model)
                        return [
                            {**art, "category": cat, "query": q.strip()}
                            for q in out.queries
                        ]

                gen_tasks = [
                    _gen_queries_task(art, cat)
                    for art in artifact_rows
                    for cat in [
                        "precise_assertion",
                        "imperfect_recall",
                        "conceptual_search",
                        "exploratory_search",
                    ]
                ]
                if gen_tasks:
                    gen_results = await asyncio.gather(*gen_tasks)
                    for rows in gen_results:
                        query_rows.extend(rows)

                if not query_rows:
                    continue

                # --- Step 4: Retrieval ---
                async def _retrieve_task(q_row: Dict[str, Any]):
                    async with semaphore:
                        output = await aexecute_prompt(
                            prompt_closed_book(q_row["query"]),
                            MultiRetrievalOutput,
                            model=model,
                        )
                        match_metrics = evaluate_retrieval_list(
                            output.candidates, q_row["title"]
                        )
                        detailed_logs.append(
                            {
                                "arxiv_id": q_row["arxiv_id"],
                                "arxiv_title": q_row.get("title", ""),
                                "bucket": q_row["bucket"],
                                "artifact_id": q_row["artifact_id"],
                                "artifact_text": q_row.get("artifact_text", ""),
                                "query": q_row["query"],
                                "category": q_row["category"],
                                "predicted_candidates": [
                                    c.model_dump() for c in output.candidates
                                ],
                                **match_metrics,
                            }
                        )
                        return {**q_row, "mode": "closed_book_top5", **match_metrics}

                retrieval_results = await asyncio.gather(
                    *[_retrieve_task(q) for q in query_rows]
                )
                processed_results.extend(retrieval_results)
                successful_papers += 1
                logger.info(
                    f"{bucket_name}: {successful_papers} papers fully processed so far."
                )

            except Exception as e:
                logger.error(f"Error processing {paper_meta['arxiv_id']}: {e}")

        if successful_papers < n_successful_papers_per_bucket:
            logger.warning(
                f"Bucket '{bucket_name}' reached only {successful_papers}/{n_successful_papers_per_bucket} successful papers."
            )
        else:
            logger.success(
                f"Bucket '{bucket_name}' reached {successful_papers} successful papers!"
            )

        final_results.extend(processed_results)

    # --- Step 5: Evaluation ---
    logger.info("Compiling evaluation results...")
    if detailed_logs:
        df_detailed = pd.DataFrame(detailed_logs)
        detailed_log_path = run_dir / "eval" / "detailed_log.csv"
        await asyncio.to_thread(df_detailed.to_csv, detailed_log_path, index=False)
        logger.success(
            f"Wrote {len(df_detailed)} detailed log entries to {detailed_log_path}"
        )

    df_results = pd.DataFrame(final_results)
    if not df_results.empty:
        df_summary = (
            df_results.groupby(["bucket", "artifact_type", "category"])
            .agg(n=("is_in_top_k", "size"), recall_at_5_rate=("is_in_top_k", "mean"))
            .reset_index()
        )
        out_path = run_dir / "eval" / "summary.csv"
        await asyncio.to_thread(df_summary.to_csv, out_path, index=False)
        logger.success(f"Wrote summary to {out_path}")

    logger.success(f"Experiment completed. Results in {run_dir}")


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate-then-verify arXiv math retrieval experiment (sampling + async pipeline)."
    )
    parser.add_argument(
        "-n",
        "--n-per-bucket",
        type=int,
        default=10,
        help="Number of valid math IDs to collect per bucket.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="math-llm-retrieval-gnv",
        help="Run name prefix for output directory.",
    )
    parser.add_argument(
        "--max-artifacts-per-paper",
        type=int,
        default=10,
        help="Cap artifacts per paper to include.",
    )
    parser.add_argument("-k", "--k", type=int, default=3, help="Queries per category.")
    parser.add_argument(
        "--min-ok-per-bucket",
        type=int,
        default=5,
        help="Minimum successfully processed papers per bucket.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="OpenAI model used for prompts.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Async concurrency limit for retrieval calls.",
    )
    parser.add_argument(
        "--queries-concurrency",
        type=int,
        default=8,
        help="Async concurrency limit for query generation.",
    )
    parser.add_argument(
        "--verify-sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between arXiv verify batches.",
    )
    parser.add_argument(
        "--id-cache-path",
        type=str,
        default=str(EXPERIMENTS_ROOT / "gnv_id_cache.json"),
        help="Path to JSON cache file for validated arXiv IDs.",
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        default=True,
        help="Use cache of validated IDs (default).",
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable reading cache of validated IDs.",
    )
    parser.add_argument(
        "--persist-cache",
        dest="persist_cache",
        action="store_true",
        default=True,
        help="Persist validated IDs to cache (default).",
    )
    parser.add_argument(
        "--no-persist-cache",
        dest="persist_cache",
        action="store_false",
        help="Do not write validated IDs to cache.",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", flush=True)
        raise SystemExit(2)

    buckets = sample_by_generating_and_verifying(
        n_per_bucket=args.n_per_bucket,
        verify_sleep=args.verify_sleep,
        cache_path=Path(args.id_cache_path),
        use_cache=args.use_cache,
        persist_cache=args.persist_cache,
    )
    asyncio.run(
        run_in_memory_experiment(
            sampled_id_buckets=buckets,
            run_name=args.run_name,
            max_artifacts_per_paper=args.max_artifacts_per_paper,
            queries_per_category=args.k,
            queries_concurrency=args.queries_concurrency,
            n_successful_papers_per_bucket=args.min_ok_per_bucket,
            model=args.model,
            concurrency_limit=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
