"""Export processed papers from the SQLite DB into per-paper JSON files
for the ArxiGraph webapp / Hugging Face dataset.

This tool **does not** re-run extraction or LLM calls. Instead, it:

  - discovers which papers have been successfully processed via
    ``ProcessedIndex`` (processed_papers table),
  - reconstructs the document graph from the normalized SQLite schema using
    ``load_document_graph``, and
  - rebuilds the definition bank + artifact→terms mapping from the
    ``definitions``, ``definition_aliases``, ``definition_dependencies`` and
    ``artifact_terms`` tables.

For each selected arXiv ID it writes a JSON file of the form::

    {
      "graph": { ... DocumentGraph.to_dict(...) ... },
      "definition_bank": { ... } | null,
      "artifact_to_terms_map": { "artifact_id": ["term1", ...], ... },
      "latex_macros": { "cF": "\\mathcal{F}", ... }
    }

The filename convention matches the Hugging Face dataset layout used by
the Next.js frontend::

    arxiv_{arxiv_id.replace('/', '_')}.json

Example usage::

    python -m arxitex.tools.export_hf_dataset \
      --db-path path/to/arxitex.sqlite \
      --output-dir /path/to/hf-dataset/data

    # Restrict to a subset of IDs
    python -m arxitex.tools.export_hf_dataset \
      --db-path path/to/arxitex.sqlite \
      --output-dir /path/to/hf-dataset/data \
      --only-arxiv-id 2103.14030 --only-arxiv-id 2211.11689
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from loguru import logger

from arxitex.db.connection import connect
from arxitex.db.persistence import load_document_graph
from arxitex.db.schema import ensure_schema
from arxitex.indices.processed import ProcessedIndex


def _safe_id_to_arxiv_id(safe_id: str) -> str:
    """Inverse of the output filename normalization.

    We export files using `arxiv_{arxiv_id.replace('/', '_')}.json`.
    """

    return safe_id.replace("_", "/")


def _arxiv_id_to_safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _parse_arxiv_id_from_repo_path(repo_path: str) -> str | None:
    """Parse an arXiv id from a HF repo path like `data/arxiv_2211.1234v1.json`."""

    p = repo_path.replace("\\", "/")
    if not p.startswith("data/"):
        return None
    name = p.split("/", 1)[1]
    if not (name.startswith("arxiv_") and name.endswith(".json")):
        return None
    safe_id = name[len("arxiv_") : -len(".json")]
    if not safe_id:
        return None
    return _safe_id_to_arxiv_id(safe_id)


def _list_remote_exported_arxiv_ids(
    *, repo_id: str, repo_type: str, revision: str | None, token: str | None
) -> Set[str]:
    """List arxiv_ids already present on HF under `data/arxiv_*.json`."""

    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to list remote files. Install it with 'pip install huggingface-hub'."
        ) from e

    api = HfApi(token=token) if token else HfApi()
    repo_files = api.list_repo_files(repo_id, repo_type=repo_type, revision=revision)
    out: Set[str] = set()
    for f in repo_files:
        arxiv_id = _parse_arxiv_id_from_repo_path(f)
        if arxiv_id:
            out.add(arxiv_id)
    return out


def _iter_successful_arxiv_ids(processed_index: ProcessedIndex) -> Iterable[str]:
    """Yield all arxiv_ids with a successful status from processed_papers.

    We intentionally bypass the high-level helpers and query the underlying
    table directly so we can iterate over the whole corpus efficiently.
    """

    with processed_index._get_connection() as conn:  # type: ignore[attr-defined]
        cur = conn.execute(
            "SELECT arxiv_id, status FROM processed_papers ORDER BY processed_timestamp_utc"
        )
        for row in cur.fetchall():
            status = str(row["status"] or "")
            if status.startswith("success"):
                yield str(row["arxiv_id"])


def _load_definition_bank_and_mappings(
    db_path: str | Path, paper_id: str
) -> tuple[Dict[str, Dict[str, Any]] | None, Dict[str, list[str]]]:
    """Reconstruct definition_bank and artifact_to_terms_map from SQLite.

    This mirrors the shape produced by DefinitionBank.to_dict() and the
    artifact_to_terms_map originally passed into replace_definitions_and_mappings.
    """

    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        # Base definitions
        defs: Dict[str, Dict[str, Any]] = {}

        drows = conn.execute(
            """
            SELECT term_canonical, term_original, definition_text, source_artifact_id
            FROM definitions
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchall()

        for r in drows:
            term_canonical = str(r["term_canonical"])
            defs[term_canonical] = {
                "term": str(r["term_original"]),
                "aliases": [],  # filled below
                "definition_text": str(r["definition_text"] or ""),
                "source_artifact_id": r["source_artifact_id"],
                "dependencies": [],  # filled below
            }

        if not defs:
            # No definition content persisted for this paper.
            return None, {}

        # Aliases
        arows = conn.execute(
            """
            SELECT term_canonical, alias
            FROM definition_aliases
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchall()

        for r in arows:
            term_canonical = str(r["term_canonical"])
            alias = str(r["alias"])
            if term_canonical in defs:
                defs[term_canonical].setdefault("aliases", []).append(alias)

        # Dependencies (stored as canonical terms)
        dep_rows = conn.execute(
            """
            SELECT term_canonical, depends_on_term_canonical
            FROM definition_dependencies
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchall()

        for r in dep_rows:
            term_canonical = str(r["term_canonical"])
            dep_canonical = str(r["depends_on_term_canonical"])
            if term_canonical in defs:
                defs[term_canonical].setdefault("dependencies", []).append(
                    dep_canonical
                )

        # artifact_to_terms_map: map artifact_id -> [term_raw, ...]
        artifact_to_terms: Dict[str, list[str]] = {}
        trows = conn.execute(
            """
            SELECT artifact_id, term_raw
            FROM artifact_terms
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchall()

        for r in trows:
            aid = str(r["artifact_id"])
            term_raw = str(r["term_raw"])
            artifact_to_terms.setdefault(aid, []).append(term_raw)

        return defs, artifact_to_terms
    finally:
        conn.close()


def export_paper(
    *, db_path: str | Path, arxiv_id: str, output_dir: str | Path
) -> Path | None:
    """Export a single successfully processed paper to a JSON file.

    Returns the path to the written file, or ``None`` if nothing was exported
    (e.g. no graph nodes found).
    """

    logger.info(f"Exporting {arxiv_id} from DB → JSON...")

    # 1) Rebuild the graph from SQLite
    graph = load_document_graph(
        db_path=db_path, paper_id=arxiv_id, include_prerequisites=True
    )

    if not graph.nodes:
        logger.warning(f"Paper {arxiv_id} has an empty graph; skipping export.")
        return None

    # 2) Reconstruct definition bank + artifact_to_terms_map
    definition_bank, artifact_to_terms_map = _load_definition_bank_and_mappings(
        db_path, arxiv_id
    )

    # 3) Derive extractor_mode from paper_ingestion_state
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT mode
            FROM paper_ingestion_state
            WHERE paper_id = ? AND stage = 'complete'
            ORDER BY updated_at_utc DESC
            LIMIT 1
            """,
            (arxiv_id,),
        ).fetchone()
    finally:
        conn.close()

    mode = str(row["mode"]) if row and row["mode"] is not None else None
    if mode == "regex":
        extractor_mode = "regex-only"
    elif mode == "defs":
        extractor_mode = "hybrid (content-only)"
    elif mode == "full":
        extractor_mode = "full-hybrid (deps + content)"
    else:
        extractor_mode = "unspecified"

    graph_dict = graph.to_dict(arxiv_id=arxiv_id, extractor_mode=extractor_mode)

    # NOTE: As of now, LaTeX macros are not persisted in SQLite, so the
    # HF export payload does not include a per-paper macro map. The key is
    # present for forward-compatibility with the live backend / CLI payload
    # shape, and remains an empty object for historical exports.
    payload: Dict[str, Any] = {
        "graph": graph_dict,
        "definition_bank": definition_bank,
        "artifact_to_terms_map": artifact_to_terms_map,
        "latex_macros": {},
    }

    # 4) Write to output directory using the Hugging Face-compatible naming convention
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    safe_id = _arxiv_id_to_safe_id(arxiv_id)
    out_path = output_dir_path / f"arxiv_{safe_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    logger.success(f"Exported {arxiv_id} → {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export successfully processed papers from the arxitex SQLite DB "
            "into per-paper JSON files for the ArxiGraph webapp / Hugging Face dataset."
        )
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the shared arxitex SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where arxiv_{id}.json files will be written.",
    )
    parser.add_argument(
        "--only-arxiv-id",
        action="append",
        dest="only_ids",
        help=(
            "If provided, restrict export to these arXiv IDs. "
            "Can be passed multiple times. If omitted, all successfully "
            "processed papers are exported."
        ),
    )
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help=(
            "After exporting JSON files, upload the output directory to a "
            "Hugging Face dataset. By default, uploads are incremental (only files "
            "missing from the remote repo are exported+uploaded)."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help=(
            "Target Hugging Face repo id (e.g. 'turnstilelabs/mathxiv'). "
            "Required when --hf-upload is set."
        ),
    )
    parser.add_argument(
        "--hf-repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type for upload_folder (default: dataset).",
    )
    parser.add_argument(
        "--hf-commit-message",
        default="Update exported ArxiTex graphs",
        help="Commit message to use when uploading to Hugging Face.",
    )

    parser.add_argument(
        "--force-full-export",
        action="store_true",
        help=(
            "Disable the default incremental behavior for --hf-upload and export/upload "
            "all selected papers, even if they already exist on Hugging Face."
        ),
    )

    parser.add_argument(
        "--hf-upload-all-in-output-dir",
        action="store_true",
        help=(
            "Upload ALL arxiv_*.json files currently present in --output-dir (batched commits). "
            "This is the closest behavior to the old upload_folder-based approach. "
            "WARNING: this can be very slow / time out for large folders; prefer incremental upload."
        ),
    )

    parser.add_argument(
        "--hf-batch-size",
        type=int,
        default=200,
        help=(
            "When uploading to Hugging Face, commit files in batches of this size "
            "(default: 200). Smaller batches reduce timeouts for large updates."
        ),
    )

    parser.add_argument(
        "--hf-use-upload-large-folder",
        action="store_true",
        help=(
            "Use HfApi().upload_large_folder(...) for upload. This is more resilient for very "
            "large folders but has limitations (no custom commit message, no custom path_in_repo). "
            "When enabled, the parent directory of --output-dir is uploaded so that the `data/` "
            "folder lands at the repo root."
        ),
    )

    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional HF revision/branch to upload to (default: main).",
    )

    args = parser.parse_args(argv)

    db_path = args.db_path
    output_dir = args.output_dir

    processed_index = ProcessedIndex(db_path)

    # If uploading to HF, we may need to know what's already present.
    # We only hit HF once and reuse the result for:
    # - incremental export (skip already existing papers)
    # - optional upload-all mode filtering
    hf_existing_ids: Set[str] = set()
    if args.hf_upload:
        if not args.hf_repo_id:
            logger.error("--hf-upload was set but --hf-repo-id is missing.")
            return
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            hf_existing_ids = _list_remote_exported_arxiv_ids(
                repo_id=args.hf_repo_id,
                repo_type=args.hf_repo_type,
                revision=args.hf_revision,
                token=token,
            )
            logger.info(
                f"HF remote index: found {len(hf_existing_ids)} existing papers in {args.hf_repo_id}."
            )
        except Exception as e:
            logger.warning(
                f"Failed to list repo files from Hugging Face ({args.hf_repo_id}). "
                f"Incremental behavior may be degraded. Error: {e}"
            )
            hf_existing_ids = set()

    if args.only_ids:
        arxiv_ids: Iterable[str] = args.only_ids
    else:
        arxiv_ids = _iter_successful_arxiv_ids(processed_index)

    exported_paths: List[Path] = []
    skipped_existing = 0
    for arxiv_id in arxiv_ids:
        # Skip any that are not actually marked successful (in case the user
        # provided an explicit list including failures).
        if not processed_index.is_successfully_processed(arxiv_id):
            logger.warning(
                f"Skipping {arxiv_id}: not marked as successfully processed."
            )
            continue

        if (
            args.hf_upload
            and not args.force_full_export
            and hf_existing_ids
            and arxiv_id in hf_existing_ids
        ):
            skipped_existing += 1
            logger.info(
                f"Skipping {arxiv_id}: already exists on Hugging Face (use --force-full-export to override)."
            )
            continue

        try:
            out_path = export_paper(
                db_path=db_path, arxiv_id=arxiv_id, output_dir=output_dir
            )
            if out_path is not None:
                exported_paths.append(out_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to export {arxiv_id}: {e}")

    logger.info(
        f"Finished export. Total papers exported: {len(exported_paths)}"
        + (f" (skipped existing on HF: {skipped_existing})" if skipped_existing else "")
    )

    # Optional: upload the exported folder to a Hugging Face dataset.
    if args.hf_upload:
        try:
            # Imported lazily so that huggingface_hub is only required when
            # users actually want to upload.
            from huggingface_hub import CommitOperationAdd  # type: ignore[import]
            from huggingface_hub import HfApi, login  # type: ignore[import]
        except ImportError:
            logger.error(
                "huggingface_hub is not installed. Install it with 'pip install huggingface-hub' "
                "or disable --hf-upload."
            )
            return

        if not args.hf_repo_id:
            logger.error("--hf-upload was set but --hf-repo-id is missing.")
            return

        # Prefer an explicit token from the environment; this avoids
        # hard-coding credentials in the repo.
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token)
        else:
            # Fall back to existing huggingface-cli login / cached credentials.
            logger.info(
                "HUGGINGFACE_HUB_TOKEN not set; relying on existing Hugging Face login. "
                "You can run 'huggingface-cli login' once or export HUGGINGFACE_HUB_TOKEN to configure auth."
            )

        api = HfApi(token=token) if token else HfApi()

        # Decide what to upload.
        # Default: only upload the files written during this run. This is what makes uploads
        # incremental and avoids timeouts.
        output_dir_path = Path(output_dir)
        paths_to_upload = exported_paths
        upload_filtered_skipped = 0

        if args.hf_upload_all_in_output_dir:
            # Still avoid creating thousands of no-op commits by filtering against the remote index.
            all_local = sorted(output_dir_path.glob("arxiv_*.json"))
            if hf_existing_ids:
                filtered: List[Path] = []
                skipped = 0
                for p in all_local:
                    # p.name == arxiv_<safe_id>.json
                    safe_id = (
                        p.stem[len("arxiv_") :] if p.stem.startswith("arxiv_") else ""
                    )
                    arxiv_id = _safe_id_to_arxiv_id(safe_id) if safe_id else None
                    if arxiv_id and arxiv_id in hf_existing_ids:
                        skipped += 1
                        continue
                    filtered.append(p)
                logger.info(
                    f"--hf-upload-all-in-output-dir: {len(filtered)} missing on HF, {skipped} already present; uploading missing only."
                )
                paths_to_upload = filtered
                upload_filtered_skipped = skipped
            else:
                # If we couldn't list the remote repo, fall back to uploading everything.
                paths_to_upload = all_local

        if not paths_to_upload:
            logger.info("No new files to upload to Hugging Face. Done.")
            return

        if args.hf_use_upload_large_folder:
            # NOTE: upload_large_folder cannot set `path_in_repo`, so we upload the parent folder
            # containing the local `data/` directory.
            folder_to_upload = output_dir_path.parent
            logger.info(
                f"Uploading large folder '{folder_to_upload}' to Hugging Face repo '{args.hf_repo_id}' "
                f"(type={args.hf_repo_type}) using upload_large_folder..."
            )
            api.upload_large_folder(
                repo_id=args.hf_repo_id,
                repo_type=args.hf_repo_type,
                folder_path=folder_to_upload,
                revision=args.hf_revision,
            )
            logger.success("Upload to Hugging Face completed (upload_large_folder).")
            return

        # Default upload method: create commits in batches to avoid timeouts with upload_folder.
        # This also makes incremental updates very fast.
        batch_size = max(1, int(args.hf_batch_size))
        total = len(paths_to_upload)
        logger.info(
            f"Uploading {total} file(s) to Hugging Face repo '{args.hf_repo_id}' (type={args.hf_repo_type}) "
            f"in batches of {batch_size}..."
        )

        def _chunks(seq: Sequence[Path], n: int) -> Iterable[List[Path]]:
            for i in range(0, len(seq), n):
                yield list(seq[i : i + n])

        def _create_commit_with_retries(*, operations: List[Any], message: str) -> None:
            # Retrying helps mitigate transient HTTP timeouts.
            for attempt in range(1, 4):
                try:
                    api.create_commit(
                        repo_id=args.hf_repo_id,
                        repo_type=args.hf_repo_type,
                        revision=args.hf_revision,
                        commit_message=message,
                        operations=operations,
                    )
                    return
                except Exception as e:
                    if attempt >= 3:
                        raise
                    wait_s = 5 * attempt
                    logger.warning(
                        f"HF commit failed (attempt {attempt}/3): {e}. Retrying in {wait_s}s..."
                    )
                    time.sleep(wait_s)

        batch_idx = 0
        empty_commits_skipped = 0
        for batch in _chunks(paths_to_upload, batch_size):
            batch_idx += 1
            ops = [
                CommitOperationAdd(
                    path_in_repo=f"data/{p.name}",
                    path_or_fileobj=str(p),
                )
                for p in batch
            ]
            msg = args.hf_commit_message
            if total > batch_size:
                msg = f"{msg} (batch {batch_idx})"

            logger.info(
                f"Creating HF commit for batch {batch_idx}: {len(batch)} file(s)..."
            )
            try:
                _create_commit_with_retries(operations=ops, message=msg)
            except Exception as e:
                # huggingface_hub raises an exception for no-op commits.
                # Keep it non-fatal but count it so the user can see what's happening.
                msg_lower = str(e).lower()
                if (
                    "no files have been modified" in msg_lower
                    or "empty commit" in msg_lower
                ):
                    empty_commits_skipped += 1
                    logger.info("Batch produced no changes on HF; skipping.")
                    continue
                raise

        logger.success(
            "Upload to Hugging Face completed."
            + (
                f" (upload-all skipped already-present: {upload_filtered_skipped})"
                if upload_filtered_skipped
                else ""
            )
            + (
                f" (no-op batches skipped: {empty_commits_skipped})"
                if empty_commits_skipped
                else ""
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
