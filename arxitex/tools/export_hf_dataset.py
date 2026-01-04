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
from pathlib import Path
from typing import Any, Dict, Iterable

from loguru import logger

from arxitex.db.connection import connect
from arxitex.db.persistence import load_document_graph
from arxitex.db.schema import ensure_schema
from arxitex.indices.processed import ProcessedIndex


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

    safe_id = arxiv_id.replace("/", "_")
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
            "Hugging Face dataset using huggingface_hub.upload_folder."
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

    args = parser.parse_args(argv)

    db_path = args.db_path
    output_dir = args.output_dir

    processed_index = ProcessedIndex(db_path)

    if args.only_ids:
        arxiv_ids: Iterable[str] = args.only_ids
    else:
        arxiv_ids = _iter_successful_arxiv_ids(processed_index)

    exported_count = 0
    for arxiv_id in arxiv_ids:
        # Skip any that are not actually marked successful (in case the user
        # provided an explicit list including failures).
        if not processed_index.is_successfully_processed(arxiv_id):
            logger.warning(
                f"Skipping {arxiv_id}: not marked as successfully processed."
            )
            continue

        try:
            out_path = export_paper(
                db_path=db_path, arxiv_id=arxiv_id, output_dir=output_dir
            )
            if out_path is not None:
                exported_count += 1
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to export {arxiv_id}: {e}")

    logger.info(f"Finished export. Total papers exported: {exported_count}")

    # Optional: upload the exported folder to a Hugging Face dataset.
    if args.hf_upload:
        if not args.hf_repo_id:
            logger.error("--hf-upload was set but --hf-repo-id is missing.")
            return

        try:
            # Imported lazily so that huggingface_hub is only required when
            # users actually want to upload.
            from huggingface_hub import login, upload_folder  # type: ignore[import]
        except ImportError:
            logger.error(
                "huggingface_hub is not installed. Install it with 'pip install huggingface-hub' "
                "or disable --hf-upload."
            )
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

        logger.info(
            f"Uploading folder '{output_dir}' to Hugging Face repo "
            f"'{args.hf_repo_id}' (type={args.hf_repo_type}) under path 'data/'..."
        )

        # We always upload into a `data/` subdirectory in the remote repo so
        # that the resulting URLs match the Next.js expectations:
        #   .../resolve/<ref>/data/arxiv_XXXX.json
        upload_folder(
            folder_path=str(output_dir),
            path_in_repo="data",
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            commit_message=args.hf_commit_message,
        )
        logger.success("Upload to Hugging Face completed.")


if __name__ == "__main__":  # pragma: no cover
    main()
