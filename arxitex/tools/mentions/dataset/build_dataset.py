#!/usr/bin/env python3
"""Build clean-core mention supervision data.

Outputs:
- combined_statements.json
- statements.jsonl
- mention_contexts.jsonl
- mention_gold_links.json
- splits/*_mention_contexts.jsonl
- splits/*_mention_gold_links.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arxitex.tools.mentions.dataset.acquisition_pipeline import (
    Target,
    collect_mentions_rows,
    derive_target_id,
    load_targets,
    prepare_statement_paths,
)
from arxitex.tools.mentions.dataset.mapping_pipeline import (
    build_mentions_dataset,
    write_target_splits,
)
from arxitex.tools.mentions.dataset.statements import (
    build_statement_registry_and_corpus,
)
from arxitex.utils import ensure_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean-core mention supervision."
    )
    parser.add_argument("--targets-json", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--out-dir", default="data/mentions")
    parser.add_argument("--statements-dir", default="data/statements/mentions")
    parser.add_argument("--debug-mapping", action="store_true")
    parser.add_argument("--fail-on-statement-duplicates", action="store_true")
    parser.add_argument("--val-target", default="1303.5113")
    parser.add_argument("--test-target", default="1709.10033")
    return parser.parse_args()


def _assert_required_inputs(
    *, targets: list[Target], out_dir: Path, statements_dir: Path
) -> None:
    missing_statements: list[str] = []
    for target in targets:
        if target.local_statements:
            path = Path(target.local_statements)
        else:
            path = statements_dir / f"{target.arxiv_id.replace('/', '_')}.json"
        if not path.exists():
            missing_statements.append(str(path))

    missing_mentions: list[str] = []
    for target in targets:
        mentions_path = out_dir / f"{derive_target_id(target.arxiv_id)}_mentions.jsonl"
        if not mentions_path.exists():
            missing_mentions.append(str(mentions_path))

    if missing_statements or missing_mentions:
        problems: list[str] = []
        if missing_statements:
            problems.append(
                "missing statements:\n- " + "\n- ".join(missing_statements[:10])
            )
        if missing_mentions:
            problems.append(
                "missing mentions:\n- " + "\n- ".join(missing_mentions[:10])
            )
        details = "\n".join(problems)
        raise RuntimeError(
            "Missing required local inputs for deterministic build.\n"
            f"{details}\n"
            "Run acquisition first with the same target arguments, e.g.:\n"
            "python -m arxitex.tools.mentions.dataset.acquire_inputs "
            f"--out-dir {out_dir} --statements-dir {statements_dir} --cache-dir data/cache"
        )


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    statements_dir = Path(args.statements_dir)
    ensure_dir(str(out_dir))
    targets = load_targets(args.targets_json, args.targets)
    _assert_required_inputs(
        targets=targets, out_dir=out_dir, statements_dir=statements_dir
    )

    statement_paths = prepare_statement_paths(targets, statements_dir)
    source_map = {target.arxiv_id: target.local_source_dir for target in targets}
    target_registry = build_statement_registry_and_corpus(
        statement_paths=statement_paths,
        source_map=source_map,
        out_dir=out_dir,
        fail_on_duplicates=args.fail_on_statement_duplicates,
    )

    mentions_rows = collect_mentions_rows(
        targets=targets,
        out_dir=out_dir,
    )
    mapping_report_path = (
        out_dir / "mapping_report.jsonl" if args.debug_mapping else None
    )
    mapping_summary_path = (
        out_dir / "mapping_summary.json" if args.debug_mapping else None
    )
    legacy_labels_path = out_dir / "mention_labels.json"
    if legacy_labels_path.exists():
        legacy_labels_path.unlink()
    build_mentions_dataset(
        mentions_rows=mentions_rows,
        target_registry=target_registry,
        contexts_out_path=out_dir / "mention_contexts.jsonl",
        gold_links_out_path=out_dir / "mention_gold_links.json",
        mapping_report_path=mapping_report_path,
        mapping_summary_path=mapping_summary_path,
    )

    write_target_splits(
        contexts_path=out_dir / "mention_contexts.jsonl",
        gold_links_path=out_dir / "mention_gold_links.json",
        out_dir=out_dir / "splits",
        val_target=args.val_target,
        test_target=args.test_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
