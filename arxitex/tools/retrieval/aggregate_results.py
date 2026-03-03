#!/usr/bin/env python3
"""Aggregate retrieval benchmark summaries into a single report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_rows(root_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for summary_path in root_dir.rglob("summary.json"):
        summary = _load_json(summary_path)
        meta_path = summary_path.with_name("run_metadata.json")
        metadata = _load_json(meta_path) if meta_path.exists() else {}

        metrics = summary.get("metrics", {})
        runtimes = summary.get("runtimes_sec", {})
        for exp, vals in metrics.items():
            rows.append(
                {
                    "run_id": summary.get("run_id"),
                    "path": str(summary_path.parent),
                    "experiment": exp,
                    "nDCG@10": vals.get("nDCG@10"),
                    "Recall@10": vals.get("Recall@10"),
                    "Hit@10": vals.get("Hit@10"),
                    "pylate_latency_sec": vals.get("pylate_latency_sec"),
                    "runtime_sec": runtimes.get(exp),
                    "artifact_count": summary.get("artifact_count"),
                    "index_mode": metadata.get("index_mode"),
                    "normalize_mode": metadata.get("normalize_mode"),
                    "include_type_prefix": metadata.get("include_type_prefix"),
                    "include_proofs": metadata.get("include_proofs"),
                    "query_count_filtered": (metadata.get("query_counts") or {}).get(
                        "filtered"
                    ),
                }
            )
    return rows


def _write_markdown(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text(
            "# Retrieval Aggregate Report\n\nNo summary.json files found.\n",
            encoding="utf-8",
        )
        return
    cols = [
        "run_id",
        "experiment",
        "nDCG@10",
        "Recall@10",
        "Hit@10",
        "pylate_latency_sec",
        "runtime_sec",
        "index_mode",
        "normalize_mode",
        "include_type_prefix",
        "include_proofs",
        "query_count_filtered",
        "artifact_count",
        "path",
    ]
    lines = ["# Retrieval Aggregate Report", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
        lines.append(line)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for row in rows:
        lines.append(
            ",".join("" if row.get(c) is None else str(row.get(c)) for c in cols)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(rows: List[Dict], out_path: Path) -> None:
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate retrieval benchmark summaries."
    )
    parser.add_argument(
        "--root-dir",
        default="data/retrieval",
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output file path (default: <root-dir>/aggregate_report.md).",
    )
    parser.add_argument(
        "--format", default="md", choices=["md", "csv", "json"], help="Output format."
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_path = Path(args.out) if args.out else root_dir / "aggregate_report.md"
    rows = _collect_rows(root_dir)

    if args.format == "md":
        _write_markdown(rows, out_path)
    elif args.format == "csv":
        _write_csv(rows, out_path)
    else:
        _write_json(rows, out_path)

    print(f"Wrote aggregate report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
