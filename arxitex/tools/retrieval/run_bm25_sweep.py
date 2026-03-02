#!/usr/bin/env python3
"""Run a constrained BM25 parameter sweep across index variants."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_GRID = [
    (0.6, 0.2),
    (0.9, 0.5),
    (1.2, 0.75),
    (1.5, 0.9),
    (1.8, 0.75),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BM25 sweep across index variants."
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--normalize-mode", default="auto")
    parser.add_argument("--single-ref-only", action="store_true")
    parser.add_argument("--no-proofs", action="store_true")
    parser.add_argument(
        "--index-modes",
        default="content,content+prereqs,content+semantic,content+all",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_modes = [m.strip() for m in args.index_modes.split(",") if m.strip()]
    for mode in index_modes:
        for k1, b in DEFAULT_GRID:
            target_dir = out_dir / mode / f"k1_{k1}_b_{b}"
            log_dir = target_dir / "logs"
            cmd = [
                "python",
                "-m",
                "arxitex.tools.retrieval.retrieval_benchmark",
                "--graph",
                args.graph,
                "--queries",
                args.queries,
                "--out-dir",
                str(target_dir),
                "--experiment",
                "e1",
                "--top-k",
                "10",
                "--index-mode",
                mode,
                "--normalize-mode",
                args.normalize_mode,
                "--bm25-k1",
                str(k1),
                "--bm25-b",
                str(b),
                "--run-id",
                args.run_id,
                "--log-dir",
                str(log_dir),
            ]
            if args.single_ref_only:
                cmd.append("--single-ref-only")
            if args.no_proofs:
                cmd.append("--no-proofs")

            subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
