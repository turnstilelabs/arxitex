import argparse
import sqlite3
from pathlib import Path


def load_ids(conn, sql: str):
    cur = conn.execute(sql)
    return {row[0] for row in cur.fetchall()}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "One-off consistency check between discovered_papers, "
            "processed_papers, skipped_papers, and normalized papers table."
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help=(
            "Path to arxitex_indices.db. "
            "Defaults to <repo_root>/pipeline_output/arxitex_indices.db."
        ),
    )
    args = parser.parse_args()

    if args.db is not None:
        db_path = Path(args.db)
    else:
        # Mirror ArxivPipelineComponents default: project_root / "pipeline_output" / "arxitex_indices.db"
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
        db_path = project_root / "pipeline_output" / "arxitex_indices.db"

    if not db_path.exists():
        print(f"[ERROR] DB path does not exist: {db_path}")
        return 1

    print(f"Using DB: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Basic sets from the lightweight indices
        pending = load_ids(conn, "SELECT arxiv_id FROM discovered_papers")

        cur = conn.execute("SELECT arxiv_id, status FROM processed_papers")
        processed = {row[0]: row[1] for row in cur.fetchall()}
        success_ids = {
            pid for pid, s in processed.items() if s and s.startswith("success")
        }
        failure_ids = {
            pid for pid, s in processed.items() if s and s.startswith("failure")
        }

        skipped = load_ids(conn, "SELECT arxiv_id FROM skipped_papers")

        # Normalized persisted papers
        papers = load_ids(conn, "SELECT paper_id FROM papers")

        indexed = pending | set(processed.keys()) | skipped
        orphan_papers = papers - indexed

        failures_not_pending = failure_ids - pending
        success_still_pending = success_ids & pending

        print("\n=== High-level counts ===")
        print(f"Pending in discovery queue : {len(pending)}")
        print(f"Processed (any status)     : {len(processed)}")
        print(f"  - success                : {len(success_ids)}")
        print(f"  - failure                : {len(failure_ids)}")
        print(f"Skipped                     : {len(skipped)}")
        print(f"Papers in normalized DB    : {len(papers)}")

        print("\n=== Potential anomalies ===")
        print(
            f"Orphan papers (in papers table but not in discovery/processed/skipped): {len(orphan_papers)}"
        )
        if orphan_papers:
            for pid in sorted(list(orphan_papers))[:50]:
                print("  ", pid)

        print(
            f"\nFailures not pending (failure status but not in discovery queue): {len(failures_not_pending)}"
        )
        if failures_not_pending:
            for pid in sorted(list(failures_not_pending))[:50]:
                print("  ", pid)

        print(
            f"\nSuccesses still pending (success status but still in discovery queue): {len(success_still_pending)}"
        )
        if success_still_pending:
            for pid in sorted(list(success_still_pending))[:50]:
                print("  ", pid)

        print("\n[Done] Consistency check complete.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
