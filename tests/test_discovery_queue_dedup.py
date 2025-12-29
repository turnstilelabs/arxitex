import sqlite3

from arxitex.tools.discovery_queue_dedup import dedup_discovery_queue


def _seed_discovered_papers(db_path, arxiv_ids):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS discovered_papers (arxiv_id TEXT PRIMARY KEY, metadata TEXT NOT NULL)"
        )
        for aid in arxiv_ids:
            conn.execute(
                "INSERT OR REPLACE INTO discovered_papers (arxiv_id, metadata) VALUES (?, ?) ",
                (aid, "{}"),
            )
        conn.commit()
    finally:
        conn.close()


def _fetch_ids(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT arxiv_id FROM discovered_papers ORDER BY arxiv_id")
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def test_dedup_discovery_queue_keeps_highest_version(tmp_path):
    db_path = tmp_path / "arxitex_indices.db"
    _seed_discovered_papers(
        db_path,
        [
            "2406.01082v1",
            "2406.01082v2",
            "2305.14545v3",
            "2305.14545v4",
            "9999.00001",  # no explicit version
        ],
    )

    # Dry run: no changes.
    report = dedup_discovery_queue(db_path, dry_run=True, make_backup=False)
    assert report.rows_to_delete == 2
    assert report.rows_deleted == 0
    assert set(_fetch_ids(db_path)) == {
        "2305.14545v3",
        "2305.14545v4",
        "2406.01082v1",
        "2406.01082v2",
        "9999.00001",
    }

    # Apply: remove older versions.
    report2 = dedup_discovery_queue(db_path, dry_run=False, make_backup=False)
    assert report2.rows_deleted == 2
    assert report2.base_ids_duplicated_after == 0
    assert set(_fetch_ids(db_path)) == {
        "2305.14545v4",
        "2406.01082v2",
        "9999.00001",
    }
