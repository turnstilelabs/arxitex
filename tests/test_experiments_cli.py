from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List

from arxitex.experiments.cli import (
    bucket_for_date,
    parse_date_from_arxiv_id,
    parse_iso8601,
    stratified_sample,
    stratified_sample_ids,
    strong_match,
)


def test_parse_iso8601_z():
    ts = "2025-01-15T12:34:56Z"
    dt = parse_iso8601(ts)
    assert dt.tzinfo is not None
    assert dt == datetime(2025, 1, 15, 12, 34, 56, tzinfo=timezone.utc)


def test_bucket_for_date_ranges():
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert bucket_for_date(datetime(2010, 1, 1, tzinfo=timezone.utc), cutoff) == "older"
    assert (
        bucket_for_date(datetime(2017, 6, 1, tzinfo=timezone.utc), cutoff) == "recent"
    )
    assert (
        bucket_for_date(datetime(2022, 3, 1, tzinfo=timezone.utc), cutoff)
        == "latest_pre_cutoff"
    )
    assert (
        bucket_for_date(datetime(2024, 2, 1, tzinfo=timezone.utc), cutoff)
        == "post_cutoff"
    )


def test_stratified_sample_assigns_buckets_and_limits_per_bucket():
    # Build minimal paper metadata with published timestamps across buckets
    def iso(dt: datetime) -> str:
        return dt.isoformat().replace("+00:00", "Z")

    papers: List[Dict] = [
        {
            "arxiv_id": "old-1",
            "title": "Old A",
            "published": iso(datetime(2010, 5, 1, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "old-2",
            "title": "Old B",
            "published": iso(datetime(2012, 7, 1, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "recent-1",
            "title": "Recent A",
            "published": iso(datetime(2016, 1, 1, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "recent-2",
            "title": "Recent B",
            "published": iso(datetime(2020, 6, 1, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "latest-1",
            "title": "Latest A",
            "published": iso(datetime(2021, 2, 1, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "latest-2",
            "title": "Latest B",
            "published": iso(datetime(2023, 11, 1, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "post-1",
            "title": "Post A",
            "published": iso(datetime(2024, 1, 2, tzinfo=timezone.utc)),
        },
        {
            "arxiv_id": "post-2",
            "title": "Post B",
            "published": iso(datetime(2025, 6, 1, tzinfo=timezone.utc)),
        },
    ]

    sampled, buckets = stratified_sample(
        papers, per_bucket=1, seed=123, cutoff=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    # We should get at most 1 per bucket; all assigned a bucket
    assert 1 <= len(sampled) <= 4
    for s in sampled:
        assert "bucket" in s
        assert s["bucket"] in {"older", "recent", "latest_pre_cutoff", "post_cutoff"}

    # Count occurrences per bucket in the sampled set — should be <= per_bucket
    counts = Counter([s["bucket"] for s in sampled])
    for c in counts.values():
        assert c <= 1

    # Buckets aggregation should classify all provided papers
    assert sum(len(v) for v in buckets.values()) == len(papers)


def test_strong_match_id_and_title_normalization():
    # Same ID and title differs only by case/punctuation => strong match
    pred_id = "1234.5678"
    true_id = "1234.5678"
    pred_title = "A Title: Proof of X."
    true_title = "a title  proof of x"
    res = strong_match(pred_id, pred_title, true_id, true_title)
    assert res["id_match"] == 1
    assert res["title_match"] == 1
    assert res["strong_match"] == 1

    # Different ID or title -> not strong match
    res2 = strong_match("1234.5678", "Some Other Title", "1234.5678", "Original Title")
    assert res2["id_match"] == 1
    assert res2["title_match"] == 0
    assert res2["strong_match"] == 0


# -------- ID-based sampling tests --------


def test_parse_date_from_arxiv_id_new_style():
    # 2507.05087 -> 2025-07-01
    dt = parse_date_from_arxiv_id("2507.05087v1")
    assert dt == datetime(2025, 7, 1, tzinfo=timezone.utc)

    # 2312.99999 -> 2023-12-01
    dt2 = parse_date_from_arxiv_id("2312.99999")
    assert dt2 == datetime(2023, 12, 1, tzinfo=timezone.utc)


def test_parse_date_from_arxiv_id_old_style():
    # math.AG/0601001v2 -> 2006-01-01
    dt = parse_date_from_arxiv_id("math.AG/0601001v2")
    assert dt == datetime(2006, 1, 1, tzinfo=timezone.utc)


def test_parse_date_from_arxiv_id_invalid():
    assert parse_date_from_arxiv_id("not-an-id") is None
    assert parse_date_from_arxiv_id("x") is None


def test_stratified_sample_ids():
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ids = [
        "2507.05087",  # 2025-07 -> post_cutoff
        "2401.00001v3",  # 2024-01 -> post_cutoff
        "2312.99999",  # 2023-12 -> latest_pre_cutoff
        "math.AG/0601001v2",  # 2006-01 -> older
        "1706.00001",  # 2017-06 -> recent
    ]
    sampled_ids, buckets = stratified_sample_ids(
        ids, per_bucket=1, seed=7, cutoff=cutoff
    )

    # We should sample at most one per bucket
    assert len(sampled_ids) <= 4
    # Buckets map should reflect our inputs
    assert "older" in buckets and "math.AG/0601001v2" in buckets["older"]
    assert "recent" in buckets and "1706.00001" in buckets["recent"]
    assert (
        "latest_pre_cutoff" in buckets and "2312.99999" in buckets["latest_pre_cutoff"]
    )
    assert "post_cutoff" in buckets and {"2507.05087", "2401.00001v3"} & set(
        buckets["post_cutoff"]
    )


# -------- generate_then_verify argparse CLI smoke tests --------


def test_generate_then_verify_cli_help(monkeypatch, capsys):
    """
    Ensure the argparse-based CLI in arxitex.experiments.generate_then_verify is wired and exposes help.
    This should exit with code 0 and print the parser description/usage.
    """
    import sys

    from arxitex.experiments import generate_then_verify

    # Simulate: python -m arxitex.experiments.generate_then_verify --help
    monkeypatch.setattr(
        sys, "argv", ["arxitex.experiments.generate_then_verify", "--help"]
    )
    try:
        generate_then_verify.main()
    except SystemExit as e:
        assert e.code == 0

    out = capsys.readouterr().out
    # Description string comes from the argparse in generate_then_verify.main()
    assert ("Generate-then-verify" in out) or ("arXiv" in out) or ("usage:" in out)


def test_generate_then_verify_cli_requires_openai_api_key(monkeypatch, capsys):
    """
    Running the CLI without OPENAI_API_KEY should fail fast before any network calls.
    """
    import sys

    from arxitex.experiments import generate_then_verify

    # Ensure the env var is not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Provide minimal args; add --verify-sleep 0 to keep things quick if code path ever changes.
    monkeypatch.setattr(
        sys,
        "argv",
        ["arxitex.experiments.generate_then_verify", "-n", "1", "--verify-sleep", "0"],
    )

    try:
        generate_then_verify.main()
    except SystemExit as e:
        # The CLI in generate_then_verify.py raises with exit code 2 when the env var is missing
        assert e.code == 2

    out = capsys.readouterr().out
    assert "OPENAI_API_KEY is not set" in out
