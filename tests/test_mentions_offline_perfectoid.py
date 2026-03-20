import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from arxitex.tools.mentions.extraction import extract_mentions_cli

# Offline mention fixture parity relies on HTML extraction via BeautifulSoup+lxml.
pytest.importorskip("bs4")
pytest.importorskip("lxml")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _canon_rows(path: Path) -> Counter:
    rows = []
    for row in _read_jsonl(path):
        row = dict(row)
        # Keep fixture check robust to metadata-only tuning in target matching.
        row.pop("target_match_confidence", None)
        row.pop("target_match_reason", None)
        rows.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    return Counter(rows)


def test_offline_mentions_match_perfectoid(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    works_path = Path("data/citation_dataset/perfectoid_works.jsonl")
    cache_dir = Path("data/citation_dataset/cache")
    if not works_path.exists() or not cache_dir.exists():
        pytest.skip("Perfectoid offline fixtures not available in this environment.")

    args = [
        "--target-arxiv",
        "https://arxiv.org/abs/1205.2208",
        "--target-title",
        "Perfectoid Spaces",
        "--target-id",
        "perfectoid",
        "--works-file",
        str(works_path),
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(cache_dir),
        "--offline",
    ]

    asyncio.run(extract_mentions_cli.main(args))

    generated = out_dir / "perfectoid_mentions.jsonl"
    expected = Path("data/citation_dataset/perfectoid_mentions.jsonl")

    assert generated.exists()
    assert expected.exists()
    got = _canon_rows(generated)
    want = _canon_rows(expected)
    overlap = sum((got & want).values())
    overlap_ratio = overlap / max(1, sum(want.values()))
    count_delta = abs(sum(got.values()) - sum(want.values()))
    # Allow small drift from extraction heuristics while catching major regressions.
    assert overlap_ratio >= 0.95
    assert count_delta <= 10
