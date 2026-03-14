"""Shared helpers for mentions source download scripts."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger

from arxitex.arxiv_utils import normalize_arxiv_id, try_parse_arxiv_id
from arxitex.downloaders.async_downloader import (
    ArxivExtractorError,
    AsyncSourceDownloader,
)


def canonicalize_arxiv_id(raw: str) -> str | None:
    parsed = try_parse_arxiv_id(raw)
    if parsed:
        return parsed
    normalized = normalize_arxiv_id(raw or "")
    return normalized or None


def iter_arxiv_ids_from_works(paths: Iterable[Path]) -> Iterable[str]:
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = row.get("arxiv_id") or ""
            if not arxiv_id:
                continue
            parsed = canonicalize_arxiv_id(str(arxiv_id))
            if parsed:
                yield parsed


def load_targets(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_targets(path: Path, targets: List[Dict]) -> None:
    path.write_text(json.dumps(targets, ensure_ascii=False, indent=2))


def download_source_to_dir(
    arxiv_id: str,
    dest_dir: Path,
    *,
    cache_dir: Path,
    sleep_sec: float,
) -> bool:
    dest_dir = Path(dest_dir)
    cache_dir = Path(cache_dir)

    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info("Source already exists for {}", arxiv_id)
        return True
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    async def _download() -> Path:
        async with AsyncSourceDownloader(cache_dir=cache_dir) as downloader:
            return await downloader.download_and_extract_source(arxiv_id)

    try:
        extracted = asyncio.run(_download())
    except ArxivExtractorError as exc:
        logger.warning("Download failed for {}: {}", arxiv_id, exc)
        return False
    except Exception as exc:
        logger.warning("Download failed for {}: {}", arxiv_id, exc)
        return False

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if extracted != dest_dir:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(extracted), str(dest_dir))

    time.sleep(sleep_sec)
    return True
