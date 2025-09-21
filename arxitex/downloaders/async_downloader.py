import asyncio
import re
from pathlib import Path
from typing import Optional

import aiofiles
import httpx
from loguru import logger

from arxitex.downloaders.utils import (
    detect_file_type,
    try_extract_gzip,
    try_extract_tar,
    try_extract_zip,
    try_handle_plain_text,
)

DEFAULT_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "base_wait_time": 2,
    "chunk_size": 8192,
}


class ArxivExtractorError(Exception):
    pass


class AsyncSourceDownloader:
    def __init__(self, cache_dir=None, config: Optional[dict] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "arxiv_cache"
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.http_client = None

    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(
            timeout=self.config["timeout"],
            follow_redirects=True,
            headers={
                "User-Agent": "ArxivConjectureScraper/1.0 (For academic research)"
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()

    def validate_arxiv_id(self, arxiv_id: str) -> str:
        arxiv_id = arxiv_id.strip().lower()
        patterns = [
            r"^\d{4}\.\d{4,5}(v\d+)?$",
            r"^[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$",
        ]
        if not any(re.match(pattern, arxiv_id) for pattern in patterns):
            raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")
        return arxiv_id

    async def download_and_extract_source(self, arxiv_id: str) -> Path:
        """
        Orchestrates the download and extraction of ALL source files for a given arXiv ID.

        Args:
            arxiv_id: The arXiv identifier to download.

        Returns:
            The path to the directory containing the extracted source files.

        Raises:
            ArxivExtractorError: If any step in the process fails.
            RuntimeError: If not used within an async context manager.
        """
        if not self.http_client:
            raise RuntimeError(
                "AsyncSourceDownloader must be used as an async context manager."
            )

        validated_id = self.validate_arxiv_id(arxiv_id)
        download_dir = self.cache_dir / "downloads"
        extract_dir = self.cache_dir / "source" / validated_id.replace("/", "_")

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"Using cached source directory for {arxiv_id}: {extract_dir}")
            return extract_dir

        try:
            # Step 1: Download the source archive
            source_archive_path = await self._async_download_source(
                validated_id, download_dir
            )
            if not source_archive_path:
                raise ArxivExtractorError(
                    f"Failed to download source for {arxiv_id} after multiple retries."
                )

            # Step 2: Extract the archive
            await self._async_extract_source(
                source_archive_path, extract_dir, validated_id
            )
            logger.info(f"[{arxiv_id}] Source successfully extracted to: {extract_dir}")
            return extract_dir

        except Exception as e:
            logger.error(f"Error in download/extract for {arxiv_id}: {e}")
            raise ArxivExtractorError(
                f"Error in download/extract for {arxiv_id}: {e}"
            ) from e

    async def _async_download_source(
        self, arxiv_id: str, download_dir: Path
    ) -> Optional[Path]:
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"

        if download_path.exists() and download_path.stat().st_size > 0:
            logger.info(f"Using cached download for {arxiv_id}")
            return download_path

        for attempt in range(self.config["max_retries"]):
            try:
                logger.info(
                    f"Downloading source for {arxiv_id} (Attempt {attempt + 1}/{self.config['max_retries']})..."
                )
                async with self.http_client.stream("GET", url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(download_path, "wb") as f:
                        async for chunk in response.aiter_bytes(
                            chunk_size=self.config["chunk_size"]
                        ):
                            await f.write(chunk)
                logger.info(f"Successfully downloaded source to {download_path}")
                return download_path
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"Download attempt {attempt + 1} failed for {arxiv_id}: {e}"
                )
                if attempt + 1 < self.config["max_retries"]:
                    wait_time = self.config["base_wait_time"] * (2**attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        return None

    async def _async_extract_source(
        self, source_path: Path, extract_dir: Path, arxiv_id: str
    ):
        await asyncio.to_thread(
            self._blocking_extract, source_path, extract_dir, arxiv_id
        )

    def _blocking_extract(self, file_path: Path, dest_path: Path, arxiv_id: str):
        dest_path.mkdir(parents=True, exist_ok=True)
        file_type = detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")

        extractors = [
            lambda: try_extract_tar(file_path, dest_path),
            lambda: try_extract_gzip(file_path, dest_path, arxiv_id),
            lambda: try_extract_zip(file_path, dest_path),
            lambda: try_handle_plain_text(file_path, dest_path, arxiv_id),
        ]

        if file_type == "pdf":
            raise ArxivExtractorError(
                f"Paper {arxiv_id} is PDF-only. LaTeX source is required."
            )

        success = False
        if file_type != "unknown":
            success = extractors[["tar", "gzip", "zip", "tex"].index(file_type)]()

        if not success:
            logger.warning("Trying all extraction methods for unknown file type...")
            for extractor in extractors:
                if extractor():
                    success = True
                    break

        if not success:
            raise ArxivExtractorError(
                "Unable to extract or identify downloaded file format."
            )
