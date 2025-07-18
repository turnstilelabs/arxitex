import os
import asyncio
import httpx
import aiofiles
import logging
from pathlib import Path
from typing import Optional, List
from arxitex.downloaders.utils import (
    detect_file_type,
    is_gzipped,
    try_extract_tar,
    try_extract_zip,
    try_extract_gzip,
    try_handle_plain_text,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'base_wait_time': 2,
    'chunk_size': 8192,
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
            timeout=self.config['timeout'],
            follow_redirects=True,
            headers={'User-Agent': 'ArxivConjectureScraper/1.0 (For academic research)'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()

    def validate_arxiv_id(self, arxiv_id: str) -> str:
        import re
        arxiv_id = arxiv_id.strip().lower()
        patterns = [
            r'^\d{4}\.\d{4,5}(v\d+)?$',
            r'^[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$',
        ]
        if not any(re.match(pattern, arxiv_id) for pattern in patterns):
            raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")
        return arxiv_id

    async def async_download_and_read_latex(self, arxiv_id: str) -> Optional[str]:
        """
        Main async method to download, extract, and read LaTeX files.
        
        Args:
            arxiv_id: The arXiv identifier (e.g., '2305.12345').
            
        Returns:
            The combined content of all .tex files as a string, or None on failure.
        """
        if not self.http_client:
            raise RuntimeError("AsyncSourceDownloader must be used as an async context manager")
            
        download_dir = self.cache_dir / "downloads"
        extract_dir = self.cache_dir / "source" / arxiv_id.replace('/', '_')
        
        try:
            validated_id = self.validate_arxiv_id(arxiv_id)
            
            source_archive_path = await self._async_download_source(validated_id, download_dir)
            if not source_archive_path:
                return None

            await self._async_extract_source(source_archive_path, extract_dir, validated_id)

            # Find ALL .tex files
            tex_files = await self.find_tex_files(extract_dir)
            if not tex_files:
                logger.error(f"No .tex files found in the extracted source for {arxiv_id}.")
                return None
            
            logger.info(f"Found {len(tex_files)} .tex files: {[f.name for f in tex_files]}")

            # Read and combine content from ALL files
            combined_content = await self.read_latex_content(tex_files)
            return combined_content

        except Exception as e:
            logger.error(f"An error occurred during the download/extraction process for {arxiv_id}: {e}")
            return None

    async def _async_download_source(self, arxiv_id: str, download_dir: Path) -> Optional[Path]:
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"

        # Check if already downloaded
        if download_path.exists() and download_path.stat().st_size > 0:
            logger.info(f"Using cached download for {arxiv_id}")
            return download_path

        for attempt in range(self.config['max_retries']):
            try:
                logger.info(f"Downloading source for {arxiv_id} (Attempt {attempt + 1}/{self.config['max_retries']})...")
                async with self.http_client.stream("GET", url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(download_path, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=self.config['chunk_size']):
                            await f.write(chunk)
                logger.info(f"Successfully downloaded source to {download_path}")
                return download_path
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {arxiv_id}: {e}")
                if attempt + 1 < self.config['max_retries']:
                    wait_time = self.config['base_wait_time'] * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        return None

    async def _async_extract_source(self, source_path: Path, extract_dir: Path, arxiv_id: str):
        await asyncio.to_thread(self._blocking_extract, source_path, extract_dir, arxiv_id)

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

        if file_type == 'pdf':
            raise ArxivExtractorError(f"Paper {arxiv_id} is PDF-only. LaTeX source is required.")

        success = False
        if file_type != 'unknown':
            success = extractors[['tar', 'gzip', 'zip', 'tex'].index(file_type)]()

        if not success:
            logger.warning(f"Trying all extraction methods for unknown file type...")
            for extractor in extractors:
                if extractor():
                    success = True
                    break

        if not success:
            raise ArxivExtractorError("Unable to extract or identify downloaded file format.")
    
    async def find_tex_files(self, source_path: Path) -> List[Path]:
        """Find all LaTeX files in the directory."""
        def _blocking_find_tex_files(path: Path) -> List[Path]:
            tex_files = list(path.rglob('*.tex'))
            if not tex_files:
                all_files = list(path.glob('*'))
                logger.error(f"No .tex files found. Available files: {[f.name for f in all_files]}")
                return []
            
            logger.info(f"Found {len(tex_files)} .tex files: {[f.name for f in tex_files]}")
            return tex_files
        
        return await asyncio.to_thread(_blocking_find_tex_files, source_path)
    
    async def read_latex_content(self, tex_files: List[Path]) -> str:
        """Read and combine content from all found LaTeX files."""
        full_content = []
        for tex_file in sorted(tex_files):  # Sort for deterministic order
            try:
                async with aiofiles.open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                # Add a comment to mark the beginning of each file
                full_content.append(f"\n% --- Source File: {tex_file.name} ---\n{content}")
                logger.debug(f"Read {len(content)} characters from {tex_file.name}")
            except Exception as e:
                logger.warning(f"Could not read {tex_file.name}: {e}")
        
        combined_content = ''.join(full_content)
        logger.info(f"Combined total LaTeX content: {len(combined_content)} characters")
        return combined_content
