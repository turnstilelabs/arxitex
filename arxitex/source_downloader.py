import requests
import time
import os
import re
from loguru import logger
from functools import lru_cache
from pathlib import Path
import tarfile
import zipfile
import gzip
import tempfile
import shutil
from typing import Optional, Dict, Any, List
import asyncio
import aiofiles
import httpx

DEFAULT_CONFIG = {
    'max_retries': 3,
    'timeout': 60,
    'chunk_size': 8192,
    'base_wait_time': 2
}

class ArxivExtractorError(Exception):
    """Custom exception for ArXiv extraction errors."""
    pass

class SourceDownloader:
    """Handles downloading and extracting source files from ArXiv"""
    
    def __init__(self, cache_dir=None, config: Optional[Dict[str, Any]] = None):
        """Initialize downloader with cache directory and optional configuration."""
        self.cache_dir = cache_dir
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ArxivConjectureScraper/1.0 (For academic research)',
        })
        
    def validate_arxiv_id(self, arxiv_id: str) -> str:
        """Validate and normalize arXiv ID format."""
        arxiv_id = arxiv_id.strip().lower()

        # Regex for new (YYMM.NNNN) and old (subject-class/YYMMnnn) formats
        patterns = [
            r'^\d{4}\.\d{4,5}(v\d+)?$',
            r'^[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$',
        ]

        if not any(re.match(pattern, arxiv_id) for pattern in patterns):
            raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")

        return arxiv_id
    
    @lru_cache(maxsize=128)
    def download_source(self, source_url, output_dir):
        """Download and cache LaTeX source files"""
        arxiv_id = source_url.split('/')[-1]
        source_dir = os.path.join(output_dir, arxiv_id)
        
        os.makedirs(source_dir, exist_ok=True)
        source_file = os.path.join(source_dir, arxiv_id)

        if os.path.exists(source_file) and os.path.getsize(source_file) > 0:
            tex_files = [f for f in os.listdir(source_dir) if f.endswith('.tex')]
            if tex_files:
                logger.info(f"Using cached source for: {arxiv_id}")
                return source_file, True
            else:
                logger.info(f"Found cached source but no extracted .tex files for: {arxiv_id}")

        logger.info(f"Downloading source for: {arxiv_id}")

        try:
            dest_path = Path(source_dir)
            validated_id = self.validate_arxiv_id(arxiv_id)
            
            extraction_success = self._download_and_extract(validated_id, dest_path)
            
            return source_dir, extraction_success
            
        except Exception as e:
            logger.error(f"  Error in download_source: {e}")
            logger.error(f"  Paper URL: https://arxiv.org/abs/{arxiv_id}")
            return None, False

    def _download_and_extract(self, arxiv_id: str, dest_path: Path) -> bool:
        """Internal method that does the actual downloading and extraction."""
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        logger.info(f"Downloading source from {url}")

        temp_file = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=f"_{arxiv_id.replace('/', '_')}.tmp",
                dir=dest_path,
                prefix="download_"
            )
            temp_file = Path(temp_path)
            os.close(temp_fd)
    
            retry_count = 0
            max_retries = self.config['max_retries']
            
            while retry_count < max_retries:
                try:
                    response = self.session.get(url, timeout=self.config['timeout'], stream=True)
                    response.raise_for_status()
                    break
                except (requests.RequestException, requests.Timeout) as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = self.config['base_wait_time'] ** retry_count
                        logger.warning(f"Request failed (attempt {retry_count}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to download source for {arxiv_id} after multiple attempts")
                        raise ArxivExtractorError(f"Failed to download {url}: {e}")

            # Save response to temporary file
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.config['chunk_size']):
                    f.write(chunk)
            logger.info(f"Downloaded {temp_file.stat().st_size} bytes")

            file_type = self._detect_file_type(temp_file)
            logger.info(f"Detected file type: {file_type}")
            
            if file_type == 'tar' and self._try_extract_tar(temp_file, dest_path):
                pass
            elif file_type == 'gzip' and self._try_extract_gzip(temp_file, dest_path, arxiv_id):
                pass
            elif file_type == 'zip' and self._try_extract_zip(temp_file, dest_path):
                pass
            elif file_type == 'tex' and self._try_handle_plain_text(temp_file, dest_path, arxiv_id):
                pass
            elif file_type == 'pdf':
                raise ArxivExtractorError(f"Paper {arxiv_id} is PDF-only. LaTeX source is required.")
            else:
                # Fallback: try all methods if detection fails or returns unknown
                logger.warning(f"Unknown file type '{file_type}', trying all extraction methods...")
                if not (self._try_extract_tar(temp_file, dest_path) or
                       self._try_extract_gzip(temp_file, dest_path, arxiv_id) or
                       self._try_extract_zip(temp_file, dest_path) or
                       self._try_handle_plain_text(temp_file, dest_path, arxiv_id)):
                    raise ArxivExtractorError("Unable to extract or identify downloaded file format.")

            logger.info(f"Successfully processed source to {dest_path}")
            return True

        except requests.RequestException as e:
            raise ArxivExtractorError(f"Failed to download {url}: {e}")
        
        finally:
            # Ensure cleanup happens regardless of success/failure
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError as e:
                    logger.warning(f"Failed to cleanup temporary file {temp_file}: {e}")

    def _try_extract_zip(self, file_path: Path, dest_path: Path) -> bool:
        """Attempt to extract a ZIP file."""
        try:
            if zipfile.is_zipfile(str(file_path)):
                with zipfile.ZipFile(file_path) as zipf:
                    safe_names = [m for m in zipf.namelist() if not (os.path.isabs(m) or '..' in m)]
                    zipf.extractall(path=dest_path, members=safe_names)
                    logger.info("Extracted as ZIP archive.")
                    return True
        except zipfile.BadZipFile as e:
            logger.debug(f"Not a ZIP file: {e}")
        return False

    def _try_extract_tar(self, file_path: Path, dest_path: Path) -> bool:
        """Attempt to extract a file as a gzipped tar or regular tar."""
        try:
            if tarfile.is_tarfile(str(file_path)):
                mode = "r:gz" if self._is_gzipped(file_path) else "r"
                with tarfile.open(file_path, mode=mode) as tar:
                    logger.info(f"Extracting as {mode} tar archive.")
                    safe_members = [m for m in tar.getmembers() if not (os.path.isabs(m.name) or '..' in m.name)]
                    tar.extractall(path=dest_path, members=safe_members)
                    return True
        except tarfile.TarError as e:
            logger.debug(f"Not a tar file: {e}")
        return False

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on magic bytes and content inspection."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
            # Check magic bytes for binary formats
            if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06'):
                return 'zip'
            elif header.startswith(b'\x1f\x8b'):
                return 'gzip'
            elif header.startswith(b'%PDF'):
                return 'pdf'
            elif b'ustar' in header[257:262]:  # TAR magic at offset 257
                return 'tar'
            else:
                # For text files, check content for TeX indicators
                try:
                    content_str = header.decode('utf-8', errors='ignore')
                    if '\\documentclass' in content_str or '\\begin{document}' in content_str:
                        return 'tex'
                except:
                    pass
                
                return 'unknown'
                
        except (IOError, OSError) as e:
            logger.warning(f"Could not read file for type detection: {e}")
            return 'unknown'

    def _is_gzipped(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                return magic == b'\x1f\x8b'
        except (IOError, OSError):
            return False

    def _try_extract_gzip(self, file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
        """Attempt to decompress a gzipped file."""
        try:
            with gzip.open(file_path, 'rb') as gz_file:
                content = gz_file.read()
            
            # Heuristic to check if it's a TeX file
            if b'\\documentclass' in content or b'\\begin{document}' in content:
                tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
                tex_file.write_bytes(content)
                logger.info("File detected as gzipped TeX file.")
                return True
        except gzip.BadGzipFile as e:
            logger.debug(f"Not a gzip file: {e}")
        return False

    def _try_handle_plain_text(self, file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
        """Attempt to handle the file as a plain text LaTeX file."""
        try:
            content = file_path.read_bytes()
            if content.startswith(b'%PDF'):
                raise ArxivExtractorError(f"Paper {arxiv_id} is PDF-only. LaTeX source is required.")
            
            content_str = content.decode('utf-8', errors='ignore')
            if '\\documentclass' in content_str or '\\begin{document}' in content_str:
                tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
                tex_file.write_text(content_str, encoding='utf-8')
                logger.info("File detected as plain TeX file.")
                return True
        except UnicodeDecodeError:
            raise ArxivExtractorError("Failed to decode file. It might be a binary format.")
        return False
    
    def find_tex_files(self, source_path):
        """Find all LaTeX files in the directory - maintains original signature."""
        source_dir = Path(source_path)
        tex_files = list(source_dir.glob('**/*.tex'))
        if not tex_files:
            all_files = list(source_dir.glob('*'))
            logger.error(f"No .tex files found. Available files: {[f.name for f in all_files]}")
            return []
        
        logger.info(f"Found {len(tex_files)} .tex files: {[f.name for f in tex_files]}")
        return [str(f) for f in tex_files]
    
    def read_latex_content(self, tex_files: List[str]) -> str:
        """Read and combine content from all found LaTeX files."""
        full_content = []
        for tex_file_str in sorted(tex_files):  # Sort for deterministic order
            tex_file = Path(tex_file_str)
            try:
                content = tex_file.read_text(encoding='utf-8', errors='ignore')
                # Add a comment to mark the beginning of each file
                full_content.append(f"\n% --- Source File: {tex_file.name} ---\n{content}")
                logger.debug(f"Read {len(content)} characters from {tex_file.name}")
            except Exception as e:
                logger.warning(f"Could not read {tex_file.name}: {e}")
        
        combined_content = ''.join(full_content)
        logger.info(f"Combined total LaTeX content: {len(combined_content)} characters")
        return combined_content

    def download_and_read_latex(self, source_url: str, output_dir: str) -> tuple[str, List[str]]:
        """Convenience method: download, extract, and read LaTeX content in one call"""
        source_dir, success = self.download_source(source_url, output_dir)
        
        if not success or source_dir is None:
            raise ArxivExtractorError("Failed to download source")
        
        tex_files = self.find_tex_files(source_dir)
        if not tex_files:
            raise FileNotFoundError("No .tex files found in the extracted source.")
            
        latex_content = self.read_latex_content(tex_files)
        
        return latex_content, tex_files
    
    def clear_cache(self):
        import shutil
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
                logger.info(f"Cache cleared: {self.cache_dir}")
            else:
                logger.info(f"Cache directory doesn't exist: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

class AsyncSourceDownloader:
    """Async version of the ArXiv source downloader"""
    
    def __init__(self, cache_dir=None, config: Optional[Dict[str, Any]] = None):
        """Initialize async downloader with cache directory and optional configuration."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "arxiv_cache"
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.http_client = None  # Will be created when needed
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.http_client = httpx.AsyncClient(
            timeout=self.config['timeout'],
            follow_redirects=True,
            headers={'User-Agent': 'ArxivConjectureScraper/1.0 (For academic research)'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.http_client:
            await self.http_client.aclose()
    
    def validate_arxiv_id(self, arxiv_id: str) -> str:
        """Validate and normalize arXiv ID format."""
        arxiv_id = arxiv_id.strip().lower()

        # Regex for new (YYMM.NNNN) and old (subject-class/YYMMnnn) formats
        patterns = [
            r'^\d{4}\.\d{4,5}(v\d+)?$',
            r'^[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$',
        ]

        if not any(re.match(pattern, arxiv_id) for pattern in patterns):
            raise ValueError(f"Invalid arXiv ID format: {arxiv_id}")

        return arxiv_id

    async def async_download_and_read_latex(self, arxiv_id: str) -> Optional[str]:
        """
        Main async method to download, extract, and read the primary .tex file.
        
        Args:
            arxiv_id: The arXiv identifier (e.g., '2305.12345').
            
        Returns:
            The content of the main .tex file as a string, or None on failure.
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

            main_tex_file = await self._async_find_main_tex_file(extract_dir)
            if not main_tex_file:
                logger.error(f"Could not find a main .tex file in the extracted source for {arxiv_id}.")
                return None
            
            logger.info(f"Identified main LaTeX file: {main_tex_file.name}")

            async with aiofiles.open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()

            return content

        except Exception as e:
            logger.error(f"An error occurred during the download/extraction process for {arxiv_id}: {e}")
            return None

    async def _async_download_source(self, arxiv_id: str, download_dir: Path) -> Optional[Path]:
        """Download the source archive for the given arXiv ID."""
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        download_dir.mkdir(parents=True, exist_ok=True)
        sanitized_id = arxiv_id.replace('/', '_')
        download_path = download_dir / f"{sanitized_id}.tar.gz"

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
                else:
                    logger.error(f"All download attempts failed for {arxiv_id}.")
                    return None
        return None

    def _blocking_extract(self, source_path: Path, extract_dir: Path, arxiv_id: str):
        """Blocking extraction method to be run in a thread."""
        extract_dir.mkdir(exist_ok=True, parents=True)
        
        file_type = self._detect_file_type(source_path)
        logger.info(f"Detected file type: {file_type}")
        
        if file_type == 'tar' and self._try_extract_tar(source_path, extract_dir):
            pass
        elif file_type == 'gzip' and self._try_extract_gzip(source_path, extract_dir, arxiv_id):
            pass
        elif file_type == 'zip' and self._try_extract_zip(source_path, extract_dir):
            pass
        elif file_type == 'tex' and self._try_handle_plain_text(source_path, extract_dir, arxiv_id):
            pass
        elif file_type == 'pdf':
            raise ArxivExtractorError(f"Paper {arxiv_id} is PDF-only. LaTeX source is required.")
        else:
            # Fallback: try all methods if detection fails or returns unknown
            logger.warning(f"Unknown file type '{file_type}', trying all extraction methods...")
            if not (self._try_extract_tar(source_path, extract_dir) or
                   self._try_extract_gzip(source_path, extract_dir, arxiv_id) or
                   self._try_extract_zip(source_path, extract_dir) or
                   self._try_handle_plain_text(source_path, extract_dir, arxiv_id)):
                raise ArxivExtractorError("Unable to extract or identify downloaded file format.")

        logger.info(f"Successfully processed source to {extract_dir}")

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on magic bytes and content inspection."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
            # Check magic bytes for binary formats
            if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06'):
                return 'zip'
            elif header.startswith(b'\x1f\x8b'):
                return 'gzip'
            elif header.startswith(b'%PDF'):
                return 'pdf'
            elif b'ustar' in header[257:262]:  # TAR magic at offset 257
                return 'tar'
            else:
                # For text files, check content for TeX indicators
                try:
                    content_str = header.decode('utf-8', errors='ignore')
                    if '\\documentclass' in content_str or '\\begin{document}' in content_str:
                        return 'tex'
                except:
                    pass
                
                return 'unknown'
                
        except (IOError, OSError) as e:
            logger.warning(f"Could not read file for type detection: {e}")
            return 'unknown'

    def _is_gzipped(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                return magic == b'\x1f\x8b'
        except (IOError, OSError):
            return False

    def _try_extract_zip(self, file_path: Path, dest_path: Path) -> bool:
        """Attempt to extract a ZIP file."""
        try:
            if zipfile.is_zipfile(str(file_path)):
                with zipfile.ZipFile(file_path) as zipf:
                    safe_names = [m for m in zipf.namelist() if not (os.path.isabs(m) or '..' in m)]
                    zipf.extractall(path=dest_path, members=safe_names)
                    logger.info("Extracted as ZIP archive.")
                    return True
        except zipfile.BadZipFile as e:
            logger.debug(f"Not a ZIP file: {e}")
        return False

    def _try_extract_tar(self, file_path: Path, dest_path: Path) -> bool:
        """Attempt to extract a file as a gzipped tar or regular tar."""
        try:
            if tarfile.is_tarfile(str(file_path)):
                mode = "r:gz" if self._is_gzipped(file_path) else "r"
                with tarfile.open(file_path, mode=mode) as tar:
                    logger.info(f"Extracting as {mode} tar archive.")
                    safe_members = [m for m in tar.getmembers() if not (os.path.isabs(m.name) or '..' in m.name)]
                    tar.extractall(path=dest_path, members=safe_members)
                    return True
        except tarfile.TarError as e:
            logger.debug(f"Not a tar file: {e}")
        return False

    def _try_extract_gzip(self, file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
        """Attempt to decompress a gzipped file."""
        try:
            with gzip.open(file_path, 'rb') as gz_file:
                content = gz_file.read()
            
            # Heuristic to check if it's a TeX file
            if b'\\documentclass' in content or b'\\begin{document}' in content:
                tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
                tex_file.write_bytes(content)
                logger.info("File detected as gzipped TeX file.")
                return True
        except gzip.BadGzipFile as e:
            logger.debug(f"Not a gzip file: {e}")
        return False

    def _try_handle_plain_text(self, file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
        """Attempt to handle the file as a plain text LaTeX file."""
        try:
            content = file_path.read_bytes()
            if content.startswith(b'%PDF'):
                raise ArxivExtractorError(f"Paper {arxiv_id} is PDF-only. LaTeX source is required.")
            
            content_str = content.decode('utf-8', errors='ignore')
            if '\\documentclass' in content_str or '\\begin{document}' in content_str:
                tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
                tex_file.write_text(content_str, encoding='utf-8')
                logger.info("File detected as plain TeX file.")
                return True
        except UnicodeDecodeError:
            raise ArxivExtractorError("Failed to decode file. It might be a binary format.")
        return False

    async def _async_extract_source(self, source_path: Path, extract_dir: Path, arxiv_id: str):
        """Extract the downloaded source archive."""
        logger.info(f"Extracting {source_path.name} to {extract_dir}...")
        await asyncio.to_thread(self._blocking_extract, source_path, extract_dir, arxiv_id)

    def _blocking_find_main_file(self, search_dir: Path) -> Optional[Path]:
        """Find the main .tex file in the extracted directory."""
        tex_files = list(search_dir.rglob("*.tex"))
        
        if not tex_files: 
            return None
        if len(tex_files) == 1: 
            return tex_files[0]
        
        # Look for files containing \documentclass
        for f_path in tex_files:
            try:
                head = f_path.read_text(encoding='utf-8', errors='ignore')[:1024]
                if r"\documentclass" in head:
                    return f_path
            except Exception:
                continue
        
        # Fallback: return the file with the shortest path (likely main file)
        return min(tex_files, key=lambda p: len(str(p)))

    async def _async_find_main_tex_file(self, search_dir: Path) -> Optional[Path]:
        """Find the main .tex file asynchronously."""
        logger.info(f"Searching for main .tex file in {search_dir}...")
        return await asyncio.to_thread(self._blocking_find_main_file, search_dir)

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
    
    