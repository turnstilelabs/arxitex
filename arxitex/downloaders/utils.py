
import os
import tarfile
import zipfile
import gzip
from loguru import logger
from pathlib import Path


def detect_file_type(file_path: Path) -> str:
    try:
        with open(file_path, 'rb') as f:
            header = f.read(512)

        if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06'):
            return 'zip'
        elif header.startswith(b'\x1f\x8b'):
            return 'gzip'
        elif header.startswith(b'%PDF'):
            return 'pdf'
        elif b'ustar' in header[257:262]:
            return 'tar'
        else:
            try:
                content_str = header.decode('utf-8', errors='ignore')
                if '\\documentclass' in content_str or '\\begin{document}' in content_str:
                    return 'tex'
            except:
                pass
            return 'unknown'
    except Exception as e:
        logger.warning(f"Could not read file for type detection: {e}")
        return 'unknown'

def is_gzipped(file_path: Path) -> bool:
    try:
        with open(file_path, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except:
        return False

def try_extract_tar(file_path: Path, dest_path: Path) -> bool:
    try:
        if tarfile.is_tarfile(str(file_path)):
            mode = "r:gz" if is_gzipped(file_path) else "r"
            with tarfile.open(file_path, mode=mode) as tar:
                safe_members = [m for m in tar.getmembers() if not (os.path.isabs(m.name) or '..' in m.name)]
                tar.extractall(path=dest_path, members=safe_members)
                logger.info(f"Extracted as {mode} tar archive.")
                return True
    except tarfile.TarError as e:
        logger.debug(f"Not a tar file: {e}")
    return False

def try_extract_zip(file_path: Path, dest_path: Path) -> bool:
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

def try_extract_gzip(file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
    try:
        with gzip.open(file_path, 'rb') as gz_file:
            content = gz_file.read()

        if b'\\documentclass' in content or b'\\begin{document}' in content:
            tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
            tex_file.write_bytes(content)
            logger.info("File detected as gzipped TeX file.")
            return True
    except gzip.BadGzipFile as e:
        logger.debug(f"Not a gzip file: {e}")
    return False

def try_handle_plain_text(file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
    try:
        content = file_path.read_bytes()
        if content.startswith(b'%PDF'):
            return False
        
        content_str = content.decode('utf-8', errors='ignore')
        if '\\documentclass' in content_str or '\\begin{document}' in content_str:
            tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
            tex_file.write_text(content_str, encoding='utf-8')
            logger.info("File detected as plain TeX file.")
            return True
    except Exception:
        pass
    return False