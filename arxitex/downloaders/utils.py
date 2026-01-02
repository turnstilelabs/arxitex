import gzip
import os
import tarfile
import zipfile
from pathlib import Path

from loguru import logger


class ExtractionError(Exception):
    """Raised when a source archive appears to be the right format but is corrupted.

    AsyncSourceDownloader._blocking_extract upgrades this into an
    ArxivExtractorError so that higher layers can classify and report
    the problem in a structured way.
    """

    pass


def detect_file_type(file_path: Path) -> str:
    try:
        with open(file_path, "rb") as f:
            header = f.read(512)

        if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06"):
            return "zip"
        elif header.startswith(b"\x1f\x8b"):
            return "gzip"
        elif header.startswith(b"%PDF"):
            return "pdf"
        elif b"ustar" in header[257:262]:
            return "tar"
        else:
            try:
                content_str = header.decode("utf-8", errors="ignore")
                if (
                    "\\documentclass" in content_str
                    or "\\begin{document}" in content_str
                ):
                    return "tex"
            except UnicodeDecodeError:
                pass
            return "unknown"
    except Exception as e:
        logger.warning(f"Could not read file for type detection: {e}")
        return "unknown"


def is_gzipped(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def try_extract_tar(file_path: Path, dest_path: Path) -> bool:
    """Try to extract a tar (optionally gzipped) archive.

    Returns False if the file is clearly not a tar archive.
    Raises ExtractionError if it looks like a tar but is corrupted
    or cannot be fully read.
    """

    try:
        if not tarfile.is_tarfile(str(file_path)):
            return False

        mode = "r:gz" if is_gzipped(file_path) else "r"
        try:
            with tarfile.open(file_path, mode=mode) as tar:
                safe_members = [
                    m
                    for m in tar.getmembers()
                    if not (os.path.isabs(m.name) or ".." in m.name)
                ]
                tar.extractall(path=dest_path, members=safe_members)
                logger.info(f"Extracted as {mode} tar archive.")
                return True
        except tarfile.TarError as e:
            # At this point tarfile.is_tarfile was True, so treat this as
            # a genuinely corrupted archive rather than "not a tar".
            msg = f"Tar archive is corrupted or unreadable: {e}"
            logger.error(msg)
            raise ExtractionError(msg) from e

    except ExtractionError:
        # Propagate our own signal for the caller to upgrade.
        raise
    except Exception as e:
        logger.debug(f"Failed to handle tar archive candidate: {e}")
        return False


def try_extract_zip(file_path: Path, dest_path: Path) -> bool:
    """Try to extract a ZIP archive.

    Returns False if the file is clearly not a ZIP archive. Raises
    ExtractionError if it appears to be a ZIP but is corrupted.
    """

    try:
        if not zipfile.is_zipfile(str(file_path)):
            return False

        try:
            with zipfile.ZipFile(file_path) as zipf:
                safe_names = [
                    m for m in zipf.namelist() if not (os.path.isabs(m) or ".." in m)
                ]
                zipf.extractall(path=dest_path, members=safe_names)
                logger.info("Extracted as ZIP archive.")
                return True
        except zipfile.BadZipFile as e:
            msg = f"ZIP archive is corrupted or unreadable: {e}"
            logger.error(msg)
            raise ExtractionError(msg) from e

    except ExtractionError:
        raise
    except Exception as e:
        logger.debug(f"Failed to handle ZIP archive candidate: {e}")
        return False


def try_extract_gzip(file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
    """Try to extract a single gzipped TeX file.

    Returns False if the file does not appear to be gzip-compressed. Raises
    ExtractionError if it is gzip-compressed but corrupted.
    """

    try:
        with gzip.open(file_path, "rb") as gz_file:
            content = gz_file.read()

        if b"\\documentclass" in content or b"\\begin{document}" in content:
            tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
            tex_file.write_bytes(content)
            logger.info("File detected as gzipped TeX file.")
            return True
    except gzip.BadGzipFile as e:
        # Distinguish between "not actually gzip" and "gzip but broken".
        from arxitex.downloaders.utils import is_gzipped  # local import to avoid cycles

        if is_gzipped(file_path):
            msg = f"Gzip archive is corrupted and cannot be decompressed: {e}"
            logger.error(msg)
            raise ExtractionError(msg) from e
        logger.debug(f"Not a gzip file: {e}")
    except ExtractionError:
        raise
    except Exception as e:
        logger.debug(f"Failed to handle gzip candidate: {e}")
    return False


def try_handle_plain_text(file_path: Path, dest_path: Path, arxiv_id: str) -> bool:
    try:
        content = file_path.read_bytes()
        if content.startswith(b"%PDF"):
            return False

        content_str = content.decode("utf-8", errors="ignore")
        if "\\documentclass" in content_str or "\\begin{document}" in content_str:
            tex_file = dest_path / f"{arxiv_id.replace('/', '_')}.tex"
            tex_file.write_text(content_str, encoding="utf-8")
            logger.info("File detected as plain TeX file.")
            return True
    except Exception:
        pass
    return False


def read_and_combine_tex_files(project_dir: Path) -> str:
    """
    Finds all .tex files in a directory, reads them, and concatenates their content
    into a single string, adding source file comments.
    """
    if not project_dir.is_dir():
        logger.error(f"Provided path is not a directory: {project_dir}")
        return ""

    tex_files = sorted(list(project_dir.rglob("*.tex")))

    # Fallback: some archives contain a single extensionless file that is actually
    # LaTeX. If we see no *.tex files, scan for extensionless files that look like
    # TeX (by content) and treat them as sources.
    if not tex_files:
        candidates = []
        for p in project_dir.rglob("*"):
            if not p.is_file() or p.suffix:
                continue
            try:
                header = p.read_bytes()[:4096]
                text = header.decode("utf-8", errors="ignore")
            except Exception as e:
                logger.debug(f"Could not inspect extensionless file {p}: {e}")
                continue

            if "\\documentclass" in text or "\\begin{document}" in text:
                candidates.append(p)

        if candidates:
            logger.info(
                "No .tex files found, but detected %d extensionless LaTeX-looking files: %s",
                len(candidates),
                [c.name for c in candidates],
            )
            tex_files = sorted(candidates)
        else:
            logger.warning(
                f"No .tex or extensionless LaTeX files found in directory: {project_dir}"
            )
            return ""

    logger.info(
        f"Found {len(tex_files)} .tex files to parse: {[f.name for f in tex_files]}"
    )
    full_content = []
    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            full_content.append(f"\n% --- Source File: {tex_file.name} ---\n{content}")
        except Exception as e:
            logger.warning(f"Could not read {tex_file.name}: {e}")

    combined_content = "".join(full_content)
    logger.info(f"Combined total LaTeX content: {len(combined_content)} characters")
    return combined_content
