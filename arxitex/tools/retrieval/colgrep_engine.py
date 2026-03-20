"""ColGREP CLI wrapper with flexible output parsing."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger


@dataclass
class ColGrepHit:
    path: str
    score: Optional[float]
    snippet: str


@dataclass
class ColGrepCandidate:
    statement_id: str
    score: Optional[float]
    path: str
    type: str
    number: str
    section: str
    subsection: str
    title: str
    arxiv_id: str
    text_preview: str
    prev_paragraph: str


_SCORE_RE = re.compile(r"score\s*[:=]\s*([-+]?\d*\.?\d+)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ColGrepEngine:
    def __init__(
        self,
        *,
        chunks_dir: str,
        index_dir: Optional[str] = None,
        colgrep_bin: str = "colgrep",
        timeout: int = 60,
    ) -> None:
        self.chunks_dir = str(chunks_dir)
        self.index_dir = str(index_dir) if index_dir else None
        self.colgrep_bin = colgrep_bin
        self.timeout = timeout
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        candidates = []
        chunks_dir = Path(self.chunks_dir)
        if chunks_dir.is_dir():
            candidates.append(chunks_dir / "manifest.jsonl")
        if self.index_dir:
            idx = Path(self.index_dir)
            if idx.is_dir():
                candidates.append(idx / "manifest.jsonl")
        for path in candidates:
            if path.exists():
                return _read_manifest(path)
        logger.warning("No manifest.jsonl found in {}", self.chunks_dir)
        return {}

    def build(self) -> None:
        if self.index_dir:
            logger.warning(
                "ColGREP does not support custom index dir; ignoring {}", self.index_dir
            )
        commands = [
            [self.colgrep_bin, "init", "-y", self.chunks_dir],
            [self.colgrep_bin, "init", self.chunks_dir],
        ]
        _run_first_success(commands, timeout=self.timeout)

    def search(self, query: str, k: int = 10) -> List[ColGrepHit]:
        if not query or not query.strip():
            return []
        query = _sanitize_query(query)
        if self.index_dir:
            logger.warning(
                "ColGREP does not support custom index dir; ignoring {}", self.index_dir
            )
        commands = [
            [
                self.colgrep_bin,
                "search",
                "--json",
                "-k",
                str(k),
                query,
                self.chunks_dir,
            ],
            [self.colgrep_bin, "search", "-k", str(k), query, self.chunks_dir],
            [self.colgrep_bin, "--json", "-k", str(k), query, self.chunks_dir],
            [self.colgrep_bin, "-k", str(k), query, self.chunks_dir],
        ]
        stdout = _run_first_success(commands, timeout=self.timeout)
        return parse_colgrep_output(stdout)

    def search_candidates(self, query: str, k: int = 10) -> List[ColGrepCandidate]:
        hits = self.search(query, k=k)
        if not hits:
            return []
        return _map_hits_to_candidates(hits, self._manifest)


def _sanitize_query(query: str) -> str:
    cleaned = _CONTROL_RE.sub(" ", query)
    cleaned = " ".join(cleaned.split())
    return cleaned


def _read_manifest(path: Path) -> Dict[str, Dict]:
    manifest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        statement_id = row.get("statement_id")
        file_path = row.get("path")
        if statement_id and file_path:
            manifest[os.path.abspath(file_path)] = row
    return manifest


def _run_first_success(commands: List[List[str]], timeout: int) -> str:
    errors = []
    for cmd in commands:
        try:
            logger.info("Running: {}", " ".join(cmd))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ColGREP binary not found: {cmd[0]}. Install or pass --colgrep-bin."
            ) from exc
        except subprocess.TimeoutExpired:
            errors.append(f"timeout: {' '.join(cmd)}")
            continue

        if proc.returncode == 0:
            stdout = proc.stdout or ""
            if stdout.strip():
                return stdout
            stderr = (proc.stderr or "").strip()
            if stderr:
                errors.append(stderr)
            else:
                return ""
        else:
            err = (proc.stderr or proc.stdout or "").strip()
            errors.append(err or f"exit {proc.returncode}")
    raise RuntimeError("ColGREP command failed: " + " | ".join(errors))


def parse_colgrep_output(stdout: str) -> List[ColGrepHit]:
    if not stdout:
        return []
    raw = stdout.strip()
    hits: List[ColGrepHit] = []

    # JSON array
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    hit = _parse_json_hit(item)
                    if hit:
                        hits.append(hit)
                return hits
        except Exception:
            pass

    # JSON lines
    lines = raw.splitlines()
    json_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                item = json.loads(line)
                json_lines += 1
                hit = _parse_json_hit(item)
                if hit:
                    hits.append(hit)
                continue
            except Exception:
                pass

        hit = _parse_text_hit(line)
        if hit:
            hits.append(hit)

    return hits


def _parse_json_hit(item: Dict) -> Optional[ColGrepHit]:
    if not isinstance(item, dict):
        return None
    path = item.get("path") or item.get("file") or item.get("filename")
    if not path:
        unit = item.get("unit") or {}
        if isinstance(unit, dict):
            path = unit.get("file") or unit.get("path")
    if not path:
        return None
    score = item.get("score")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    snippet = item.get("text") or item.get("snippet") or ""
    return ColGrepHit(path=str(path), score=score, snippet=str(snippet))


def _parse_text_hit(line: str) -> Optional[ColGrepHit]:
    if not line:
        return None
    path = None
    snippet = ""
    score = None

    # path:line:col (note: colgrep doesn't always include a colon after col)
    m = re.match(r"^(?P<path>.+):\d+:\d+\s*(?P<rest>.*)$", line)
    if m:
        path = m.group("path")
        snippet = m.group("rest")
    else:
        # path:
        m = re.match(r"^(?P<path>\S+):\s*(?P<rest>.*)$", line)
        if m:
            path = m.group("path")
            snippet = m.group("rest")
        else:
            # path (score=...)
            m = re.match(r"^(?P<path>\S+)\s+\((?P<rest>.*)\)$", line)
            if m:
                path = m.group("path")
                snippet = m.group("rest")

    if not path:
        return None

    sm = _SCORE_RE.search(line)
    if sm:
        try:
            score = float(sm.group(1))
        except Exception:
            score = None
    return ColGrepHit(path=path, score=score, snippet=snippet)


def _map_hits_to_candidates(
    hits: Iterable[ColGrepHit],
    manifest: Dict[str, Dict],
) -> List[ColGrepCandidate]:
    out: List[ColGrepCandidate] = []
    for hit in hits:
        abs_path = os.path.abspath(hit.path)
        row = manifest.get(abs_path)
        if row is None:
            # fallback: match by filename
            fname = os.path.basename(hit.path)
            row = next(
                (v for k, v in manifest.items() if os.path.basename(k) == fname),
                None,
            )
        if row is None:
            continue
        out.append(
            ColGrepCandidate(
                statement_id=row.get("statement_id") or "",
                score=hit.score,
                path=row.get("path") or hit.path,
                type=row.get("type") or "",
                number=row.get("number") or "",
                section=row.get("section") or "",
                subsection=row.get("subsection") or "",
                title=row.get("title") or "",
                arxiv_id=row.get("arxiv_id") or "",
                text_preview=row.get("text_preview") or "",
                prev_paragraph=row.get("prev_paragraph") or "",
            )
        )
    return out
