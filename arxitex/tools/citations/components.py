"""Extract largest connected components from external-reference arXiv matches.

We treat a directed citation edge as:

    paper_id (citing)  ->  matched_arxiv_id (cited)

Connectivity for component extraction is UNDIRECTED: two papers are in the same
component if there exists any citation path between them.

Nodes are arXiv base identifiers only.

Example
-------

    python -m arxitex.tools.citations.components \
      --db-path pipeline_output/arxitex_indices.db \
      --top-k 10

This prints a summary to stdout and writes component JSON files under:

    pipeline_output/extref_components/
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema
from arxitex.tools.citations.openalex import strip_arxiv_version


@dataclass(frozen=True)
class DirectedEdge:
    source: str
    target: str


@dataclass
class ComponentResult:
    rank: int
    nodes: list[str]
    edges: list[DirectedEdge]
    node_count: int
    edge_count: int
    top_out_degree: list[tuple[str, int]]

    def to_json_dict(self) -> dict:
        return {
            "rank": self.rank,
            "stats": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "top_out_degree": self.top_out_degree,
            },
            "nodes": self.nodes,
            "edges": [{"source": e.source, "target": e.target} for e in self.edges],
        }


def _default_out_dir() -> Path:
    # Repo root is two parents up from arxitex/tools/...
    return Path(__file__).resolve().parents[2] / "pipeline_output" / "extref_components"


def _load_edges(
    db_path: str | Path,
    *,
    only_processed_success: bool = False,
    normalize_arxiv_ids: bool = True,
) -> list[DirectedEdge]:
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        if only_processed_success:
            rows = conn.execute(
                """
                SELECT m.paper_id, m.matched_arxiv_id
                FROM external_reference_arxiv_matches m
                JOIN processed_papers p
                  ON p.arxiv_id = m.paper_id
                WHERE p.status LIKE 'success%'
                  AND m.matched_arxiv_id IS NOT NULL
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT paper_id, matched_arxiv_id
                FROM external_reference_arxiv_matches
                WHERE matched_arxiv_id IS NOT NULL
                """
            ).fetchall()
        out: list[DirectedEdge] = []
        for r in rows:
            src = str(r[0] or "").strip()
            dst = str(r[1] or "").strip()
            if not src or not dst:
                continue
            if normalize_arxiv_ids:
                src = strip_arxiv_version(src)
                dst = strip_arxiv_version(dst)
            out.append(DirectedEdge(source=src, target=dst))
        return out
    finally:
        conn.close()


def _connected_components(
    nodes: Iterable[str], undirected_adj: dict[str, set[str]]
) -> list[list[str]]:
    """Return components as lists of node ids (unsorted)."""

    seen: set[str] = set()
    comps: list[list[str]] = []

    for n in nodes:
        if n in seen:
            continue
        seen.add(n)
        q: deque[str] = deque([n])
        comp: list[str] = [n]
        while q:
            cur = q.popleft()
            for nb in undirected_adj.get(cur, set()):
                if nb in seen:
                    continue
                seen.add(nb)
                q.append(nb)
                comp.append(nb)
        comps.append(comp)

    return comps


def extract_top_k_reference_components(
    *,
    db_path: str | Path,
    top_k: int = 10,
    min_size: int = 1,
    only_processed_success: bool = False,
    normalize_arxiv_ids: bool = True,
) -> list[ComponentResult]:
    """Compute top-K largest undirected connected components.

    Components are computed from directed citation edges in
    `external_reference_arxiv_matches` by ignoring direction for connectivity.
    Returned JSON edges remain directed (citing -> cited).
    """

    edges = _load_edges(
        db_path,
        only_processed_success=only_processed_success,
        normalize_arxiv_ids=normalize_arxiv_ids,
    )

    undirected_adj: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for e in edges:
        all_nodes.add(e.source)
        all_nodes.add(e.target)
        undirected_adj[e.source].add(e.target)
        undirected_adj[e.target].add(e.source)

    comps = _connected_components(sorted(all_nodes), undirected_adj)
    comps = [c for c in comps if len(c) >= int(min_size)]
    comps.sort(key=len, reverse=True)

    out: list[ComponentResult] = []
    k = max(0, int(top_k))

    # Pre-group directed edges by component membership for efficiency.
    for idx, comp_nodes in enumerate(comps[:k], start=1):
        comp_set = set(comp_nodes)
        directed_edges = [
            e for e in edges if e.source in comp_set and e.target in comp_set
        ]
        out_deg = Counter(e.source for e in directed_edges)
        top_out = out_deg.most_common(20)
        out.append(
            ComponentResult(
                rank=idx,
                nodes=sorted(comp_nodes),
                edges=directed_edges,
                node_count=len(comp_nodes),
                edge_count=len(directed_edges),
                top_out_degree=top_out,
            )
        )

    return out


def _write_components(out_dir: Path, comps: list[ComponentResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for comp in comps:
        path = out_dir / f"component_{comp.rank:03d}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(comp.to_json_dict(), f, ensure_ascii=False, indent=2)


def _chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1", (table,)
    ).fetchone()
    return row is not None


def _load_titles_from_papers(conn, paper_ids: list[str]) -> dict[str, str]:
    """Return base_id -> title from normalized `papers` table.

    Important: in this repo, `papers.paper_id` is typically stored *with* version
    suffix (e.g. 1110.0099v1). Component nodes are base ids.
    We therefore query both:
      - exact `paper_id IN (<base ids>)` (in case some are stored without version)
      - `paper_id LIKE '<base>v%'` (most common)
    and normalize results back to base ids.
    """

    if not paper_ids:
        return {}
    if not _table_exists(conn, "papers"):
        return {}

    out: dict[str, str] = {}

    # Chunk to avoid SQLite parameter limit.
    for chunk in _chunked(paper_ids, 150):
        in_placeholders = ",".join(["?"] * len(chunk))
        like_params = [f"{pid}v%" for pid in chunk]
        like_clause = " OR ".join(["paper_id LIKE ?" for _ in like_params])

        where = f"paper_id IN ({in_placeholders})"
        params: list[str] = list(chunk)
        if like_clause:
            where = f"({where}) OR ({like_clause})"
            params.extend(like_params)

        rows = conn.execute(
            f"SELECT paper_id, title FROM papers WHERE {where}",
            params,
        ).fetchall()

        for r in rows:
            raw_id = str(r["paper_id"] or "").strip()
            if not raw_id:
                continue
            base_id = strip_arxiv_version(raw_id)
            if base_id not in paper_ids:
                continue
            if base_id in out:
                continue
            title = (r["title"] or "").strip() if r["title"] is not None else ""
            if title:
                out[base_id] = title

    return out


def _load_titles_from_discovered(conn, paper_ids: list[str]) -> dict[str, str]:
    """Best-effort: pull titles from legacy discovered_papers.metadata JSON.

    The discovery queue may store arxiv_id with version suffix. We normalize to base ids.
    """

    if not paper_ids:
        return {}
    if not _table_exists(conn, "discovered_papers"):
        return {}

    # Ensure the table has expected columns.
    try:
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(discovered_papers)").fetchall()
        }
    except Exception:
        return {}
    if "arxiv_id" not in cols or "metadata" not in cols:
        return {}

    # Query both exact ids and common versioned variants via LIKE.
    # Keep this bounded by only using ids we need.
    out: dict[str, str] = {}

    # Split into chunks to avoid hitting SQLite parameter limit.
    for chunk in _chunked(paper_ids, 150):
        like_params = [f"{pid}v%" for pid in chunk]
        in_placeholders = ",".join(["?"] * len(chunk))
        like_clause = " OR ".join(["arxiv_id LIKE ?" for _ in like_params])
        where = f"arxiv_id IN ({in_placeholders})"
        params: list[str] = list(chunk)
        if like_clause:
            where = f"({where}) OR ({like_clause})"
            params.extend(like_params)

        try:
            rows = conn.execute(
                f"SELECT arxiv_id, metadata FROM discovered_papers WHERE {where}",
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return {}

        for r in rows:
            raw_id = str(r["arxiv_id"] or "").strip()
            if not raw_id:
                continue
            base_id = strip_arxiv_version(raw_id)
            if base_id not in paper_ids:
                continue
            if base_id in out:
                continue

            meta_raw = r["metadata"]
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except Exception:
                continue
            title = (meta.get("title") or "").strip()
            if title:
                out[base_id] = title

    return out


def build_paper_titles_map(
    *, db_path: str | Path, paper_ids: list[str]
) -> dict[str, str]:
    """Build a base-arXiv-id -> title map for a set of node IDs."""

    if not paper_ids:
        return {}

    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        titles = _load_titles_from_papers(conn, paper_ids)

        missing = [pid for pid in paper_ids if pid not in titles]
        if missing:
            titles.update(_load_titles_from_discovered(conn, missing))

        return titles
    finally:
        conn.close()


def _write_paper_titles(out_dir: Path, titles: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paper_titles.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Extract the top-K largest connected components induced by external-reference arXiv matches. "
            "Connectivity ignores direction (undirected), but JSON edges are directed (citing -> cited)."
        )
    )
    p.add_argument(
        "--db-path",
        required=True,
        help="Path to the arxitex SQLite database (e.g. pipeline_output/arxitex_indices.db).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of largest components to output.",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Filter out components smaller than this many nodes.",
    )
    p.add_argument(
        "--only-processed-success",
        action="store_true",
        help="Restrict to citing paper_ids that are processed_papers with status LIKE 'success%'.",
    )
    p.add_argument(
        "--no-normalize-arxiv-ids",
        action="store_true",
        help=(
            "Do NOT normalize node IDs by stripping version suffixes (vN). "
            "By default, we normalize so e.g. '1202.1159v1' and '1202.1159' are merged."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Directory to write JSON component files. Default: pipeline_output/extref_components (repo-relative)."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = p.parse_args(argv)
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="DEBUG" if args.verbose else "INFO")

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()

    comps = extract_top_k_reference_components(
        db_path=args.db_path,
        top_k=args.top_k,
        min_size=args.min_size,
        only_processed_success=bool(args.only_processed_success),
        normalize_arxiv_ids=not bool(args.no_normalize_arxiv_ids),
    )

    # Summary
    logger.info(
        "Extracted {} components (top_k={}, min_size={}) -> writing to {}",
        len(comps),
        args.top_k,
        args.min_size,
        str(out_dir),
    )
    for c in comps:
        sample = ", ".join(c.nodes[:8])
        logger.info(
            "#{:02d}: nodes={} edges={} sample=[{}]",
            c.rank,
            c.node_count,
            c.edge_count,
            sample,
        )

    _write_components(out_dir, comps)

    # Also write a title map for the union of all nodes, for UI tooltips.
    all_node_ids = sorted({n for c in comps for n in c.nodes})
    titles = build_paper_titles_map(db_path=args.db_path, paper_ids=all_node_ids)
    _write_paper_titles(out_dir, titles)

    logger.info(
        "Wrote paper_titles.json for {}/{} nodes (missing titles fall back to arXiv id)",
        len(titles),
        len(all_node_ids),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
