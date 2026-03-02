"""Path-aware expansion using dependency edges."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

import networkx as nx
from loguru import logger


def build_dependency_graph(edges: List[Dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            graph.add_edge(source, target)
    return graph


def expand_with_prereqs(
    results: List[str],
    dep_graph: nx.DiGraph,
    node_types: Dict[str, str],
    allowed_types: Iterable[str],
) -> List[str]:
    allowed = set(allowed_types)
    expanded: List[str] = list(results)
    seen: Set[str] = set(results)
    for artifact_id in list(results):
        node_id = artifact_id.split("#", 1)[0]
        if node_id not in dep_graph:
            continue
        for prereq in dep_graph.successors(node_id):
            if prereq in seen:
                continue
            prereq_type = node_types.get(prereq)
            if prereq_type and prereq_type not in allowed:
                continue
            expanded.append(prereq)
            seen.add(prereq)
    if len(expanded) != len(results):
        logger.debug("Expanded {} -> {} results", len(results), len(expanded))
    return expanded
