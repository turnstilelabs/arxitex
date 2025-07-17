"""
Builds a graph using a robust, pairwise hybrid approach:
1. Regex-based extraction of all artifacts and their full content.
2. LLM-based analysis on every ordered pair of artifacts to determine
   the precise nature of their relationship, if any.
"""

import asyncio
from itertools import combinations
from loguru import logger

from arxitex.extractor.core.base_extractor import build_graph_from_latex
from arxitex.extractor.utils import (
    Edge, DocumentGraph)
from arxitex.extractor.dependency_inference.dependency_inference import GraphDependencyInference


async def build_graph_with_hybrid_model(latex_content: str) -> DocumentGraph:
    """
    Orchestrates the hybrid pairwise regex + LLM extraction process concurrently
    """
    # PASS 1: EXTRACT ALL NODES AND THEIR FULL CONTENT
    logger.info("Starting Pass 1: Calling regex builder to extract all artifacts...")
    document_graph = build_graph_from_latex(latex_content)

    if not document_graph.nodes:
        logger.warning("Regex pass found no artifacts. Aborting LLM analysis.")
        return DocumentGraph()
    
    logger.info(f"Regex pass completed. Found {len(document_graph.nodes)} artifacts.")

    # PASS 2: PAIRWISE DEPENDENCY ANALYSIS WITH LLM
    node_pairs = list(combinations(document_graph.nodes, 2)) 
    if not node_pairs:
        logger.info("No pairs to analyze. Skipping LLM pass.")
        return document_graph
    
    logger.info(f"Starting Pass 2: Concurrently analyzing {len(node_pairs)} artifact pairs with LLM.")
    
    dependency_checker = GraphDependencyInference()
    tasks_with_context = []
    for i, (source_node, target_node) in enumerate(node_pairs):
        source_dict = source_node.to_dict()
        target_dict = target_node.to_dict()
        
        task = asyncio.create_task(
            dependency_checker.ainfer_dependency(source_dict, target_dict)
        )
        tasks_with_context.append((source_node, target_node, task))

    tasks = [t for _, _, t in tasks_with_context]
    all_results = []
    if tasks:
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.debug(f"LLM analysis completed for {len(all_results)} pairs.")

    for (source_node, target_node, _), result in zip(tasks_with_context, all_results):
        if isinstance(result, Exception):
            logger.error(f"Could not process pair ({source_node.id}, {target_node.id}): {result}")
        elif result and result.has_dependency:
            existing_edge = None
            for edge in document_graph.edges:
                if edge.source_id == source_node.id and edge.target_id == target_node.id:
                    existing_edge = edge
                    break
            
            if existing_edge:
                existing_edge.dependency_type = result.dependency_type
                existing_edge.dependency = result.justification or "No justification provided by LLM."
                logger.debug(f"Updated existing edge: {source_node.id} -> {target_node.id} with dependency type: {result.dependency_type}")
            else:
                new_edge = Edge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    dependency_type=result.dependency_type,
                    dependency=result.justification or "No justification provided by LLM."
                )
                document_graph.add_edge(new_edge)
                logger.debug(f"Created new dependency edge: {source_node.id} -> {target_node.id} (Type: {result.dependency_type})")
        else:
            logger.debug(f"No dependency found between {source_node.id} and {target_node.id}. No edge created.")
            logger.debug(f"Result: {result} for pair ({source_node.id}, {target_node.id})")
    
    logger.info(f"Hybrid extraction complete. Found {len(document_graph.nodes)} artifacts and {len(document_graph.edges)} total edges.")
    
    reference_edges = len([e for e in document_graph.edges if hasattr(e, 'context') and e.context])
    dependency_edges = len([e for e in document_graph.edges if e.dependency_type is not None])
    
    logger.info(f"Edge breakdown: {reference_edges} reference edges, {dependency_edges} dependency edges")
    return document_graph