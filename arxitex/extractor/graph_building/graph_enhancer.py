import asyncio
from itertools import combinations
from typing import Optional
from loguru import logger

from arxitex.extractor.utils import (
    Edge, DocumentGraph)
from arxitex.extractor.dependency_inference.dependency_inference import GraphDependencyInference
from arxitex.extractor.graph_building.base_builder import BaseGraphBuilder
from arxitex.symdef.document_enhancer import DocumentEnhancer
from arxitex.symdef.utils import ContextFinder
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.definition_builder.definition_builder import DefinitionBuilder

class GraphEnhancer:
    """
    Orchestrates a comprehensive graph construction and enhancement process by:
    1. Building a base graph from LaTeX structure (Pass 1).
    2. Enhancing the graph with LLM-inferred dependencies (Pass 2).
    3. Enriching each artifact's content with prerequisite definitions (Pass 3).
    """

    def __init__(self):
        self.regex_builder = BaseGraphBuilder()
        self.llm_dependency_checker = GraphDependencyInference()

        definition_builder = DefinitionBuilder()
        context_finder = ContextFinder()
        definition_bank = DefinitionBank()
        
        self.document_enhancer = DocumentEnhancer(
            llm_enhancer=definition_builder,
            context_finder=context_finder,
            definition_bank=definition_bank
        )

    async def build_graph(self, latex_content: str, source_file: Optional[str] = None, 
                          infer_dependencies: bool = True, enrich_content: bool = True ) -> DocumentGraph:
        logger.info("Starting Pass 1: Building base graph from LaTeX structure...")
        graph = self.regex_builder.build_graph(latex_content, source_file)

        if not graph.nodes:
            logger.warning("Regex pass found no artifacts. Aborting LLM analysis.")
            return DocumentGraph()
        
        if infer_dependencies:
            logger.info("--- Starting Pass 2: Enhancing graph with LLM-inferred dependencies ---")
            graph = await self._infer_and_add_dependencies(graph)
        
        bank = None
        artifact_to_terms_map = {}
        if enrich_content:
            logger.info("--- Starting Pass 3: Enriching artifact content with definitions ---")
            bank, artifact_to_terms_map = await self._enrich_artifact_content(graph, latex_content)
       
        reference_edges = len([e for e in graph.edges if e.reference_type])
        dependency_edges = len([e for e in graph.edges if e.dependency_type])
        logger.success(
            f"Hybrid extraction complete. Graph has {len(graph.nodes)} artifacts and {len(graph.edges)} total edges."
        )
        logger.info(f"Edge breakdown: {reference_edges} reference-based, {dependency_edges} dependency-based.")
        
        return graph, bank, artifact_to_terms_map

    async def _infer_and_add_dependencies(self, graph: DocumentGraph):
        """
        Analyzes all pairs of nodes to infer dependencies and updates the graph.
        """
        node_pairs = list(combinations(graph.nodes, 2))
        if not node_pairs:
            logger.info("Not enough nodes to form pairs. Skipping LLM pass.")
            return
        
        logger.info(f"Concurrently analyzing {len(node_pairs)} artifact pairs with LLM.")
        
        tasks_with_context = []
        for source_node, target_node in node_pairs:
            # Skip pairs involving external nodes for dependency analysis
            if source_node.is_external or target_node.is_external:
                continue

            task = asyncio.create_task(
                self.llm_dependency_checker.ainfer_dependency(
                    source_node.to_dict(), target_node.to_dict()
                )
            )
            tasks_with_context.append({'source': source_node, 'target': target_node, 'task': task})

        tasks = [t['task'] for t in tasks_with_context]
        all_results = []
        if tasks:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for context, result in zip(tasks_with_context, all_results):
            source_node = context['source']
            target_node = context['target']
            
            if isinstance(result, Exception):
                logger.error(f"Error processing pair ({source_node.id}, {target_node.id}): {result}")
                continue
            
            if result and result.has_dependency:
                existing_edge = graph.find_edge(source_node.id, target_node.id)
                
                if existing_edge:
                    existing_edge.dependency_type = result.dependency_type
                    existing_edge.dependency = result.justification or "Justification not provided by LLM."
                    logger.debug(f"Updated existing edge {source_node.id} -> {target_node.id} with type: {result.dependency_type}")
                else:
                    new_edge = Edge(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        dependency_type=result.dependency_type,
                        dependency=result.justification or "Justification not provided by LLM."
                    )
                    graph.add_edge(new_edge)
                    logger.debug(f"Created new dependency edge: {source_node.id} -> {target_node.id} (Type: {result.dependency_type})")
        return graph

    async def _enrich_artifact_content(self, graph: DocumentGraph, latex_content: str):
        """
        Uses the DocumentEnhancer to create self-contained content for each node.
        """
        nodes_to_enhance = [node for node in graph.nodes if not node.is_external]
        if not nodes_to_enhance:
            logger.info("No internal nodes to enhance. Skipping content enrichment.")
            return
            
        logger.info(f"Enhancing content for {len(nodes_to_enhance)} artifacts...")
        
        enhanced_results = await self.document_enhancer.enhance_document(
            nodes_to_enhance, latex_content
        )
        enhanced_content_map = enhanced_results.get("artifacts", {})
        populated_bank = enhanced_results.get("definition_bank", {})
        artifact_to_terms_map = enhanced_results.get("artifact_to_terms_map", {})
        
        updated_count = 0
        for node in graph.nodes:
            if node.id in enhanced_content_map:
                node.content = enhanced_content_map[node.id]
                updated_count += 1
        
        logger.success(f"Successfully enriched the content of {updated_count} artifacts.")
        return populated_bank, artifact_to_terms_map
