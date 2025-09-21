import asyncio
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from arxitex.downloaders.async_downloader import read_and_combine_tex_files
from arxitex.extractor.dependency_inference.dependency_inference import (
    GraphDependencyInference,
)
from arxitex.extractor.graph_building.base_builder import BaseGraphBuilder
from arxitex.extractor.models import DocumentGraph, Edge
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.definition_builder.definition_builder import DefinitionBuilder
from arxitex.symdef.document_enhancer import DocumentEnhancer
from arxitex.symdef.utils import ContextFinder


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
            definition_bank=definition_bank,
        )

    async def build_graph(
        self,
        project_dir: Path,
        source_file: Optional[str] = None,
        infer_dependencies: bool = True,
        enrich_content: bool = True,
    ) -> DocumentGraph:
        logger.info("Starting Pass 1: Building base graph from LaTeX structure...")
        graph = self.regex_builder.build_graph(project_dir, source_file)

        if not graph.nodes:
            logger.warning("Regex pass found no artifacts. Aborting LLM analysis.")
            return DocumentGraph()

        latex_content = read_and_combine_tex_files(project_dir)
        bank = None
        artifact_to_terms_map = {}
        should_enrich = enrich_content or infer_dependencies

        if should_enrich:
            logger.info(
                "--- Starting Pass 2: Enriching artifact content with definitions ---"
            )
            try:
                enrichment_results = await self._enrich_artifact_content(
                    graph, latex_content
                )
                bank = enrichment_results["definition_bank"]
                artifact_to_terms_map = enrichment_results["artifact_to_terms_map"]
            except Exception as e:
                logger.error(
                    f"Content enrichment failed: {e}. Proceeding without enriched content.",
                    exc_info=True,
                )
                bank = DefinitionBank()
                artifact_to_terms_map = {}

        if infer_dependencies:
            if not artifact_to_terms_map:
                logger.warning(
                    "Cannot infer dependencies because term extraction failed or was skipped."
                )
            else:
                logger.info(
                    "--- Starting Pass 3: Inferring dependencies with efficient filtering ---"
                )
                graph = await self._infer_and_add_dependencies(
                    graph, artifact_to_terms_map, bank
                )

        reference_edges = len([e for e in graph.edges if e.reference_type])
        dependency_edges = len([e for e in graph.edges if e.dependency_type])
        logger.success(
            f"Hybrid extraction complete. Graph has {len(graph.nodes)} artifacts and {len(graph.edges)} total edges."
        )
        logger.info(
            f"Edge breakdown: {reference_edges} reference-based, {dependency_edges} dependency-based."
        )

        return graph, bank, artifact_to_terms_map

    async def _enrich_artifact_content(self, graph: DocumentGraph, latex_content: str):
        """
        Uses the DocumentEnhancer to create self-contained content for each node.
        """
        nodes_to_enhance = [node for node in graph.nodes if not node.is_external]
        if not nodes_to_enhance:
            logger.info("No internal nodes to enhance. Skipping content enrichment.")
            return {}

        logger.info(f"Enhancing content for {len(nodes_to_enhance)} artifacts...")

        enhanced_results = await self.document_enhancer.enhance_document(
            nodes_to_enhance, latex_content
        )
        definitions_map = enhanced_results.get("definitions_map", {})
        updated_count = 0
        for node in graph.nodes:
            if node.id in definitions_map:
                node.prerequisite_defs = definitions_map[node.id]
                updated_count += 1

        logger.success(
            f"Successfully added prerequisite definitions to {updated_count} artifacts."
        )
        return enhanced_results

    def _is_subword_of(self, term_a: str, term_b: str) -> bool:
        """
        Checks if term_a is a subword of term_b.
        This is true if the set of words in term_a is a proper subset of the words in term_b.
        e.g., "union closed" is a subword of "approximate union closed".
        """
        if term_a == term_b:
            return False

        # Split terms into component words.
        # This handles normal words, numbers, and LaTeX commands.
        words_a = set(re.findall(r"[a-zA-Z0-9_]+|\\?[a-zA-Z@]+", term_a))
        words_b = set(re.findall(r"[a-zA-Z0-9_]+|\\?[a-zA-Z@]+", term_b))

        if not words_a or not words_b:
            return False

        return words_a.issubset(words_b)

    async def _infer_and_add_dependencies(
        self,
        graph: DocumentGraph,
        artifact_to_terms_map: Dict[str, List[str]],
        bank: DefinitionBank,
    ) -> DocumentGraph:
        """
        Analyzes artifacts to infer dependencies efficiently using conceptual and
        term overlap to generate candidates.
        """
        internal_nodes = [node for node in graph.nodes if not node.is_external]
        if len(internal_nodes) < 2:
            logger.info("Not enough internal nodes to infer dependencies. Skipping.")
            return graph

        id_to_node_map = {node.id: node for node in internal_nodes}
        final_candidate_pairs = []

        # --- THIS IS THE KEY LOGIC BRANCH ---
        # Check if we have the necessary data to be "smart".
        has_enrichment_data = bool(bank and artifact_to_terms_map)

        if has_enrichment_data:
            # --- Phase 1: Build the "Conceptual Footprint" for Each Artifact ---
            logger.info(
                "Building conceptual footprint for each artifact by including term dependencies..."
            )
            artifact_footprints = defaultdict(set)
            for artifact in internal_nodes:
                direct_terms = artifact_to_terms_map.get(artifact.id, [])
                artifact_footprints[artifact.id].update(direct_terms)

                for term in direct_terms:
                    definition = await bank.find(term)
                    if definition and definition.dependencies:
                        artifact_footprints[artifact.id].update(definition.dependencies)

            # --- Phase 2: Generate Candidate Pairs from Overlaps ---
            logger.info(
                "Generating candidate pairs from conceptual and subword overlap..."
            )
            candidate_pairs_unordered = set()

            # Create all unique combinations of artifact IDs to compare.
            for id1, id2 in combinations(id_to_node_map.keys(), 2):
                footprint1 = artifact_footprints[id1]
                footprint2 = artifact_footprints[id2]

                # Rule 1: Check for direct conceptual overlap.
                if not footprint1.isdisjoint(footprint2):
                    candidate_pairs_unordered.add(tuple(sorted((id1, id2))))
                    continue

                # Rule 2: If no direct overlap, check for subword overlap.
                found_subword_link = False
                for term1 in footprint1:
                    for term2 in footprint2:
                        if self._is_subword_of(term1, term2) or self._is_subword_of(
                            term2, term1
                        ):
                            candidate_pairs_unordered.add(tuple(sorted((id1, id2))))
                            found_subword_link = True
                            break
                    if found_subword_link:
                        break

            # --- Phase 3: Filter, Finalize, and Verify with LLM ---
            for id1, id2 in candidate_pairs_unordered:
                node1, node2 = id_to_node_map[id1], id_to_node_map[id2]
                source_node, target_node = (
                    (node2, node1)
                    if node1.position.line_start < node2.position.line_start
                    else (node1, node2)
                )

                # Filter out pairs that already have a direct \ref edge in the graph.
                if not graph.find_edge(source_node.id, target_node.id):
                    final_candidate_pairs.append((source_node, target_node))

        else:
            logger.warning(
                "No enrichment data found. Falling back to checking all artifact pairs."
            )
            logger.info(
                "This will be less efficient and may produce lower-quality results, but ensures all possibilities are checked."
            )

            for node1, node2 in combinations(internal_nodes, 2):
                source_node, target_node = (
                    (node1, node2)
                    if node1.position.line_start < node2.position.line_start
                    else (node2, node1)
                )

                if not graph.find_edge(source_node.id, target_node.id):
                    final_candidate_pairs.append((source_node, target_node))

        if not final_candidate_pairs:
            logger.debug(
                "No promising candidate pairs found after filtering. Skipping LLM verification."
            )
            return graph

        logger.info(
            f"Generated {len(final_candidate_pairs)} candidate pairs for LLM verification."
        )

        tasks_with_context = []
        for source_node, target_node in final_candidate_pairs:
            task = asyncio.create_task(
                self.llm_dependency_checker.ainfer_dependency(
                    source_node.to_dict(), target_node.to_dict()
                )
            )
            tasks_with_context.append(
                {"source": source_node, "target": target_node, "task": task}
            )

        results = await asyncio.gather(
            *[t["task"] for t in tasks_with_context], return_exceptions=True
        )

        added_edges_count = 0
        for context, result in zip(tasks_with_context, results):
            if isinstance(result, Exception):
                source_id = context["source"].id
                target_id = context["target"].id
                logger.error(
                    f"Error processing pair ({source_id}, {target_id}): {result}"
                )
                continue

            if result and result.has_dependency:
                source_node = context["source"]
                target_node = context["target"]

                new_edge = Edge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    dependency_type=result.dependency_type,
                    dependency=result.justification
                    or "Inferred by LLM based on shared terminology.",
                )
                graph.add_edge(new_edge)
                added_edges_count += 1
                logger.debug(
                    f"Created new dependency edge: {source_node.id} -> {target_node.id} (Type: {result.dependency_type})"
                )

        logger.success(
            f"Added {added_edges_count} new dependency edges based on LLM verification."
        )
        return graph
