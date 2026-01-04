import asyncio
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from loguru import logger

from arxitex.downloaders.utils import read_and_combine_tex_files
from arxitex.extractor.dependency_inference.auto_mode import choose_mode_auto
from arxitex.extractor.dependency_inference.dependency_inference import (
    GraphDependencyInference,
)
from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
    DependencyInferenceMode,
)
from arxitex.extractor.dependency_inference.global_dependency_inference import (
    GlobalGraphDependencyInference,
)
from arxitex.extractor.dependency_inference.global_dependency_proposer import (
    GlobalDependencyProposer,
)
from arxitex.extractor.graph_building.base_builder import BaseGraphBuilder
from arxitex.extractor.models import ArtifactNode, DocumentGraph, Edge
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.definition_builder.definition_builder import DefinitionBuilder
from arxitex.symdef.document_enhancer import DocumentEnhancer
from arxitex.symdef.utils import ContextFinder, extract_latex_macros


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
        self.global_dependency_inferencer = GlobalGraphDependencyInference()
        self.global_dependency_proposer = GlobalDependencyProposer()

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
        dependency_mode: DependencyInferenceMode = "pairwise",
        dependency_config: Optional[DependencyInferenceConfig] = None,
        on_base_graph: Optional[Callable[[DocumentGraph], Awaitable[None]]] = None,
        on_enriched_node: Optional[Callable[[ArtifactNode], Awaitable[None]]] = None,
        on_dependency_edge: Optional[Callable[[Edge], Awaitable[None]]] = None,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> tuple[DocumentGraph, DefinitionBank, dict[str, list[str]], dict[str, str]]:
        logger.info(
            f"[{source_file}] Starting Pass 1: Building base graph from LaTeX structure..."
        )

        if on_status is not None:
            await on_status("building base graph")

        # PERF: read/concatenate LaTeX once and reuse for all subsequent passes.
        latex_content = read_and_combine_tex_files(project_dir)

        # Extract simple, argument-free LaTeX macros from the preamble so the
        # frontend (MathJax) can render paper-specific shorthand like "\\cF".
        try:
            latex_macros: dict[str, str] = extract_latex_macros(latex_content)
        except Exception:
            # Macro extraction is best-effort only; never break graph building.
            latex_macros = {}

        graph = self.regex_builder.build_graph_from_content(
            content=latex_content, source_file=source_file, project_dir=project_dir
        )

        # Attach macros to the graph instance for introspection/debugging. The
        # primary consumer is the higher-level pipeline, which returns
        # `latex_macros` separately alongside the graph.
        # Best-effort only: if graph doesn't support attribute assignment, we
        # let it fail silently.
        try:  # pragma: no cover - best-effort attribute set
            graph.latex_macros = latex_macros
        except Exception:
            pass

        if not graph.nodes:
            logger.warning("Regex pass found no artifacts. Aborting LLM analysis.")
            # Still return the (possibly empty) macro map so callers can
            # inspect it if needed.
            return DocumentGraph(), DefinitionBank(), {}, latex_macros

        # Give callers a chance to observe/stream the base regex graph before
        # any LLM-based enhancements are applied.
        if on_base_graph is not None:
            await on_base_graph(graph)

        bank = None
        artifact_to_terms_map = {}
        should_enrich = enrich_content or infer_dependencies

        if should_enrich:
            logger.info(
                "--- Starting Pass 2: Enriching artifact content with definitions ---"
            )
            if on_status is not None:
                await on_status("enriching symbols and definitions for artifacts")
            try:
                enrichment_results = await self._enrich_artifact_content(
                    graph, latex_content, on_enriched_node
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
                    f"[{source_file}] Cannot infer dependencies because term extraction "
                    "failed or was skipped."
                )
                if on_status is not None:
                    await on_status(
                        "skipping dependency inference (no enrichment data)"
                    )
            else:
                logger.info(
                    f"[{source_file}] --- Starting Pass 3: Inferring dependencies (mode-aware) ---"
                )
                if on_status is not None:
                    await on_status("inferring dependencies between artifacts")
                cfg = dependency_config or DependencyInferenceConfig()
                graph = await self._infer_and_add_dependencies_mode_aware(
                    graph=graph,
                    artifact_to_terms_map=artifact_to_terms_map,
                    bank=bank,
                    dependency_mode=dependency_mode,
                    cfg=cfg,
                    on_dependency_edge=on_dependency_edge,
                )

        reference_edges = len([e for e in graph.edges if e.reference_type])
        dependency_edges = len([e for e in graph.edges if e.dependency_type])
        logger.success(
            f"[{source_file}] Hybrid extraction complete. Graph has {len(graph.nodes)} "
            f"artifacts and {len(graph.edges)} total edges."
        )
        logger.info(
            f"[{source_file}] Edge breakdown: {reference_edges} reference-based, {dependency_edges} dependency-based."
        )

        return graph, bank, artifact_to_terms_map, latex_macros

    async def _infer_and_add_dependencies_mode_aware(
        self,
        graph: DocumentGraph,
        artifact_to_terms_map: Dict[str, List[str]],
        bank: DefinitionBank,
        dependency_mode: DependencyInferenceMode,
        cfg: DependencyInferenceConfig,
        on_dependency_edge: Optional[Callable[[Edge], Awaitable[None]]] = None,
    ) -> DocumentGraph:
        internal_nodes = [node for node in graph.nodes if not node.is_external]
        if len(internal_nodes) < 2:
            logger.info("Not enough internal nodes to infer dependencies. Skipping.")
            return graph

        # Choose mode if auto.
        selected_mode = dependency_mode
        reason = None
        tok_est = None
        if dependency_mode == "auto":
            selected_mode, reason, tok_est = choose_mode_auto(internal_nodes, cfg)

        logger.info(
            f"Dependency inference mode={selected_mode} (requested={dependency_mode})"
            + (f" | {reason}" if reason else "")
            + (f" | tok_est≈{tok_est}" if tok_est is not None else "")
        )

        if selected_mode == "pairwise":
            return await self._infer_and_add_dependencies_pairwise(
                graph=graph,
                artifact_to_terms_map=artifact_to_terms_map,
                bank=bank,
                cfg=cfg,
                on_dependency_edge=on_dependency_edge,
            )

        if selected_mode == "global":
            return await self._infer_and_add_dependencies_global(
                graph, internal_nodes, cfg, on_dependency_edge
            )

        if selected_mode == "hybrid":
            return await self._infer_and_add_dependencies_hybrid(
                graph=graph,
                internal_nodes=internal_nodes,
                cfg=cfg,
                on_dependency_edge=on_dependency_edge,
            )

        logger.warning(
            f"Unknown dependency_mode={selected_mode}. Falling back to pairwise."
        )
        return await self._infer_and_add_dependencies_pairwise(
            graph=graph,
            artifact_to_terms_map=artifact_to_terms_map,
            bank=bank,
            cfg=cfg,
            on_dependency_edge=on_dependency_edge,
        )

    async def _infer_and_add_dependencies_global(
        self,
        graph: DocumentGraph,
        internal_nodes: List,
        cfg: DependencyInferenceConfig,
        on_dependency_edge: Optional[Callable[[Edge], Awaitable[None]]] = None,
    ) -> DocumentGraph:
        logger.info(
            f"[global] Running one-shot dependency inference for N={len(internal_nodes)}"
        )
        result = await self.global_dependency_inferencer.ainfer_dependencies(
            internal_nodes, cfg
        )

        id_to_node = {n.id: n for n in internal_nodes}
        added = 0
        dropped = 0

        for e in result.edges:
            if e.source_id == e.target_id:
                dropped += 1
                continue
            if e.source_id not in id_to_node or e.target_id not in id_to_node:
                dropped += 1
                continue
            if graph.find_edge(e.source_id, e.target_id):
                continue

            new_edge = Edge(
                source_id=e.source_id,
                target_id=e.target_id,
                dependency_type=e.dependency_type,
                dependency=e.justification,
            )
            graph.add_edge(new_edge)
            if on_dependency_edge is not None:
                await on_dependency_edge(new_edge)
            added += 1

        logger.success(
            f"[global] Added {added} dependency edges (dropped_invalid={dropped})."
        )
        return graph

    async def _infer_and_add_dependencies_hybrid(
        self,
        graph: DocumentGraph,
        internal_nodes: List,
        cfg: DependencyInferenceConfig,
        on_dependency_edge: Optional[Callable[[Edge], Awaitable[None]]] = None,
    ) -> DocumentGraph:
        logger.info(
            f"[hybrid] Propose+verify dependency inference for N={len(internal_nodes)}"
        )

        proposal = await self.global_dependency_proposer.apropose(internal_nodes, cfg)

        # Filter + dedupe candidates from global proposer.
        id_to_node = {n.id: n for n in internal_nodes}
        seen: set[tuple[str, str]] = set()
        final_candidates: list[tuple[str, str]] = []

        for pe in proposal.edges:
            if pe.source_id == pe.target_id:
                continue
            if pe.source_id not in id_to_node or pe.target_id not in id_to_node:
                continue
            if graph.find_edge(pe.source_id, pe.target_id):
                continue
            key = (pe.source_id, pe.target_id)
            if key in seen:
                continue
            seen.add(key)
            final_candidates.append(key)

        num_candidates = len(final_candidates)

        if num_candidates == 0:
            logger.info("[stage=deps_hybrid] No candidates to verify.")
            return graph

        if num_candidates > cfg.max_total_pairs:
            logger.warning(
                f"[stage=deps_hybrid] candidate_pairs={num_candidates} exceeds "
                f"cap={cfg.max_total_pairs}; skipping hybrid dependency inference "
                "for this paper to avoid excessive LLM calls."
            )
            return graph

        logger.info(
            f"[stage=deps_hybrid] Verifying {num_candidates} candidate pairs from global proposer."
        )

        # Verify candidates with pairwise checker.
        tasks_with_context = []
        for source_id, target_id in final_candidates:
            source_node = id_to_node[source_id]
            target_node = id_to_node[target_id]
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
                    f"[hybrid] Error verifying pair ({source_id}, {target_id}): {result}"
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
                    or "Inferred by LLM based on global proposal.",
                )
                graph.add_edge(new_edge)
                if on_dependency_edge is not None:
                    await on_dependency_edge(new_edge)
                added_edges_count += 1

        logger.success(f"[hybrid] Added {added_edges_count} verified dependency edges.")
        return graph

    async def _enrich_artifact_content(
        self,
        graph: DocumentGraph,
        latex_content: str,
        on_enriched_node: Optional[Callable[[ArtifactNode], Awaitable[None]]] = None,
    ):
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
                if on_enriched_node is not None:
                    await on_enriched_node(node)

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

    async def _infer_and_add_dependencies_pairwise(
        self,
        graph: DocumentGraph,
        artifact_to_terms_map: Dict[str, List[str]],
        bank: DefinitionBank,
        cfg: DependencyInferenceConfig | None = None,
        on_dependency_edge: Optional[Callable[[Edge], Awaitable[None]]] = None,
    ) -> DocumentGraph:
        """
        Analyzes artifacts to infer dependencies efficiently using conceptual and
        term overlap to generate candidates.
        """
        # Default configuration when called directly from tests or legacy paths.
        cfg = cfg or DependencyInferenceConfig()

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

            # PERF: Prefetch definitions in batch to avoid many bank.find() calls
            # (each of which acquires the bank lock). This keeps footprint building
            # in pure Python after a single lock acquisition.
            all_terms = set().union(*artifact_to_terms_map.values())
            definitions = await bank.find_many(list(all_terms))
            canonical_term_to_deps = {
                bank._normalize_term(d.term): set(d.dependencies or [])
                for d in definitions
            }

            artifact_footprints = defaultdict(set)
            for artifact in internal_nodes:
                direct_terms = artifact_to_terms_map.get(artifact.id, [])
                artifact_footprints[artifact.id].update(direct_terms)

                for term in direct_terms:
                    deps = canonical_term_to_deps.get(bank._normalize_term(term))
                    if deps:
                        artifact_footprints[artifact.id].update(deps)

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

        num_candidates = len(final_candidate_pairs)

        if num_candidates == 0:
            logger.debug(
                "No promising candidate pairs found after filtering. Skipping LLM verification."
            )
            return graph

        if num_candidates > cfg.max_total_pairs:
            logger.warning(
                f"[stage=deps_pairwise] candidate_pairs={num_candidates} exceeds "
                f"cap={cfg.max_total_pairs}; skipping pairwise dependency "
                "inference for this paper to avoid excessive LLM calls."
            )
            return graph

        logger.info(
            f"[stage=deps_pairwise] Generated {num_candidates} candidate pairs for LLM verification."
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
                if on_dependency_edge is not None:
                    await on_dependency_edge(new_edge)
                added_edges_count += 1
                logger.debug(
                    f"Created new dependency edge: {source_node.id} -> {target_node.id} (Type: {result.dependency_type})"
                )

        logger.success(
            f"Added {added_edges_count} new dependency edges based on LLM verification."
        )
        return graph
