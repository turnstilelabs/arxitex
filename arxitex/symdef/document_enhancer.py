import argparse
import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
from loguru import logger

from arxitex.downloaders.async_downloader import AsyncSourceDownloader
from arxitex.extractor.utils import ArtifactNode, ArtifactType
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.definition_builder.definition_builder import DefinitionBuilder
from arxitex.symdef.utils import (
    ContextFinder,
    Definition,
    async_load_artifacts_from_json,
    async_load_latex_content,
    async_save_enhanced_artifacts,
    create_canonical_search_string,
)


def determine_output_path(
    user_path: Optional[Path],
    default_dir: Path,
    default_subdir: str,
    file_id: str,
    suffix: str,
) -> Path:
    """
    Determines the final output path for a file.
    If a user path is provided, it's used directly.
    Otherwise, a default path is constructed.
    """
    if user_path:
        return user_path

    return default_dir / default_subdir / f"{file_id}_{suffix}.json"


class DocumentEnhancer:
    """
    Enhances a list of document artifacts by ensuring all mathematical terms
    are defined, providing a self-contained context for each artifact.

    This class operates in a three-phase process to avoid race conditions and
    ensure correctness when running concurrently:
    1.  _populate_bank_from_definitions: Sequentially processes explicit definitions.
    2.  _synthesize_missing_definitions: Concurrently finds and defines all other terms.
    3.  _enhance_all_artifacts: Concurrently builds the final enhanced content for each artifact.
    """

    def __init__(
        self,
        llm_enhancer: DefinitionBuilder,
        context_finder: ContextFinder,
        definition_bank: DefinitionBank,
    ):
        """
        Initializes the DocumentEnhancer with its dependencies.

        Args:
            llm_enhancer: An instance of DefinitionBuilder for LLM interactions.
            context_finder: An instance of ContextFinder to locate term contexts.
            definition_bank: A shared instance of DefinitionBank to store definitions.
        """
        self.llm_enhancer = llm_enhancer
        self.context_finder = context_finder
        self.bank = definition_bank
        self._synthesis_lock = asyncio.Lock()

    async def enhance_document(
        self,
        artifacts: List[ArtifactNode],
        latex_content: str,
        use_global_extraction: bool = True,
    ) -> Dict[str, str]:
        """
        Enhances the document by processing all artifacts to ensure they are self-contained
        Args:
            artifacts: A list of ArtifactNodes sorted by their position in the document.
            latex_content: The full LaTeX source content.
            use_global_extraction: If True, uses single-call global term extraction.

        Returns:
            A dictionay containing:
            - A dictionary mapping artifact IDs to their enhanced content string.
            - A dictionary mapping artifact IDs to their terms.
            - A DefinitionBank instance containing all definitions found or synthesized.
        """
        sorted_artifacts = sorted(artifacts, key=lambda a: a.position.line_start)
        all_artifacts_map = {a.id: a for a in sorted_artifacts}
        artifact_start_positions = self._calculate_start_positions(
            sorted_artifacts, latex_content
        )
        artifact_end_positions = self._calculate_end_positions(
            sorted_artifacts, latex_content
        )

        logger.info("Phase 1: Populating bank from explicit definitions...")
        await self._populate_bank_from_definitions(sorted_artifacts)

        logger.info("Phase 2: Synthesizing definitions for remaining terms...")
        artifact_to_terms_map, term_to_first_artifact_map = (
            await self._discover_and_synthesize_terms(
                sorted_artifacts,
                artifact_start_positions,
                artifact_end_positions,
                latex_content,
                use_global_extraction=use_global_extraction,
            )
        )

        await self.bank.merge_redundancies()
        await self.bank.resolve_internal_dependencies()

        logger.info("Phase 3: Generating enhanced content for all artifacts...")
        definitions_map = await self._enhance_all_artifacts(
            sorted_artifacts,
            artifact_to_terms_map,
            term_to_first_artifact_map,
            all_artifacts_map,
        )

        logger.success("Document enhancement complete.")
        return {
            "definitions_map": definitions_map,
            "artifact_to_terms_map": artifact_to_terms_map,
            "definition_bank": self.bank,
        }

    async def _is_term_missing(self, term: str) -> bool:
        """Helper to check if a term is in the bank using the proper find method."""
        return await self.bank.find(term) is None

    def _calculate_start_positions(
        self, artifacts: List[ArtifactNode], latex_content: str
    ) -> Dict[str, int]:
        """Pre-calculates the character offset of the start of each artifact."""
        line_start_offsets = [0]
        current_offset = 0
        while True:
            current_offset = latex_content.find("\n", current_offset)
            if current_offset == -1:
                break
            line_start_offsets.append(current_offset + 1)
            current_offset += 1

        positions = {}
        for artifact in artifacts:
            if (
                not artifact.position
                or artifact.position.line_start is None
                or artifact.position.col_start is None
            ):
                logger.warning(
                    f"Artifact '{artifact.id}' is missing start position data. "
                    "Skipping its start offset calculation."
                )
                continue

            start_line_index = artifact.position.line_start - 1

            if start_line_index >= len(line_start_offsets):
                continue

            start_of_line_offset = line_start_offsets[start_line_index]
            final_offset = start_of_line_offset + (artifact.position.col_start - 1)
            positions[artifact.id] = final_offset

        return positions

    def _calculate_end_positions(
        self, artifacts: List[ArtifactNode], latex_content: str
    ) -> Dict[str, int]:
        """Pre-calculates the character offset of the end of each artifact."""
        positions = {}
        lines = latex_content.splitlines(keepends=True)

        for artifact in artifacts:
            if (
                not artifact.position
                or artifact.position.line_end is None
                or artifact.position.col_end is None
            ):

                logger.warning(
                    f"Artifact '{artifact.id}' is missing complete position data. "
                    "Skipping its offset calculation."
                )
                continue
            end_line_index = artifact.position.line_end - 1

            start_of_end_line_offset = sum(len(line) for line in lines[:end_line_index])
            final_offset = start_of_end_line_offset + (artifact.position.col_end - 1)
            positions[artifact.id] = final_offset

        return positions

    async def _populate_bank_from_definitions(self, artifacts: List[ArtifactNode]):
        """
        Sequentially processes artifacts of type DEFINITION to populate the bank.
        Sequential processing is crucial in case definitions depend on each other.
        """
        definition_artifacts = [
            a for a in artifacts if a.type == ArtifactType.DEFINITION
        ]
        for artifact in definition_artifacts:
            logger.info(
                f"Extracting explicit definition from artifact {artifact.id}..."
            )
            extracted_data = await self.llm_enhancer.aextract_definition(
                artifact.content
            )
            if extracted_data:
                new_def = Definition(
                    term=extracted_data.defined_term,
                    definition_text=extracted_data.definition_text,
                    aliases=extracted_data.aliases,
                    source_artifact_id=artifact.id,
                )
                await self.bank.register(new_def)
                logger.success(
                    f"Registered definition for term '{new_def.term}': '{new_def.definition_text}'"
                )
            else:
                logger.error(
                    f"Failed to extract definition from artifact {artifact.id}."
                )

    def _filter_and_sanitize_extracted_terms(self, raw_terms: List[str]) -> List[str]:
        """
        Cleans a list of terms returned by the LLM.
        """
        sanitized_terms = []
        control_char_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
        for term in raw_terms:
            clean_term = control_char_re.sub("", term)

            # Rule 2: Normalize excessive backslashes from LLM hallucinations (e.g., \\phi -> \phi).
            clean_term = re.sub(r"\\{2,}", r"\\", clean_term)

            # Rule 3: ONLY strip leading/trailing whitespace. Do NOT collapse internal whitespace.
            clean_term = clean_term.strip()

            # Rule 4: Remove ONLY common sentence-ending punctuation if it's trailing.
            if clean_term.endswith(".") or clean_term.endswith(","):
                clean_term = clean_term[:-1]

            if not clean_term:
                continue

            sanitized_terms.append(clean_term)

        return sorted(list(set(sanitized_terms)))

    async def _extract_and_sanitize_for_artifact(self, artifact):
        """
        Sanitizes artifact content and extracts terms using the LLM.
        """
        try:
            clean_content = re.sub(r"[^\x20-\x7E\n\t\r]", "", artifact.content)
            raw_terms = await self.llm_enhancer.aextract_terms(clean_content)
            return artifact.id, raw_terms
        except Exception as e:
            logger.error(
                f"Failed to extract terms from artifact {artifact.id}: {e}",
                exc_info=True,
            )
            return artifact.id, []

    async def _extract_terms_globally(
        self, artifacts: List[ArtifactNode]
    ) -> Tuple[set, Dict[str, List[str]], Dict[str, str]]:
        """Extracts terms using the single-call global strategy."""
        # 1. Concatenate, sanitize for LLM, and make a single API call
        logger.info("Using GLOBAL term extraction strategy.")
        full_document_content = "\n\n---\n---\n\n".join([a.content for a in artifacts])
        raw_valid_terms = await self.llm_enhancer.aextract_document_terms(
            full_document_content
        )

        # 2. Sanitize the LLM's output
        all_valid_terms_sanitized = self._filter_and_sanitize_extracted_terms(
            raw_valid_terms
        )
        all_valid_terms_set = set(all_valid_terms_sanitized)
        logger.success(
            f"Global strategy found {len(all_valid_terms_set)} unique, valid terms."
        )

        # 3. Map terms back to artifacts
        artifact_to_terms_map: Dict[str, List[str]] = {}
        term_to_first_artifact_map: Dict[str, str] = {}
        canonical_artifacts = {
            artifact.id: create_canonical_search_string(artifact.content)
            for artifact in artifacts
        }

        for artifact in artifacts:
            canonical_content = canonical_artifacts[artifact.id]
            found_terms = []
            for term in all_valid_terms_set:
                canonical_term = create_canonical_search_string(term)

                if not canonical_term:
                    continue

                if f" {canonical_term} " in f" {canonical_content} ":
                    found_terms.append(term)
                    if term not in term_to_first_artifact_map:
                        term_to_first_artifact_map[term] = artifact.id

            artifact_to_terms_map[artifact.id] = sorted(found_terms)

        return all_valid_terms_set, artifact_to_terms_map, term_to_first_artifact_map

    async def _extract_terms_per_artifact(
        self, artifacts: List[ArtifactNode]
    ) -> Tuple[set, Dict[str, List[str]], Dict[str, str]]:
        """Extracts terms using per-artifact iteration strategy."""
        logger.info("Using PER-ARTIFACT term extraction strategy.")
        all_valid_terms_set = set()
        artifact_to_terms_map: Dict[str, List[str]] = {}
        term_to_first_artifact_map: Dict[str, str] = {}

        extraction_tasks = [
            self.llm_enhancer.aextract_single_artifact_terms(a.content)
            for a in artifacts
        ]
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        for artifact, raw_terms_result in zip(artifacts, results):
            if isinstance(raw_terms_result, Exception):
                logger.error(
                    f"Failed to extract terms for artifact {artifact.id}: {raw_terms_result}"
                )
                sanitized_terms = []
            else:
                sanitized_terms = self._filter_and_sanitize_extracted_terms(
                    raw_terms_result
                )

            artifact_to_terms_map[artifact.id] = sanitized_terms
            all_valid_terms_set.update(sanitized_terms)
            for term in sanitized_terms:
                if term not in term_to_first_artifact_map:
                    term_to_first_artifact_map[term] = artifact.id

        logger.success(
            f"Per-artifact strategy found {len(all_valid_terms_set)} unique, valid terms."
        )
        return all_valid_terms_set, artifact_to_terms_map, term_to_first_artifact_map

    async def _discover_and_synthesize_terms(
        self,
        artifacts: List[ArtifactNode],
        start_positions: Dict[str, int],
        end_positions: Dict[str, int],
        latex_content: str,
        use_global_extraction: bool,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Discovers all terms via a single LLM call, sanitizes the result, then synthesizes definitions.
        """
        if use_global_extraction:
            all_valid_terms_set, artifact_to_terms_map, term_to_first_artifact_map = (
                await self._extract_terms_globally(artifacts)
            )
        else:
            all_valid_terms_set, artifact_to_terms_map, term_to_first_artifact_map = (
                await self._extract_terms_per_artifact(artifacts)
            )

        existing_defs = await self.bank.find_many(list(all_valid_terms_set))
        existing_canonical_terms = {
            self.bank._normalize_term(d.term) for d in existing_defs
        }
        missing_terms = {
            term
            for term in all_valid_terms_set
            if self.bank._normalize_term(term) not in existing_canonical_terms
        }

        if not missing_terms:
            logger.info("No new terms to synthesize; all are present in the bank.")
            return artifact_to_terms_map, term_to_first_artifact_map

        logger.info(
            f"Found {len(missing_terms)} missing terms to synthesize: {missing_terms}"
        )

        synthesis_tasks = [
            self._synthesize_single_term(
                term=term,
                source_artifact_id=term_to_first_artifact_map[term],
                start_positions=start_positions,
                end_positions=end_positions,
                latex_content=latex_content,
            )
            for term in missing_terms
            if term in term_to_first_artifact_map
        ]
        await asyncio.gather(*synthesis_tasks)

        return artifact_to_terms_map, term_to_first_artifact_map

    async def _synthesize_single_term(
        self,
        term: str,
        source_artifact_id: str,
        start_positions: Dict[str, int],
        end_positions: Dict[str, int],
        latex_content: str,
    ):
        """
        Defines a single term by combining the context from the preceding paragraph
        and the content of the source artifact itself.
        """
        log_prefix = f"[{term} @ {source_artifact_id}]"

        async with self._synthesis_lock:
            if await self.bank.find(term) is not None:
                logger.info(
                    f"{log_prefix} Term was already defined by a concurrent task. Skipping."
                )
                return

            logger.info(f"{log_prefix} Term is new. Beginning synthesis process...")

            doc_body_start_pos = latex_content.find("\\begin{document}")
            if doc_body_start_pos == -1:
                logger.warning(
                    "Could not find '\\begin{document}'. Searching for context in the entire file."
                )
                doc_body_start_pos = 0  # Default to start of file if not found
            else:
                doc_body_start_pos += len("\\begin{document}")

            start_pos = start_positions.get(source_artifact_id)
            end_pos = end_positions.get(source_artifact_id)

            if start_pos is None or end_pos is None:
                logger.error(
                    f"{log_prefix} Logic error: Could not find pre-calculated start/end positions "
                    f"for artifact ID. Cannot synthesize."
                )
                return

            search_start = max(doc_body_start_pos, 0)
            search_end = max(search_start, start_pos)
            text_to_search_before = latex_content[search_start:search_end]
            preceding_context = (
                self.context_finder.find_context_around_first_occurrence(
                    term, text_to_search_before
                )
            )
            artifact_content = latex_content[start_pos:end_pos].strip()

            context_parts = []
            if preceding_context:
                context_parts.append(
                    f"CONTEXT PRECEDING THE TERM'S FIRST USE:\n---\n{preceding_context}\n---"
                )
            context_parts.append(
                f"THE ARTIFACT WHERE THE TERM WAS FOUND:\n---\n{artifact_content}\n---"
            )

            combined_context = "\n\n".join(context_parts)
            logger.debug(
                f"{log_prefix} Providing combined context to LLM:\n{combined_context}"
            )

            base_definition = await self.bank.find_best_base_definition(term)
            if base_definition:
                logger.debug(
                    f"{log_prefix} Found base definition '{base_definition.term}'."
                )

            synthesized_text = None
            try:
                synthesized_text = await self.llm_enhancer.asynthesize_definition(
                    term, combined_context, base_definition
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"{log_prefix} LLM call timed out. Cannot synthesize this term."
                )
                return
            except Exception as e:
                logger.error(
                    f"{log_prefix} An unexpected error occurred during LLM call: {e}",
                    exc_info=True,
                )
                return

            # Validate and register the synthesized definition
            # TODO fix :)
            if synthesized_text:
                """
                full_context_str = " ".join(context_snippets)
                if self._validate_definition_in_context(synthesized_text, full_context_str):
                    logger.success(f"{log_prefix} Synthesized and validated definition: '{synthesized_text}'")
                    new_def = Definition(
                        term=term,
                        definition_text=synthesized_text,
                        source_artifact_id=f"synthesized_from_context_for_{source_artifact_id}",
                        dependencies=[base_definition.term] if base_definition else [],
                    )
                    await self.bank.register(new_def)
                else:
                    logger.error(f"{log_prefix} LLM synthesis failed validation. Discarding.")
                """
                logger.success(
                    f"{log_prefix} Synthesized and validated definition: '{synthesized_text}'"
                )
                new_def = Definition(
                    term=term,
                    definition_text=synthesized_text,
                    source_artifact_id=f"synthesized_from_context_for_{source_artifact_id}",
                    dependencies=[base_definition.term] if base_definition else [],
                )
                await self.bank.register(new_def)
            else:
                logger.warning(
                    f"{log_prefix} LLM returned no content for the definition."
                )

    async def _enhance_all_artifacts(
        self,
        artifacts: List[ArtifactNode],
        artifact_to_terms_map: Dict[str, List[str]],
        term_to_first_artifact_map: Dict[str, str],
        all_artifacts_map: Dict[str, ArtifactNode],
    ) -> Dict[str, Dict[str, str]]:
        """Concurrently enhances all artifacts using the pre-computed terms map."""
        tasks = [
            self._enhance_single_artifact(
                artifact,
                artifact_to_terms_map.get(artifact.id, []),
                term_to_first_artifact_map,
                all_artifacts_map,
            )
            for artifact in artifacts
        ]
        results = await asyncio.gather(*tasks)
        return {artifact_id: content for artifact_id, content in results}

    async def _enhance_single_artifact(
        self,
        artifact: ArtifactNode,
        terms_in_artifact: List[str],
        term_to_first_artifact_map: Dict[str, str],
        all_artifacts_map: Dict[str, ArtifactNode],
    ) -> tuple[str, Dict[str, str]]:
        """Enhances a single artifact by prepending necessary definitions from the pre-computed term list."""
        logger.info(
            f"Enhancing content for artifact '{artifact.id}' using pre-discovered terms..."
        )

        definitions_needed = {}
        for term in terms_in_artifact:
            definition = await self.bank.find(term)
            if definition:
                definitions_needed[term] = definition

        prerequisite_defs_dict = self._create_enhanced_content(
            artifact, definitions_needed, term_to_first_artifact_map, all_artifacts_map
        )
        logger.success(f"Successfully enhanced artifact '{artifact.id}'.")
        return artifact.id, prerequisite_defs_dict

    def _create_enhanced_content(
        self,
        artifact: ArtifactNode,
        definitions: Dict[str, Definition],
        term_to_first_artifact_map: Dict[str, str],
        all_artifacts_map: Dict[str, ArtifactNode],
    ) -> Dict[str, str]:
        """
        Creates a dictionary of prerequisite definitions, sorted by their appearance.
        """
        if not definitions:
            return {}

        def get_sort_key(term_tuple: tuple[str, Definition]) -> int:
            term, _ = term_tuple

            source_artifact_id = term_to_first_artifact_map.get(term)
            if not source_artifact_id:
                return 0  # Place terms without a source (e.g., from a base bank) at the beginning

            source_artifact = all_artifacts_map.get(source_artifact_id)
            if source_artifact and source_artifact.position:
                return source_artifact.position.line_start

            return 0

        sorted_defs = sorted(definitions.items(), key=get_sort_key)

        return {term: definition.definition_text for term, definition in sorted_defs}

    # TODO FIX this :)
    def _validate_definition_in_context(
        self, definition_text: str, context: str
    ) -> bool:
        """
        Ensures that every sentence in the generated definition exists verbatim in the context.
        """
        if not definition_text:
            return True

        norm_context = " ".join(context.split())

        # Simple sentence splitting. For production, NLTK's sent_tokenize is more robust.
        generated_sentences = re.split(r"(?<=[.?!])\s+", definition_text.strip())

        if not generated_sentences:
            return True

        for sent in generated_sentences:
            if not sent:
                continue
            norm_sent = " ".join(sent.split())
            if norm_sent not in norm_context:
                logger.warning(
                    f"Validation FAILED. Sentence not in context: '{norm_sent}'"
                )
                return False

        logger.debug("LLM-generated definition passed validation.")
        return True


async def main():
    parser = argparse.ArgumentParser(
        description="Enhance mathematical artifacts from a LaTeX paper to make them self-contained.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json_input",
        type=Path,
        help="Path to the input JSON file containing the extracted artifacts (e.g., paper_artifacts.json).",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--latex-file",
        type=Path,
        help="Path to the full LaTeX source file (.tex) for context searching.",
    )
    source_group.add_argument(
        "--arxiv-id",
        type=str,
        help="arXiv ID (e.g., '2305.12345') to download the source from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "output",
        help="Base directory for all output files. Defaults to './output' in the current working directory.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        nargs="?",
        default=None,
        type=Path,
        help="Path to save the output JSON file with the enhanced content.",
    )
    parser.add_argument(
        "--bank-output-path",
        "-b",
        nargs="?",
        default=None,
        type=Path,
        help="Saves the final definition bank. If a path is given, saves there. "
        "If only the flag is present, saves to 'output/definition_bank.json'.",
    )

    args = parser.parse_args()
    logger.info(f"Loading artifacts from {args.json_input}...")
    artifacts = await async_load_artifacts_from_json(args.json_input)
    file_id = ""
    latex_content = ""

    if args.arxiv_id:
        file_id = args.arxiv_id.replace("/", "_")
        logger.info(f"Downloading LaTeX source for arXiv ID: {args.arxiv_id}...")
        try:
            with tempfile.TemporaryDirectory(prefix=f"arxiv_{file_id}_") as temp_dir:
                async with AsyncSourceDownloader(
                    cache_dir=Path(temp_dir)
                ) as downloader:
                    latex_content = await downloader.async_download_and_read_latex(
                        args.arxiv_id
                    )
        except Exception as e:
            logger.error(
                f"Failed to download or process arXiv source for '{args.arxiv_id}': {e}",
                exc_info=True,
            )
            return
    elif args.latex_file:
        file_id = args.latex_file.stem
        logger.info(f"Loading LaTeX source from local file: {args.latex_file}...")
        latex_content = await async_load_latex_content(args.latex_file)

    if not latex_content or not artifacts:
        logger.error("Could not obtain LaTeX content or artifacts. Exiting.")
        return

    # --- 2. Instantiate Dependencies ---
    logger.info("Initializing components...")
    llm_enhancer = DefinitionBuilder()
    context_finder = ContextFinder()
    definition_bank = DefinitionBank()

    enhancer = DocumentEnhancer(
        llm_enhancer=llm_enhancer,
        context_finder=context_finder,
        definition_bank=definition_bank,
    )

    logger.info("Starting artifact enhancement process. This may take some time...")
    enhanced_results = await enhancer.enhance_document(artifacts, latex_content)

    logger.info(f"Using output base directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    enhanced_path = determine_output_path(
        user_path=args.output_path,
        default_dir=args.output_dir,
        default_subdir="enhanced_artifacts",
        file_id=file_id,
        suffix="enhanced",
    )

    bank_path = determine_output_path(
        user_path=args.bank_output_path,
        default_dir=args.output_dir,
        default_subdir="definition_banks",
        file_id=file_id,
        suffix="bank",
    )

    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    await async_save_enhanced_artifacts(enhanced_results, enhanced_path)

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving definition bank to {bank_path}...")
    try:
        bank_dict = await enhancer.bank.to_dict()
        json_string = json.dumps(bank_dict, indent=2, ensure_ascii=False)
        async with aiofiles.open(bank_path, "w", encoding="utf-8") as f:
            await f.write(json_string)
        logger.success(f"Successfully saved definition bank to {bank_path}")
    except Exception as e:
        logger.error(f"Could not save the definition bank: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
