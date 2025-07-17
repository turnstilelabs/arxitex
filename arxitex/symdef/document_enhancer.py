import argparse
from typing import List, Dict
import tempfile 
from loguru import logger
import json
import re
from pathlib import Path
import asyncio
import aiofiles

from arxitex.source_downloader import AsyncSourceDownloader
from arxitex.symdef.utils import Definition, ContextFinder
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.definition_builder.definition_builder import DefinitionBuilder
from arxitex.graph.utils import ArtifactNode, ArtifactType
from arxitex.symdef.utils import async_load_artifacts_from_json, async_load_latex_content, async_save_enhanced_artifacts

class DocumentEnhancer:
    def __init__(self, artifacts: List[ArtifactNode], latex_content: str):
        self.artifacts = sorted(artifacts, key=lambda a: a.position.line_start)
        self.latex_content = latex_content
        self.artifact_end_positions = self._calculate_end_positions()

        self.bank = DefinitionBank()
        #TODO remove llm_enhancer from here, use async version
        self.llm_enhancer = DefinitionBuilder()
        self.context_finder = ContextFinder()
        self.enhanced_artifacts = asyncio.Queue()

    def _calculate_end_positions(self) -> Dict[str, int]:
        """Pre-calculates the character offset of the end of each artifact."""
        positions = {}
        lines = self.latex_content.splitlines(keepends=True)
        for artifact in self.artifacts:
            char_pos = sum(len(line) for line in lines[:artifact.position.line_end])
            positions[artifact.id] = char_pos
        return positions

    def run(self):
        """Processes all artifacts sequentially to build the definition bank and create enhanced content."""
        for artifact in self.artifacts:
            self._process_artifact(artifact)
        return self.enhanced_artifacts
    
    async def run(self):
        """Processes all artifacts concurrently to build the definition bank and create enhanced content."""
        tasks = [self._process_artifact(artifact) for artifact in self.artifacts]
        await asyncio.gather(*tasks)

        results = {}
        while not self.enhanced_artifacts.empty():
            artifact_id, enhanced_text = self.enhanced_artifacts.get_nowait()
            results[artifact_id] = enhanced_text
        return results

    def _validate_definition_in_context(self, definition_text: str, context: str) -> bool:
        """
        Ensures that every sentence in the generated definition exists verbatim in the context.
        """
        if not definition_text:
            return True
        
        norm_context = " ".join(context.split())

        # Simple sentence splitting. For production, NLTK's sent_tokenize is more robust.
        generated_sentences = re.split(r'(?<=[.?!])\s+', definition_text.strip())
        
        if not generated_sentences:
            return True

        for sent in generated_sentences:
            if not sent: continue
            norm_sent = " ".join(sent.split())
            if norm_sent not in norm_context:
                logger.warning(f"Validation FAILED. Sentence not in context: '{norm_sent}'")
                return False
        
        logger.debug("LLM-generated definition passed validation.")
        return True
    
    def _process_artifact(self, artifact: ArtifactNode):
        """
        For a single artifact:
        1. Extracts terms.
        2. Defines any unknown terms by searching backwards.
        3. Creates the final self-contained version.
        """
        logger.info(f"--- Processing Artifact: {artifact.id} ({artifact.type.value}) ---")
        
        if artifact.type == ArtifactType.DEFINITION:
            logger.info(f"Artifact {artifact.id} is a definition. Extracting...")
            extracted_data = self.llm_enhancer.extract_definition(artifact.content)
            if extracted_data:
                new_def = Definition(
                    term=extracted_data.defined_term,
                    definition_text=extracted_data.definition_text,
                    aliases=extracted_data.aliases,
                    source_artifact_id=artifact.id
                )
                self.bank.register(new_def)
            else:
                logger.error(f"Failed to extract definition from artifact {artifact.id}.")
        
        # 1. Extract all relevant terms from the current artifact
        terms_in_artifact = self.llm_enhancer.extract_terms(artifact.content)
        
        # 2. For each term, ensure a definition exists in our bank
        for term in terms_in_artifact:
            if self.bank.find(term) is None:
                logger.warning(f"Term '{term}' is new in {artifact.id}. Synthesizing its definition...")
                
                # Search backwards from the end of the current artifact
                end_pos = self.artifact_end_positions[artifact.id]
                context_snippets = self.context_finder.find_prior_occurrences(term, self.latex_content, end_pos)
                
                if not context_snippets:
                    logger.error(f"No prior context found for '{term}'. Cannot define.")
                    continue
                
                # Check for a base definition to help the LLM (e.g., for 'abelian group', find 'group')
                base_definition = self.bank.find_best_base_definition(term)
                logger.debug(f"Base definition for '{term}': {base_definition.term if base_definition else 'None'}")
                synthesized_text = self.llm_enhancer.synthesize_definition(term, context_snippets, base_definition)
                
                #if synthesized_text and not self._validate_definition_in_context(synthesized_text, context_snippets):
                #    logger.error(f"LLM synthesis for '{term}' failed validation. Discarding definition.")
                #    synthesized_text = None
                    
                if synthesized_text:
                    logger.debug(f"Synthesized definition for '{term}': {synthesized_text}")
                    new_def = Definition(
                        term=term,
                        definition_text=synthesized_text,
                        source_artifact_id=f"synthesized_from_context_for_{artifact.id}",
                        dependencies=[base_definition.term] if base_definition else []
                    )
                    self.bank.register(new_def)
        
        # 3. Now that all terms are defined, create the enhanced artifact content
        definitions_needed = {t: self.bank.find(t) for t in terms_in_artifact if self.bank.find(t)}
        enhanced_text = self._create_enhanced_content(artifact, definitions_needed)
        self.enhanced_artifacts[artifact.id] = enhanced_text
        logger.success(f"Successfully enhanced artifact '{artifact.id}'.")
            
    async def _process_artifact(self, artifact: ArtifactNode):
        logger.info(f"--- Starting processing for Artifact: {artifact.id} ({artifact.type.value}) ---")
        
        llm_enhancer = DefinitionBuilder() 

        if artifact.type == ArtifactType.DEFINITION:
            logger.info(f"Artifact {artifact.id} is a definition. Extracting...")
            extracted_data = await llm_enhancer.aextract_definition(artifact.content)
            if extracted_data:
                new_def = Definition(
                    term=extracted_data.defined_term,
                    definition_text=extracted_data.definition_text,
                    aliases=extracted_data.aliases,
                    source_artifact_id=artifact.id
                )
                await self.bank.register(new_def)
            else:
                logger.error(f"Failed to extract definition from artifact {artifact.id}.")
        
        terms_in_artifact = await llm_enhancer.aextract_terms(artifact.content)
        
        for term in terms_in_artifact:
            if self.bank.find(term) is None:
                logger.warning(f"Term '{term}' is new in {artifact.id}. Synthesizing its definition...")
                
                end_pos = self.artifact_end_positions[artifact.id]
                context_snippets = self.context_finder.find_prior_occurrences(term, self.latex_content, end_pos)
                
                if not context_snippets:
                    logger.error(f"No prior context found for '{term}'. Cannot define.")
                    continue
                
                base_definition = self.bank.find_best_base_definition(term)
                logger.debug(f"Base definition for '{term}': {base_definition.term if base_definition else 'None'}")
                synthesized_text = await llm_enhancer.asynthesize_definition(term, context_snippets, base_definition)
                                    
                if synthesized_text:
                    logger.debug(f"Synthesized definition for '{term}': {synthesized_text}")
                    new_def = Definition(
                        term=term,
                        definition_text=synthesized_text,
                        source_artifact_id=f"synthesized_from_context_for_{artifact.id}",
                        dependencies=[base_definition.term] if base_definition else []
                    )
                    await self.bank.register(new_def)
        
        definitions_needed = {t: self.bank.find(t) for t in terms_in_artifact if self.bank.find(t)}
        enhanced_text = self._create_enhanced_content(artifact, definitions_needed)
        await self.enhanced_artifacts.put((artifact.id, enhanced_text))
        logger.success(f"Successfully finished processing artifact '{artifact.id}'.")
        
    def _create_enhanced_content(self, artifact: ArtifactNode, definitions: Dict[str, Definition]) -> str:
        """
        Builds the final string for the self-contained artifact by concatenating
        all necessary definitions, followed by the original artifact content.
        """
        if not definitions:
            return artifact.content

        definitions_list = [f"**{term}**: {definition.definition_text}" for term, definition in definitions.items()]
        definitions_block = "--- Prerequisite Definitions ---\n" + "\n\n".join(definitions_list)

        original_content_block = f"\n{artifact.content}"
        
        return f"{definitions_block}\n\n{original_content_block}"


async def main():
    parser = argparse.ArgumentParser(
        description="Enhance mathematical artifacts from a LaTeX paper to make them self-contained.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "json_input",
        type=Path,
        help="Path to the input JSON file containing the extracted artifacts (e.g., paper_artifacts.json)."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--latex-file",
        type=Path,
        help="Path to the full LaTeX source file (.tex) for context searching."
    )
    source_group.add_argument(
        "--arxiv-id",
        type=str,
        help="arXiv ID (e.g., '2305.12345') to download the source from."
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default="output/enhanced_artifacts.json",
        help="Path to save the output JSON file with the enhanced content."
    )

    parser.add_argument(
        "--bank-output-path",
        "-b",
        nargs='?',
        const="output/definition_bank.json",
        default=None,
        type=Path,
        help="Saves the final definition bank. If a path is given, saves there. "
            "If only the flag is present, saves to 'output/definition_bank.json'."
    )
    
    args = parser.parse_args()

    logger.info("Loading artifacts...")
    artifacts = await async_load_artifacts_from_json(args.json_input)
    latex_content = None

    if args.arxiv_id:
        logger.info(f"Downloading LaTeX source for arXiv ID: {args.arxiv_id}...")
        try:
            temp_dir_name = f"arxiv_{args.arxiv_id.replace('/', '_')}_"
            with tempfile.TemporaryDirectory(prefix=temp_dir_name) as temp_dir:
                temp_path = Path(temp_dir)
                async with AsyncSourceDownloader(cache_dir=temp_path) as downloader:
                    latex_content = await downloader.async_download_and_read_latex(args.arxiv_id)
        except Exception as e:
            logger.error(f"Failed to download or process arXiv source for '{args.arxiv_id}': {e}")
            return
    elif args.latex_file:
        logger.info(f"Loading LaTeX source from local file: {args.latex_file}...")
        latex_content = await async_load_latex_content(args.latex_file)
    
    if not latex_content:
        logger.error("Could not obtain LaTeX content. Exiting.")
        return

    # --- 2. Run the Enhancement Process (asynchronously) ---
    logger.info("Initializing document enhancer...")
    enhancer = DocumentEnhancer(artifacts=artifacts, latex_content=latex_content)
    
    logger.info("Starting artifact enhancement process. This may take some time...")
    enhanced_results = await enhancer.run()

    # --- 3. Save the Results (asynchronously) ---
    if enhanced_results:
        await async_save_enhanced_artifacts(enhanced_results, args.output_path)
    else:
        logger.warning("Enhancement process finished but produced no results.")

    if args.bank_output_path:
        logger.info(f"Saving definition bank to {args.bank_output_path}...")
        try:
            args.bank_output_path.parent.mkdir(parents=True, exist_ok=True)            
            bank_dict = enhancer.bank.to_dict()
            json_string = json.dumps(bank_dict, indent=2, ensure_ascii=False)
            async with aiofiles.open(args.bank_output_path, "w", encoding="utf-8") as f:
                await f.write(json_string)
            logger.info(f"Successfully saved definition bank to {args.bank_output_path}")
        except Exception as e:
            logger.error(f"Could not save the definition bank: {e}")

if __name__ == "__main__":
    asyncio.run(main())