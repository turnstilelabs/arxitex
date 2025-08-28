
from dataclasses import dataclass, field
from typing import List
from loguru import logger
import re

from pathlib import Path
import json
import sys
import aiofiles 

from pydantic import ValidationError, TypeAdapter 
from arxitex.extractor.utils import ArtifactNode


@dataclass
class Definition:
    """Represents a single, resolved definition for a term."""
    term: str
    definition_text: str
    source_artifact_id: str
    aliases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list) # e.g., "abelian group" depends on "group"


class ContextFinder:
    def find_prior_occurrences(self, term: str, full_text: str, end_char_pos: int) -> str:
        """Finds snippets of prior occurrences of a term using regex."""
        text_to_search = full_text[:end_char_pos]
        # This is a simple regex; it can be improved for symbols like `\F`
        try:
            # For symbols, use exact match. For notions, case-insensitive.
            if '$' in term or '\\' in term:
                pattern = re.escape(term)
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
            
            snippets = []
            for match in re.finditer(pattern, text_to_search, re.IGNORECASE):
                start, end = match.span()
                line_num = text_to_search.count('\n', 0, start) + 1
                snippet_start = max(0, start - 100)
                snippet_end = min(len(text_to_search), end + 100)
                snippet = text_to_search[snippet_start:snippet_end].replace('\n', ' ')
                snippets.append(f"Line ~{line_num}: ...{snippet}...")
            
            return "\n".join(snippets)
        except re.error as e:
            logger.error(f"Regex error for term '{term}': {e}")
            return ""
        
    def find_context_around_first_occurrence(
        self, 
        term: str, 
        text_to_search: str
    ) -> str:
        """
        Finds the first occurrence of a term and returns the full paragraph containing it.
        """
        try:
            # Step 1: Build the single, safest, most appropriate pattern.
            escaped_term = re.escape(term)
            pattern = ""

            # For simple, purely alphabetic terms ('f', 'G', 'phi'), we must use
            # strict boundaries to prevent false positives.
            if term.isalpha():
                # A term must be preceded by:
                #   (?:^|\s|[\(\[\{,=+\-*/<>,])
                #   - ^           : the start of the string
                #   - \s          : any whitespace character
                #   - [...]       : a set of common opening delimiters and operators
                # It must be followed by:
                #   (?=[\s\)\]\},.=+\-*/<>,]|$)
                #   - \s, delimiters, or operators
                #   - $           : the end of the string
                
                prefix = r'(?:^|\s|[\(\[\{,=+\-*/<>,])'
                suffix = r'(?=[\s\)\]\},.=+\-*/<>,]|$)'                
                pattern = f'{prefix}({escaped_term}){suffix}'
            else:
                # For any term containing non-alphabetic characters
                # a literal search is the only reliable method.
                pattern = escaped_term

            logger.debug(f"Searching for term '{term}' with pattern: {pattern}")
            
            # Step 2: Find the first match in the original text.
            first_match = next(re.finditer(pattern, text_to_search), None)
            
            if not first_match:
                logger.warning(f"Term '{term}' not found in the preceding text.")
                return ""

        except re.error as e:
            logger.error(f"Regex error for term '{term}' with pattern '{pattern}': {e}")
            return ""

        # Step 3: Extract the paragraph containing the match.
        match_start_pos = first_match.start(1) if term.isalpha() else first_match.start()

        para_start_pos = text_to_search.rfind('\n\n', 0, match_start_pos)
        para_start_pos = 0 if para_start_pos == -1 else para_start_pos + 2 

        para_end_pos = text_to_search.find('\n\n', match_start_pos)
        para_end_pos = len(text_to_search) if para_end_pos == -1 else para_end_pos

        definitional_paragraph = text_to_search[para_start_pos:para_end_pos].strip()
        
        return definitional_paragraph
        
def clean_latex_for_llm(text: str) -> str:
    """
    Removes common LaTeX structural and metadata commands to clean up context for an LLM.

    This function removes commands that define document structure but not content,
    such as environments, labels, and sectioning commands.

    Examples:
        - '\\begin{claim}' -> ''
        - '\\label{f_min}' -> ''
        - '\\end{theorem}' -> ''
        - '\\section*{Introduction}' -> 'Introduction'
    """
    if not text:
        return ""

    # Rule 1: Remove \begin{...} and \end{...} commands
    cleaned_text = re.sub(r'\\(begin|end)\{[a-zA-Z0-9_*]+\}\s*', '', text)

    # Rule 2: Remove \label{...} commands
    cleaned_text = re.sub(r'\\label\{[^\}]+\}\s*', '', cleaned_text)

    # Rule 3: Remove common no-argument commands like \item or \centering
    cleaned_text = re.sub(r'\\(item|centering|newpage|clearpage)\b\s*', '', cleaned_text)

    # Rule 4: Handle sectioning commands by keeping their title but removing the command itself.
    cleaned_text = re.sub(r'\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]+)\}', r'\2', cleaned_text)

    # Rule 5: Collapse multiple blank lines into a single one for readability.
    cleaned_text = re.sub(r'(\n\s*){3,}', '\n\n', cleaned_text).strip()

    return cleaned_text

def load_artifacts_from_json(file_path: Path) -> List[ArtifactNode]:
    """Loads artifacts from a JSON file and validates them."""
    if not file_path.exists():
        logger.error(f"Artifact JSON file not found at: {file_path}")
        sys.exit(1)
        
    logger.info(f"Loading artifacts from {file_path}...")
    try:
        # Use Pydantic's TypeAdapter for robust list validation
        ArtifactListAdapter = TypeAdapter(List[ArtifactNode])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            artifacts = ArtifactListAdapter.validate_python(data.get("nodes", []))
        logger.success(f"Successfully loaded and validated {len(artifacts)} artifacts.")
        return artifacts
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or validate artifacts from {file_path}: {e}")
        sys.exit(1)

def load_latex_content(file_path: Path) -> str:
    """Loads the full LaTeX source code from a file."""
    if not file_path.exists():
        logger.error(f"LaTeX source file not found at: {file_path}")
        sys.exit(1)
    
    logger.info(f"Loading LaTeX source from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.success("LaTeX source loaded.")
    return content

def save_enhanced_artifacts(results: dict, output_path: Path):
    """Saves the enhanced artifact data to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving enhanced artifacts to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved successfully.")

async def async_load_artifacts_from_json(file_path: Path) -> List["ArtifactNode"]:
    if not file_path.exists():
        logger.error(f"Artifact JSON file not found at: {file_path}")
        sys.exit(1)
        
    logger.info(f"Loading artifacts from {file_path}...")
    try:
        ArtifactListAdapter = TypeAdapter(List["ArtifactNode"])
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            artifacts = ArtifactListAdapter.validate_python(data.get("nodes", []))
        logger.success(f"Successfully loaded and validated {len(artifacts)} artifacts.")
        return artifacts
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or validate artifacts from {file_path}: {e}")
        sys.exit(1)

async def async_load_latex_content(file_path: Path) -> str:
    if not file_path.exists():
        logger.error(f"LaTeX source file not found at: {file_path}")
        sys.exit(1)
    
    logger.info(f"Loading LaTeX source from {file_path}...")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
    logger.success("LaTeX source loaded.")
    return content

async def async_save_enhanced_artifacts(results: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving enhanced artifacts to {output_path}...")
    enhanced_artifacts = results.get("artifacts", {})
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(enhanced_artifacts, indent=2))
    logger.success(f"Results saved successfully.")