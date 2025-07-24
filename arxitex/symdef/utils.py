
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
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, indent=2))
    logger.success(f"Results saved successfully.")