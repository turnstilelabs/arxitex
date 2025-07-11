
from dataclasses import dataclass, field
from typing import List
from loguru import logger
import re


@dataclass
class Definition:
    """Represents a single, resolved definition for a term."""
    term: str
    definition_text: str
    source_artifact_id: str  # ID of the artifact where this was defined/synthesized
    aliases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list) # e.g., "abelian group" depends on "group"


class ContextFinder:
    def find_prior_occurrences(self, term: str, full_text: str, end_char_pos: int) -> str:
        """Finds snippets of prior occurrences of a term using regex."""
        text_to_search = full_text[:end_char_pos]
        # Regex to find the term as a whole word, ignoring case for notions
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