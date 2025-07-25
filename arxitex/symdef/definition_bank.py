
from typing import Any, List, Dict, Optional
from loguru import logger
import asyncio
from arxitex.symdef.utils import Definition

class DefinitionBank:
    """The 'working memory' holding all definitions found so far."""
    def __init__(self):
        self._definitions: Dict[str, Definition] = {}
        self._alias_map: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    def _normalize_term(self, term: str) -> str:
        """
        Converts a term into its canonical string representation for use as a key.
        This is the most critical function for preventing redundancy.
        - Strips whitespace, math delimiters ($...$), and braces ({...}).
        - Removes leading backslashes (e.g., from \varphi).
        - Preserves case for single-character terms (e.g., 'f' vs 'F').
        - Converts multi-character terms to lowercase for consistency.
        """
        canonical = term.strip()
        stripped = True
        while stripped:
            stripped = False
            if canonical.startswith('$') and canonical.endswith('$') and len(canonical) > 1:
                canonical = canonical[1:-1].strip(); stripped = True
            if canonical.startswith('{') and canonical.endswith('}'):
                canonical = canonical[1:-1]; stripped = True
            if canonical.startswith('\\(') and canonical.endswith('\\)'):
                canonical = canonical[2:-2].strip(); stripped = True

        if canonical.startswith('\\'):
             core_term = canonical[1:]
        else:
             core_term = canonical

        if len(core_term) <5:
            return core_term  # Preserve case for 'f_23', 'F', etc.
        else:
            return core_term.lower() # Lowercase 'varphi', 'f_1', 'union-closed', etc.

    async def register(self, definition: Definition):
        """Adds or updates a definition, ensuring task-safe access."""
        async with self._lock:
            self._register_unlocked(definition)

    async def find(self, term: str) -> Optional[Definition]:
        """Finds a definition by its canonical form, ensuring task-safe access."""
        async with self._lock:
            return self._find_unlocked(term)
        
    async def find_many(self, terms: List[str]) -> List[Definition]:
        """
        Finds all definitions for a given list of terms in a single, efficient operation.
        """
        async with self._lock:
            return self._find_many_unlocked(terms)

    async def find_best_base_definition(self, term: str) -> Optional[Definition]:
        """Finds the best base definition, ensuring task-safe access."""
        async with self._lock:
            return self._find_best_base_definition_unlocked(term)

    async def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Exports a task-safe snapshot of the bank's definitions."""
        async with self._lock:
            return self._to_dict_unlocked()

    def _register_unlocked(self, definition: Definition):
        """The actual registration logic. Assumes lock is already held."""
        canonical_term = self._normalize_term(definition.term)
        if canonical_term in self._definitions:
            logger.debug(f"Overwriting definition for canonical term '{canonical_term}'.")
        logger.info(f"Registering definition for '{definition.term}': '{definition.definition_text}').")
        self._definitions[canonical_term] = definition
        for alias in definition.aliases:
            canonical_alias = self._normalize_term(alias)
            if canonical_alias != canonical_term:
                self._alias_map[canonical_alias] = canonical_term

    def _find_unlocked(self, term: str) -> Optional[Definition]:
        """The actual find logic. Assumes lock is already held."""
        canonical_term = self._normalize_term(term)
        if canonical_term in self._definitions:
            logger.debug(f"Found definition for '{term}' as '{canonical_term}'.")
            return self._definitions[canonical_term]
        if canonical_term in self._alias_map:
            logger.debug(f"Found alias '{term}' for canonical term '{canonical_term}'.")
            primary_canonical_term = self._alias_map[canonical_term]
            return self._definitions[primary_canonical_term]
        return None

    def _find_many_unlocked(self, terms: List[str]) -> List[Definition]:
        found_definitions = []
        found_canonical_terms = set() 

        for term in terms:
            definition = self._find_unlocked(term)
            if definition:
                canonical_key = self._normalize_term(definition.term)
                if canonical_key not in found_canonical_terms:
                    found_definitions.append(definition)
                    found_canonical_terms.add(canonical_key)

        return found_definitions
    
    def _find_best_base_definition_unlocked(self, term: str) -> Optional[Definition]:
        new_term_parts = self._normalize_term(term).split()

        # Step 1: Exact Sub-phrase Matching
        if len(new_term_parts) > 1:
            for i in range(1, len(new_term_parts)):
                sub_phrase_str = " ".join(new_term_parts[i:])
                definition = self._find_unlocked(sub_phrase_str)
                if definition:
                    logger.debug(f"Found base via exact sub-phrase: '{definition.term}' for new term '{term}'.")
                    return definition

        # Step 2: Parameterized Term Matching
        best_param_match_def = None
        max_match_len = 0
        for known_canonical_term, definition in self._definitions.items():
            known_term_parts = known_canonical_term.split()
            k_len = len(known_term_parts)
            if k_len <= 1 or k_len > len(new_term_parts):
                continue
            for i in range(len(new_term_parts) - k_len + 1):
                sub_phrase_parts = new_term_parts[i : i + k_len]
                diff_count = sum(1 for kp, sp in zip(known_term_parts, sub_phrase_parts) if kp != sp)
                if diff_count == 1:
                    if len(known_canonical_term) > max_match_len:
                        max_match_len = len(known_canonical_term)
                        best_param_match_def = definition
                        break
        if best_param_match_def:
            logger.debug(f"Found base via parameterized match: '{best_param_match_def.term}' for new term '{term}'.")
            return best_param_match_def

        return None

    def _to_dict_unlocked(self) -> Dict[str, Dict[str, Any]]:
        bank_data = {}
        for term, definition_obj in self._definitions.items():
            bank_data[term] = {
                "term": definition_obj.term,
                "aliases": definition_obj.aliases,
                "definition_text": getattr(definition_obj, 'definition_text', 'N/A'),
                "source_artifact_id": definition_obj.source_artifact_id,
                "dependencies": definition_obj.dependencies,
            }
        return bank_data