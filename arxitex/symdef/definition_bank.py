
from typing import Any, List, Dict, Optional
from loguru import logger
from conjecture.symbol.utils import Definition

class DefinitionBank:
    """The 'working memory' holding all definitions found so far."""
    def __init__(self):
        self._definitions: Dict[str, Definition] = {}
        self._alias_map: Dict[str, str] = {}

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

    def register(self, definition: Definition):
        """Adds or updates a definition using its canonical form as the key."""
        canonical_term = self._normalize_term(definition.term)

        # If a definition for this canonical term already exists, we overwrite it.
        # This is desired behavior, as later definitions in a paper are often more specific.
        if canonical_term in self._definitions:
            logger.debug(f"Overwriting definition for canonical term '{canonical_term}'. Original term: '{definition.term}'.")
        
        logger.info(f"Registering definition for '{definition.term}' (canonical: '{canonical_term}').")
        self._definitions[canonical_term] = definition

        for alias in definition.aliases:
            canonical_alias = self._normalize_term(alias)
            if canonical_alias != canonical_term:
                self._alias_map[canonical_alias] = canonical_term

    def find(self, term: str) -> Optional[Definition]:
        """Finds a definition by its canonical form, checking primary terms and aliases."""
        canonical_term = self._normalize_term(term)
        
        if canonical_term in self._definitions:
            return self._definitions[canonical_term]
            
        if canonical_term in self._alias_map:
            primary_canonical_term = self._alias_map[canonical_term]
            return self._definitions[primary_canonical_term]
            
        return None

    def find_best_base_definition(self, term: str) -> Optional[Definition]:
        """Finds the best base definition, operating on canonical forms."""
        new_term_parts = self._normalize_term(term).split()

        # Phase 1: Exact Sub-phrase Matching
        if len(new_term_parts) > 1:
            for i in range(1, len(new_term_parts)):
                sub_phrase_str = " ".join(new_term_parts[i:])
                definition = self.find(sub_phrase_str) # `find` normalizes correctly
                if definition:
                    logger.debug(f"Found base via exact sub-phrase: '{definition.term}' for new term '{term}'.")
                    return definition

        # --- Step 2: Parameterized Term Matching (New Logic) ---
        best_param_match_def = None
        max_match_len = 0

        for known_canonical_term, definition in self._definitions.items():
            known_term_parts = known_canonical_term.split()
            k_len = len(known_term_parts)

            # CRITICAL GUARD: The parameterized matching heuristic is only meaningful
            # for multi-word terms (k_len > 1).
            # Also, the known term cannot be longer than the new term.
            if k_len <= 1 or k_len > len(new_term_parts):
                continue

            # Use a "sliding window" to check every sub-phrase of the new term
            # that has the same length as the known term.
            for i in range(len(new_term_parts) - k_len + 1):
                sub_phrase_parts = new_term_parts[i : i + k_len]
                
                # Now, perform the single-difference check between the known term and the sub-phrase.
                diff_count = sum(1 for known_part, sub_part in zip(known_term_parts, sub_phrase_parts) if known_part != sub_part)
                
                # If they differ by exactly one "word", it's a parameterized match.
                if diff_count == 1:
                    if len(known_canonical_term) > max_match_len:
                        max_match_len = len(known_canonical_term)
                        best_param_match_def = definition
                        break 
        
        if best_param_match_def:
            logger.debug(f"Found base via parameterized match: '{best_param_match_def.term}' for new term '{term}'.")
            return best_param_match_def

        return None

    def get_all_known_terms(self) -> List[str]:
        return list(self._definitions.keys()) + list(self._alias_map.keys())
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Exports the bank's definitions to a JSON-serializable dictionary.
        The structure is { "primary_term_lower": { ...definition_data... } }.
        This format is suitable for writing to a JSON file.
        """
        bank_data = {}
        for term, definition_obj in self._definitions.items():
            bank_data[term] = {
                "term": definition_obj.term,
                "aliases": definition_obj.aliases,
                "definition_text": getattr(definition_obj, 'definition_text', 'N/A'),
                "source_artifact_id": definition_obj.source_artifact_id,
            }
        return bank_data