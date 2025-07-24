from typing import Optional
from arxitex.symdef.utils import Definition
from arxitex.llms.prompt import Prompt

class SymbolEnhancementPromptGenerator:

    def make_term_extraction_prompt(self, artifact_content: str) -> str:
        system = f"""
        You are a highly precise mathematical term extraction engine. Your task is to analyze a given text and extract a list of specialized, non-trivial mathematical terms.

You will perform this task by following a strict two-step process in your reasoning:

**Step 1: Candidate Identification**
First, mentally scan the text and identify all potential mathematical symbols, concepts, and named entities.

**Step 2: Strict Filtering**
Next, review your list of candidates and discard any that violate THE FOLLOWING RULES. Only terms that pass ALL rules should be in your final output.

**FILTERING RULES:**

1.  **MUST BE A TERM, NOT AN EXPRESSION:** The output must be the name of a concept, not a statement.
    -   GOOD: `$f$`, `\\varphi`, `union-closed family`, `c-approximate union closed`
    -   BAD: `$f = x^2$`, `A \\cup B`, `$x \\in [0, 1]$`, `$f(\\rho, \\rho)$`
    -   RULE: The term MUST NOT contain operators like `=`, `\\in`, `\\leq`, `\\cup`, `+`.

2.  **MUST BE NON-TRIVIAL:** Do not extract concepts that are common knowledge for a math graduate student.
    -   GOOD: `Frobenius norm`, `sunflower conjecture`
    -   BAD: `set`, `group`, `isomorphism`, `independent random variables`

3.  **MUST BE A SPECIFIC ENTITY:** Do not extract generic variables used for placeholders.
    -   GOOD: `$\\mathcal F$` (if it represents a specific family of sets), `$G$` (if defined as a specific graph).
    -   BAD: `$x$` (when used as a generic variable in an integral), `n` (when used as a generic integer).
    -   HEURISTIC: If a symbol is explicitly defined (e.g., "Let `\\mathcal F` be...") it IS a term. If it's just used in a formula without prior definition, it is likely NOT a term.

4.  **MUST BE CLEAN:** The term cannot be a formatting character like a newline (`\\n`).

Your final output MUST be a single, valid JSON object and nothing else. Do not include any explanation or preamble."""

        user = f"""Analyze the following single mathematical artifact and extract its specialized prerequisite terms according to ALL the rules.
        ---
        {artifact_content}
        ---
        Respond ONLY with the requested structured JSON format. The "terms" list can be empty.:
        {{
            "terms": ["term1", "term2", "..."]
        }}
        """

        return Prompt(system=system, user=user, id="term_extraction")

    def make_definition_extraction_prompt(self, artifact_content: str) -> str:
        """
        Generates a prompt to extract the primary term, its definition, and any aliases
        from an artifact that is itself a definition.
        """
        system = """
        You are a mathematical analyst. Your task is to analyze a text snippet that is known to be a formal definition.
        You must extract the primary term being defined, its complete definition text, and any symbolic aliases assigned to it.

        For example, in the text "A group G is called abelian if...", the primary term is "abelian group", and the alias is "G".
        In "Let F be a union-closed family...", the primary term is "union-closed family" and the alias is "F".
        """

        user = f"""Analyze the following definition:
        ---
        {artifact_content}
        ---

        Extract the following information:
        1.  `defined_term`: The primary conceptual term being defined.
        2.  `definition_text`: The full, verbatim text that constitutes the definition.
        3.  `aliases`: A list of any symbols or alternative names explicitly associated with the term (e.g., ["F", "$\\F$"]). If none, provide an empty list.

        Respond ONLY with the requested structured JSON format:
        {{
            "defined_term": "...",
            "definition_text": "...",
            "aliases": ["...", "..."]
        }}
        """
        return Prompt(system=system, user=user, id="definition_extraction")
    
    def make_definition_synthesis_prompt(self, term: str, context_snippets: str, base_definition: Optional[Definition]) -> str:        
        system = """You are a text-extraction assistant. Your task is to construct a definition for a specific term by ONLY using verbatim sentences from the provided context.
        - **DO NOT** rephrase, summarize, interpret, or generate new text.
        - Your entire response for the `definition` field must be a direct copy-and-paste of sentences from the context.
        - If multiple sentences from the context are needed to form a complete definition, concatenate them.
        - If the context does not contain any sentence that clearly and directly defines the term, you MUST indicate that the context is insufficient."""
        
        user = f"""
            You are defining the term: '{term}'.

            Context from the document:
            ---
            {context_snippets}
            ---
            """

        if base_definition:
            user += f"""
            IMPORTANT: You already have a trusted definition for the base component '{base_definition.term}':
            "{base_definition.definition_text}"
            Your new definition for '{term}' MUST build upon this existing knowledge, explaining the specialization using verbatim sentences from the context.
            """
        
        user += """
    Carefully evaluate if the provided context contains verbatim sentences that define the term.

    - If YES, set `context_was_sufficient` to `true` and construct the `definition` by extracting and concatenating the relevant sentences.
    - If NO, set `context_was_sufficient` to `false` and set the `definition` field to `null`.

    Respond ONLY with the following structured JSON format:

    {
    "context_was_sufficient": true | false,
    "definition": "..." | null
    }"""
        
        return Prompt(system=system, user=user, id="definition_synthesis")