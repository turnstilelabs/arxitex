from typing import Optional
from arxitex.symdef.utils import Definition
from arxitex.llms.prompt import Prompt

class SymbolEnhancementPromptGenerator:

    def make_term_extraction_prompt(self, artifact_content: str) -> str:
        system = f"""
        Analyze the following mathematical text. Your task is to identify and list all non-trivial mathematical symbols 
        (like $\\F$, $G_i$) and specialized concepts (like 'union-closed family', 'Frobenius norm') 
        that are crucial for understanding this specific text.

        - Do NOT include common mathematical operators (+, -, \\cup) or generic words ('set', 'element', 'theorem').
        - The goal is to identify terms whose meaning is likely defined within this document."""


        user = f"""Analyze the following single mathematical artifact and extract its specialized prerequisite terms according to ALL the rules.
        ---
        {artifact_content}
        ---
        Respond ONLY with the requested structured JSON format:
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