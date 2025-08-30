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
        Respond ONLY with the requested structured JSON format. The "terms" list can be empty.:
        {{
            "terms": ["term1", "term2", "..."]
        }}
        """

        return Prompt(system=system, user=user, id="single_artifact_term_extraction")

    def make_document_term_extraction_prompt(self, full_document_content: str) -> Prompt:
        """
        Generates a prompt to extract all significant terms from the entire document content at once.
        """
        system = """
    You are an expert mathematician and research assistant creating a "prerequisite glossary" for a graduate-level student who is about to read this paper.
    Your task is to analyze the entire document and compile a single, comprehensive list of all specialized mathematical terms, symbols, and concepts that are **crucial for understanding this specific text**.

    The key is to distinguish between specialized knowledge introduced in the paper and foundational knowledge the reader is expected to have.

    **CRITICAL EXTRACTION RULES:**

    **1. WHAT TO EXTRACT (Inclusions):**
        - **Specialized Concepts:** Multi-word terms that are specific to this field or defined in the paper (e.g., 'union-closed family', 'Frobenius norm', 'simplicial complex').
        - **Key Notations & Symbols:** All non-trivial LaTeX commands or symbols that represent core objects of study in this paper (e.g., `\mathcal{F}`, `$G_i$`, `$\hat{X}$`).
            - **This includes specific, named functions or variables (like `f`, `g`, `h`, `X`) that are used consistently as objects of study, even if the general concept (e.g., "function") is foundational.**
        - **Acronyms/Abbreviations:** Any mathematical acronyms defined and used in the text.

    **2. WHAT TO IGNORE (Exclusions):**
        - **DO NOT EXTRACT Foundational Knowledge:** Aggressively filter out any standard, undergraduate-level mathematical concepts. Assume the reader already knows what these are.
        - **DO NOT EXTRACT Procedural & Structural Words:** Ignore all words related to the structure of the paper or logical reasoning.
          Examples to ignore: 'theorem', 'proof', 'lemma', 'corollary', 'proposition', 'definition', 'remark', 'example', 'assumption', 'conclusion', 'let', 'suppose', 'consider'.
        - **DO NOT EXTRACT Common Operators & Relations:** Ignore common mathematical symbols and operators.
          Examples to ignore: `\sum`, `\int`, `\cap`, `\cup`, `\in`, `=`, `\le`, `\ge`, `+`, `-`.

    **CRITICAL FORMATTING AND ATOMICITY RULES:**
        - **ONE TERM PER ENTRY:** Each string in the final JSON list must be a single, atomic mathematical term. Do not bundle multiple terms together, even if they are related.
        - **EXTRACT THE TERM ONLY:** The string should be the pure term as it appears in the text. DO NOT include any explanatory notes, commentary, or definitions in parentheses.
        - **Split Comma-Separated Lists:** If you encounter multiple related terms separated by commas, extract each term as a separate entry.
        - **SEPARATE SYMBOLS AND DESCRIPTIONS:** If a concept and its symbol are mentioned together (e.g., "the upper central series ($Z_i$)"), they should be separate entries in the list if both are significant.
        - **PRESERVE LATEX COMMANDS VERBATIM:** LaTeX commands (e.g., `\bar`, `\mathcal`, `\mathbb`, `\gamma`) are atomic units. They must be extracted **exactly** as they appear in the source text. Do not break them apart, misspell them, or attempt to approximate them.
        - **Examples of CORRECT extraction:**
            - From "$H^1(X)$ and $H^2(X)$ (cohomology groups)" → Extract: "$H^1(X)$" and "$H^2(X)$" as separate entries
            - From "compact space, Hausdorff space, normal space" → Extract: "compact space", "Hausdorff space", "normal space" as separate entries
            - From "$\nabla f$ (gradient of function)" → Extract: "$\nabla f$" only
            - From "both $\mathcal{L}(V,W)$ and $\text{Hom}(V,W)$ (linear maps)" → Extract: "$\mathcal{L}(V,W)$" and "$\text{Hom}(V,W)$" as separate entries
                    
    **3. DEDUPLICATION RULES:**
        - **Avoid Exact Duplicates:** Ensure no term appears twice in the final list.
        - **Avoid Near-Duplicates:** If terms differ only by minor variations (e.g., with/without context in parentheses, subgroup vs group of same type, same symbol with/without descriptive text), include only the most complete or general version.

    """

        user = f"""
        Analyze the following concatenated content from a mathematical paper. Based on the strict rules, identify only the specialized prerequisite terms needed to understand this paper.

        **Full Document Content:**
        ---
        {full_document_content}
        ---

        Respond ONLY with a single structured JSON object containing a flat list of all unique, crucial terms found.
        {{
            "terms": ["term1", "term2", "..."]
        }}
        """
        return Prompt(system=system, user=user, id="document_term_extraction")
    
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
                
        user = f"""You are defining the term: '{term}'. Context from the document:
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