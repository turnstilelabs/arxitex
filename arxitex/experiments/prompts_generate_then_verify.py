from __future__ import annotations

from arxitex.llms.prompt import Prompt


def prompt_for_queries(
    category: str, artifact_type: str, artifact_text: str, k: int
) -> Prompt:
    """
    Generates a structured prompt for creating synthetic search queries.

    This prompt is carefully designed to simulate a realistic scenario: a mathematician
    is searching for a concept or result (the 'artifact_text') but does NOT know
    of the existence of the source paper. The model is explicitly forbidden from
    using information from the paper's title or abstract.
    """

    system_prompt = """You are a research mathematician simulator. Your task is to generate plausible search queries that a real mathematician would use when exploring a concept or looking for a specific type of result. You must be realistic. Your output must be a JSON object with a single key 'queries' containing a list of strings."""

    scenario_description = """Imagine you are a research mathematician. The 'Artifact Text' below represents a mathematical theorem you have in mind. You DO NOT KNOW that this has been published, and you have NO KNOWLEDGE of the paper it comes from. Your goal is to generate search queries to discover if a result like this exists in the literature."""

    style_instructions = {
        "precise_assertion": """Generate queries that formulate the *central claim* of the theorem as a precise statement or question. These queries should be specific enough that they would likely find this exact theorem.
- Frame them as a search for a known result (e.g., 'Korovkin's theorem for sublinear operators').
- Crucially, include specific mathematical notation in LaTeX (e.g., `$\\mathbb(R)^(N)$`, `$\\mathcal(F)(X)$`) where it adds necessary precision.""",
        "imperfect_recall": """Simulate a researcher recalling the theorem from memory, but imperfectly. Generate queries that are technically precise but introduce plausible, minor inaccuracies.
- Omit one of the secondary conditions or hypotheses from the theorem.
- Change variable names (e.g., use `K` instead of `X` for a set).
- Slightly alter technical terms (e.g., 'monotone positive operators' instead of 'monotone and sublinear operators').
- The query should still be specific, but flawed in a realistic way.""",
        "conceptual_search": """Generate queries that ask about the relationship or implication described in the theorem, but in less formal terms. Focus on the *meaning* of the result. For example, 'convergence criteria for sequences of operators' or 'when does pointwise convergence imply uniform convergence'.""",
        "exploratory_search": """Generate broader, higher-level queries. These should capture the general mathematical topic or field that the theorem belongs to, such as 'nonlinear approximation theory' or 'positive linear operators'.""",
    }

    rules_common = """The queries must be highly realistic for a mathematician using a search engine like Google Scholar or arXiv.
- NEVER use any names, dates, or direct quotes from the paper's title or abstract.
- NEVER include reference labels like `\\label{...}`.
- Do not use instructional phrasing like 'find a paper on'.
- You SHOULD use LaTeX for mathematical notation when it adds precision."""

    style = style_instructions.get(
        category, "Generate general queries about the topic."
    )

    user_prompt = f"""{scenario_description}

<ArtifactContext>
Artifact Type: {artifact_type}
Artifact Text: "{artifact_text}"
</ArtifactContext>

<Task>
Instruction: {style} {rules_common}
Generate exactly {k} queries.
</Task>"""

    return Prompt(system=system_prompt, user=user_prompt, id="query_generation")


def prompt_closed_book(query: str, k: int = 5) -> Prompt:
    """
    Generates a structured prompt for closed-book retrieval of multiple paper candidates.
    """
    system_prompt = """You are an expert research assistant with a deep, specialized knowledge of academic literature, particularly the arXiv preprint server and major mathematical journals. Your task is to recall and cite the most relevant research papers for a given query. You must respond strictly in the JSON format provided."""

    user_prompt = f"""Based on your knowledge, what are the top {k} most likely research papers a researcher is looking for with the following query?

Query: "{query}"

Rules:
1. Provide your answer as a ranked list in the JSON structure below. The first entry should be your top guess.
2. Each title must be the exact, full title of the publication.
3. Prioritize primary research articles. Do not recommend survey papers or books unless the query is extremely broad.

```json
{{
  "candidates": [
    {{
      "reference": {{ "title": "The full and exact title of the #1 paper" }},
      "confidence": 0.9,
      "reasoning": "A brief justification for why this is the best match."
    }},
    {{
      "reference": {{ "title": "The full and exact title of the #2 paper" }},
      "confidence": 0.8,
      "reasoning": "A brief justification for why this is a plausible alternative."
    }}
  ]
}}
```"""

    return Prompt(system=system_prompt, user=user_prompt, id="closed_book_retrieval")
