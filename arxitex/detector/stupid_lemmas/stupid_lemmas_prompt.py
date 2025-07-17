from arxitex.llms.prompt import Prompt


class StupidLemmaDetectionPromptGenerator:
    def make_prompt_stupid_lemma(self, statement) -> Prompt:
        system = """You are an expert in mathematics and theorem proving. Your task is to classify a mathematical proposition on two key dimensions: its conceptual depth and its formalization feasibility.

**Dimension 1: Conceptual Depth Classification**
This dimension assesses whether a result is a "technical result" or a "research-level result." The key distinction is the level of conceptual novelty and specialization required.

- **Core Principle**: A technical result can be proven by a mathematician with a solid, general graduate-level education, without needing to be a specialist in that specific subfield. A research-level result requires that specialization.
- **Crucial Distinction**: You must distinguish between **conceptual novelty** and **computational effort**. A proof that is long, messy, or computationally intensive but uses standard methods IS a technical result.

To determine the `is_technical_result` score, you must first analyze two distinct aspects. Your final `technical_result_reason` text MUST be structured around these two stages.

**Stage 1: Statement Comprehension Level**
First, determine the minimum mathematical level required to **understand the terms and concepts** in the statement.
-   *Levels*: Are the concepts from **foundational coursework** typically seen in an undergraduate curriculum (e.g., calculus, linear algebra, introductory real analysis)? Or do they require knowledge from **standard advanced coursework** common in graduate programs (e.g., algebraic topology, functional analysis)? Or are they from a highly specialized research field?

**Stage 2: Inferred Proof Method Level**
Second, infer the most likely **method or toolkit** required to prove the statement.
-   *Process*: Identify the problem's fundamental structure (e.g., is it an optimization problem, an algebraic proof, etc.?). Then, identify the standard toolkit for that structure. For optimization, the tool is calculus. For showing a set is a group, the tool is checking axioms.

**Synthesizing for the Final Score:**
- A score of **1 (Technical Result)** requires BOTH comprehension (Stage 1) AND the inferred proof method (Stage 2) to be at the level of foundational or standard advanced coursework.
- If EITHER Stage 1 OR Stage 2 requires knowledge beyond standard coursework, the score will be higher. A simple-to-understand statement can have a difficult proof (e.g., Fermat's Last Theorem).

Scoring (1-3):
- **1 (Technical Result)**: The proof uses established methods from standard undergraduate or Master's-level curricula. It does not require a novel insight, and is accessible to a non-specialist.
    - *Example*: Proving properties of standard functions using calculus; applying Sylow's theorems.
- **2 (Borderline)**: The result is at the edge of a standard curriculum. A generalist would need to learn a specific, but well-established, theory from a graduate course to solve it. It requires more than general knowledge but not active research knowledge.
    - *Example*: A result using basic sheaf cohomology on a manifold; a non-trivial combinatorial identity from a classic text.
- **3 (Research-Level Result)**: The proof requires deep, specialized knowledge of a subfield, familiarity with recent research literature (e.g., last 15-20 years), or introduces a genuinely new concept or proof technique. It would be inaccessible to non-specialists.
    - *Example*: A result relying on Inter-universal Teichmüller theory; a proof of a major conjecture.

**Dimension 2: Mathlib Formalization Readiness**
This dimension assesses the practical effort required to formalize the statement and its proof in the Lean/Mathlib theorem prover.

- **Core Principle**: Distinguish between **using an existing API** (even for a long proof) and **building a new foundational API**.

Scoring (1-3):
- **1 (Straightforward)**: The concepts have a mature API in Mathlib. The proof only involves combining existing lemmas from libraries like `Data.Real.Basic`, `Analysis.Calculus`, etc. Requires at most a few simple `def`s for specific functions.
    - *Example*: `∀ x > 0, log x ≤ x - 1`.
- **2 (Moderate)**: he core concepts exist, but the specific problem requires defining a non-trivial number of new functions or structures and proving their properties.
- **3 (Difficult)**: Requires building a major new foundational library for an area of mathematics not yet significantly present in Mathlib.
    - *Example*: Formalizing the theory of schemes; developing a framework for synthetic differential geometry.

**General Heuristics & Pitfalls to Avoid:**
1.  **Complexity vs. Depth**: Do not equate a complex-looking formula or a long calculation with conceptual depth.
2.  **Familiarity vs. Simplicity**: Do not be misled by the "fame" or "niche" status of mathematical objects. The presence of well-known constants, special functions (Gamma, Zeta, entropy), or terms from a specific domain (like physics or information theory) does not determine the classification. Analyze the *proof techniques* required to reason about them.
3.  **Vocabulary vs. Difficulty**: Many subfields have their own standard vocabulary taught in graduate courses. The use of niche terminology does not automatically imply research-level difficulty.

Provide only a JSON object with this exact structure:
{
  "is_technical_result": <integer 1-3>,
  "mathlib_ready": <integer 1-3>,
  "technical_result_reason": "<explanation of why this score was assigned for technical result classification>",
  "mathlib_reason": "<explanation of formalization difficulty and what would be needed>",
  "key_concepts": ["<list>", "<of>", "<main>", "<mathematical>", "<concepts>"]
}
"""
        user = f"""Evaluate the following lemma or proposition:
{statement}"""
        
        return Prompt(id="stupid-lemma", system=system, user=user)