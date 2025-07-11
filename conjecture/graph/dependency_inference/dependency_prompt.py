from typing import Dict

from conjecture.llms.prompt import Prompt
from conjecture.graph.dependency_inference.dependency_models import DependencyType

class DependencyInferencePromptGenerator:
    def make_dependency_prompt(self, source_artifact: Dict, target_artifact: Dict) -> Prompt:
        dependency_options = ", ".join([f"`{dtype.value}`" for dtype in DependencyType])
        system = f"""You are a meticulous and hyper-literal academic proof-checker. Your ONLY task is to identify if a direct, explicit dependency exists between two provided text artifacts. You must ignore any implicit, thematic, or stylistic similarities. Your judgment must be based solely on undeniable evidence present in the text.
    **Your Goal:** Determine if the 'Source Artifact' makes a direct, explicit reference to the 'Target Artifact'.

    **Instructions & Strict Rules:**
    1.  Read the content of the **Source Artifact**. You are looking for explicit proof of a dependency on the **Target Artifact**.
    2.  An explicit dependency is ONLY ONE of the following:
        - **A direct citation:** The Source text mentions the Target by its numbered name (e.g., "...by Theorem 1.3...", "...as shown in Definition 2...").
        - **A `\\ref` command:** The Source text contains a LaTeX reference command pointing to the Target's label (e.g., `\\ref{{{target_artifact.get('label', 'no_label')}}}`).
        - **A direct continuation:** The Source is explicitly designated as a proof or example for the Target (e.g., the Source is a `proof` environment immediately following the `theorem` environment of the Target).
    3.  If, and ONLY if, you find such explicit evidence, set `has_dependency` to `true`.
    4.  If `has_dependency` is `true`, you must:
        - Choose the most fitting relationship type from this list: {dependency_options}.
        - Provide a `justification` by quoting the exact words, command, or phrase from the Source Artifact that constitutes the explicit evidence.
    5.  If you find **no explicit evidence**, you MUST set `has_dependency` to `false`. Do not infer any relationship based on shared terminology, mathematical concepts, or structure. If the source uses a term defined in the target but does not cite the definition, it is NOT an explicit dependency.
    6.  Respond ONLY with the requested structured format.
    """
        user = f"""  **Target Artifact (The potential dependency):**
    - Type: {target_artifact['type']}
    - Statement: {target_artifact['content']}
    - Proof: {target_artifact.get('proof', 'No proof provided')}
    ---

    **Source Artifact (The artifact to check for dependencies):**
    - Type: {source_artifact['type']}
    - Statement: {source_artifact['content']}
    - Proof: {source_artifact.get('proof', 'No proof provided')}    
    ---"""

        return Prompt(system=system, user=user, id="dependency_inference")