from typing import Dict

from arxitex.extractor.models import DependencyType
from arxitex.llms.prompt import Prompt


class DependencyInferencePromptGenerator:
    def make_dependency_prompt(
        self, source_artifact: Dict, target_artifact: Dict
    ) -> Prompt:
        dependency_options = ", ".join([f"`{dtype.value}`" for dtype in DependencyType])
        system = f"""
        You are an expert mathematician and logician acting as a high-precision proof-checker. Your task is to determine if a direct logical or conceptual dependency exists between two provided mathematical artifacts.

        **CONTEXT:** You have been given this specific pair because a preliminary analysis found that they share significant, specialized terminology. This strongly suggests a potential relationship, and your job is to perform the final expert verification.

        **YOUR GOAL:** Determine if the 'Source Artifact' logically relies on a definition, result, or concept presented in the 'Target Artifact'.

        **INSTRUCTIONS & RESPONSE REQUIREMENTS:**
        1. **DECISION TASK:** Decide whether the Source depends on the Target.
        2. **WHEN TO LABEL A DEPENDENCY:**
           - Choose **`{DependencyType.GENERALIZES.value}`** only if the Source statement is a strict generalization of the Target statement (i.e., the Target is a clear special case of the Source).
           - Otherwise, if the Source relies on the Target in any way (uses its definitions, results, or concepts), choose **`{DependencyType.USED_IN.value}`**.
        3. **EVIDENCE REQUIREMENT (CRITICAL):**
           - Your `justification` must quote at least one short phrase from the Source that shows the dependency.
           - Also mention what in the Target is being used/generalized (term/label/concept), briefly.
        4. **OUTPUT CONTRACT:**
           - If there is a dependency: `has_dependency=true`, `dependency_type` must be one of: {dependency_options}, and `justification` must be a short string.
           - If there is NO dependency: `has_dependency=false`, `dependency_type=null`, `justification=null`.
        """

        user = f"""
        Please analyze the following pair for a logical dependency, based on all the rules provided.

        **Target Artifact (The potential prerequisite):**
        - Type: `{target_artifact['type']}`
        - Label: `{target_artifact.get('label', 'N/A')}`
        - Statement:
        ```latex
        {target_artifact['content']}
        ```
        - Proof:
        ```latex
        {target_artifact.get('proof', 'No proof provided')}
        ```
        ---

        **Source Artifact (The potential dependent):**
        - Type: `{source_artifact['type']}`
        - Label: `{source_artifact.get('label', 'N/A')}`
        - Statement:
        ```latex
        {source_artifact['content']}
        ```
        - Proof:
        ```latex
        {source_artifact.get('proof', 'No proof provided')}
        ```
        ---
        """

        return Prompt(system=system, user=user, id="dependency_inference")
