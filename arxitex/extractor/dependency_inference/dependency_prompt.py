from typing import Dict

from arxitex.llms.prompt import Prompt
from arxitex.extractor.models import DependencyType 

class DependencyInferencePromptGenerator:
    def make_dependency_prompt(self, source_artifact: Dict, target_artifact: Dict) -> Prompt:
        dependency_options = ", ".join([f"`{dtype.value}`" for dtype in DependencyType])
        system = f"""
        You are an expert mathematician and logician acting as a high-precision proof-checker. Your task is to determine if a direct logical or conceptual dependency exists between two provided mathematical artifacts.

        **CONTEXT:** You have been given this specific pair because a preliminary analysis found that they share significant, specialized terminology. This strongly suggests a potential relationship, and your job is to perform the final expert verification.

        **YOUR GOAL:** Determine if the 'Source Artifact' logically relies on a definition, result, or concept presented in the 'Target Artifact'.

        **INSTRUCTIONS & RESPONSE REQUIREMENTS:**
        1.  **THE PRINCIPLE OF FOCUSED INQUIRY:** This is your most important rule. The shared terminology is your primary clue. Your task is to **actively investigate the logical connection implied by this clue.** Your default assumption should be that the shared terminology is meaningful, not coincidental. Prioritize finding how the Source uses the Target's concepts.
        2.  **DEFINITION OF DEPENDENCY:** A dependency exists if the Source Artifact does any of the following:
            - **`{DependencyType.USES_RESULT.value}`:** Directly applies or references a theorem, lemma, proposition, or result proven in the Target.
            - **`{DependencyType.USES_DEFINITION.value}`:** Employs a term, notation, or concept that was formally defined or introduced in the Target.
            - **`{DependencyType.PROVIDES_EXAMPLE.value}`:** Serves as a specific illustration of a general concept from the Target.
            - **`{DependencyType.PROVES.value}`:** Is the formal proof of a claim or theorem statement made in the Target.
            - **`{DependencyType.IS_COROLLARY_OF.value}`:** Is a direct and immediate consequence of the Target's main result.
            - **`{DependencyType.IS_SPECIAL_CASE_OF.value}`:** Is a more specific version of a general result in the Target.
            - **`{DependencyType.IS_GENERALIZATION_OF.value}`:** Presents a result that broadens or extends a result from the Target.

        3.  **MANDATORY RESPONSE FIELDS:**
            - If you find a dependency, you MUST set `has_dependency` to `true`.
            - You MUST then choose the single most fitting relationship type for `dependency_type` from this exact list: {dependency_options}.
            - You MUST provide a concise `justification` that explains **how** the Source depends on the Target. Quote the specific words from the Source that provide the evidence.
            - If, after careful analysis, you conclude the shared terms are used coincidentally and there is no logical dependency, you MUST set `has_dependency` to `false`.
        
        4.  **RESPONSE REQUIREMENTS:**
            - If you find a dependency, you MUST set `has_dependency` to `true`.
            - You MUST then choose the single most fitting relationship type for the `dependency_type` field from this exact list: {dependency_options}.
            - You MUST provide a concise `justification` that explains **how** the Source depends on the Target. Quote the specific words or phrases from the Source that provide the evidence for your decision.
            - If the shared terminology is merely coincidental and the Source makes no logical use of the Target's content, you MUST set `has_dependency` to `false`.
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