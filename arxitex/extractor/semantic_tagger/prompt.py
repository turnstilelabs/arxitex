from arxitex.llms.prompt import Prompt


class SemanticTagPromptGenerator:
    def make_prompt(self, artifact_text: str, prompt_id: str) -> Prompt:
        system = (
            "You generate a plain-English semantic tag for a mathematical statement. "
            "Return only the JSON that matches the required schema."
        )

        user = f"""
Artifact text:
{artifact_text}

Task:
- Write exactly 1 plain-English sentence (<= 25 words).
- No numbering, no citations, no section labels, no bibliographic metadata.
- Focus on what the statement is about, not where it appears.
"""

        return Prompt(id=prompt_id, system=system, user=user)
