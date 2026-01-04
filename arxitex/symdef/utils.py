import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import aiofiles
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from arxitex.extractor.models import ArtifactNode


@dataclass
class Definition:
    """Represents a single, resolved definition for a term."""

    term: str
    definition_text: str
    source_artifact_id: str
    aliases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(
        default_factory=list
    )  # e.g., "abelian group" depends on "group"


class ContextFinder:

    def find_context_around_first_occurrence(
        self, term: str, text_to_search: str
    ) -> str:
        """
        Finds the first occurrence of a term and returns the full paragraph containing it.
        """
        try:
            # Step 1: Pre-process the term.
            search_term = (
                term[1:-1]
                if term.startswith("$") and term.endswith("$") and len(term) > 2
                else term
            )
            escaped_term = re.escape(search_term)
            first_match = None

            # A common, robust suffix for all patterns.
            suffix = r"(?=[\s\(\)\[\]\{\},.=+\-*/<>,]|\$|$)"

            # Step 2: Check if the term is an ambiguous single-character alphabetic term.
            is_ambiguous_term = len(search_term) == 1 and search_term.isalpha()

            if is_ambiguous_term:
                # STAGE 1: Strict, high-confidence search for math-mode variables (e.g., "$f").
                # Pattern must be preceded by a literal dollar sign.
                # This search is CASE-SENSITIVE by default.
                strict_pattern = rf"\$({escaped_term}){suffix}"
                logger.debug(
                    f"Ambiguous term '{term}'. First trying strict pattern: {strict_pattern}"
                )
                first_match = next(re.finditer(strict_pattern, text_to_search), None)

                if not first_match:
                    # STAGE 2: Fallback for definitions like "Let f be..."
                    # The prefix MUST NOT be a backslash, to avoid matching inside \mathcalF, etc.
                    # This uses a negative lookbehind `(?<!\\)` to assert this.
                    fallback_prefix = r"(?<!\\)(?:^|\s|[\(\[\{,=+\-*/<>,])"
                    fallback_pattern = rf"{fallback_prefix}({escaped_term}){suffix}"
                    logger.debug(
                        f"Strict pattern failed. Falling back to general pattern: {fallback_pattern}"
                    )
                    first_match = next(
                        re.finditer(fallback_pattern, text_to_search, re.IGNORECASE),
                        None,
                    )

            if not is_ambiguous_term or first_match is None:
                # Use the general, flexible pattern for all non-ambiguous terms (like 'h(x)', '\varphi')
                # or if the ambiguous search still needs a final attempt.
                prefix = r"(?:^|\s|[\(\[\{,=+\-*/<>,]|\$)"
                pattern = rf"{prefix}({escaped_term}){suffix}"
                logger.debug(f"Using general pattern for term '{term}': {pattern}")
                match_flags = re.IGNORECASE if search_term.isalpha() else 0
                first_match = next(
                    re.finditer(pattern, text_to_search, match_flags), None
                )

            if not first_match:
                logger.warning(f"Term '{term}' not found in the preceding text.")
                return ""

        except re.error as e:
            logger.error(f"Regex error for term '{term}': {e}", exc_info=True)
            return ""

        # Step 4: Extract the paragraph containing the match.
        # Group 1 always contains our desired term.
        match_start_pos = first_match.start(1)

        para_start_pos = text_to_search.rfind("\n\n", 0, match_start_pos)
        para_start_pos = 0 if para_start_pos == -1 else para_start_pos + 2

        para_end_pos = text_to_search.find("\n\n", match_start_pos)
        para_end_pos = len(text_to_search) if para_end_pos == -1 else para_end_pos

        definitional_paragraph = text_to_search[para_start_pos:para_end_pos].strip()

        return definitional_paragraph


def clean_latex_for_llm(text: str) -> str:
    """
    Removes common LaTeX structural and metadata commands to clean up context for an LLM.

    This function removes commands that define document structure but not content,
    such as environments, labels, and sectioning commands.

    Examples:
        - '\\begin{claim}' -> ''
        - '\\label{f_min}' -> ''
        - '\\end{theorem}' -> ''
        - '\\section*{Introduction}' -> 'Introduction'
    """
    if not text:
        return ""

    # Rule 1: Remove \begin{...} and \end{...} commands
    cleaned_text = re.sub(r"\\(begin|end)\{[a-zA-Z0-9_*]+\}\s*", "", text)

    # Rule 2: Remove \label{...} commands
    cleaned_text = re.sub(r"\\label\{[^\}]+\}\s*", "", cleaned_text)

    # Rule 3: Remove common no-argument commands like \item or \centering
    cleaned_text = re.sub(
        r"\\(item|centering|newpage|clearpage)\b\s*", "", cleaned_text
    )

    # Rule 4: Handle sectioning commands by keeping their title but removing the command itself.
    cleaned_text = re.sub(
        r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]+)\}",
        r"\2",
        cleaned_text,
    )

    # Rule 5: Collapse multiple blank lines into a single one for readability.
    cleaned_text = re.sub(r"(\n\s*){3,}", "\n\n", cleaned_text).strip()

    return cleaned_text


def load_artifacts_from_json(file_path: Path) -> List[ArtifactNode]:
    """Loads artifacts from a JSON file and validates them."""
    if not file_path.exists():
        logger.error(f"Artifact JSON file not found at: {file_path}")
        sys.exit(1)

    logger.info(f"Loading artifacts from {file_path}...")
    try:
        # Use Pydantic's TypeAdapter for robust list validation
        ArtifactListAdapter = TypeAdapter(List[ArtifactNode])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            artifacts = ArtifactListAdapter.validate_python(data.get("nodes", []))
        logger.success(f"Successfully loaded and validated {len(artifacts)} artifacts.")
        return artifacts
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or validate artifacts from {file_path}: {e}")
        sys.exit(1)


def load_latex_content(file_path: Path) -> str:
    """Loads the full LaTeX source code from a file."""
    if not file_path.exists():
        logger.error(f"LaTeX source file not found at: {file_path}")
        sys.exit(1)

    logger.info(f"Loading LaTeX source from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.success("LaTeX source loaded.")
    return content


def save_enhanced_artifacts(results: dict, output_path: Path):
    """Saves the enhanced artifact data to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving enhanced artifacts to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.success("Results saved successfully.")


async def async_load_artifacts_from_json(file_path: Path) -> List["ArtifactNode"]:
    if not file_path.exists():
        logger.error(f"Artifact JSON file not found at: {file_path}")
        sys.exit(1)

    logger.info(f"Loading artifacts from {file_path}...")
    try:
        ArtifactListAdapter = TypeAdapter(List["ArtifactNode"])
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            artifacts = ArtifactListAdapter.validate_python(data.get("nodes", []))
        logger.success(f"Successfully loaded and validated {len(artifacts)} artifacts.")
        return artifacts
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or validate artifacts from {file_path}: {e}")
        sys.exit(1)


async def async_load_latex_content(file_path: Path) -> str:
    if not file_path.exists():
        logger.error(f"LaTeX source file not found at: {file_path}")
        sys.exit(1)

    logger.info(f"Loading LaTeX source from {file_path}...")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
    logger.success("LaTeX source loaded.")
    return content


async def async_save_enhanced_artifacts(results: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving enhanced artifacts to {output_path}...")
    enhanced_artifacts = results.get("artifacts", {})
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(enhanced_artifacts, indent=2))
    logger.success("Results saved successfully.")


def create_canonical_search_string(text: str) -> str:
    """
    Transforms a string into a delimiter-free canonical format for robust searching.
    (This is the same robust helper we developed before).
    """
    text = text.replace("$", "")
    text = re.sub(r"([\[\]\(\)\{\},=+\-*/<>:])", r" \1 ", text)
    canonical_string = re.sub(r"\s+", " ", text).strip()
    return canonical_string


# --- LaTeX macro utilities -------------------------------------------------

_MACRO_PATTERNS = [
    # \newcommand{\cF}{\mathcal{F}}
    re.compile(
        r"\\newcommand\s*\{\s*\\(?P<name>[A-Za-z@]+)\s*\}\s*\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}",
        re.MULTILINE,
    ),
    # \renewcommand{\cF}{...}
    re.compile(
        r"\\renewcommand\s*\{\s*\\(?P<name>[A-Za-z@]+)\s*\}\s*\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}",
        re.MULTILINE,
    ),
    # \def\cF{...}
    re.compile(
        r"\\def\s*\\(?P<name>[A-Za-z@]+)\s*\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}",
        re.MULTILINE,
    ),
    # \DeclareMathOperator{\Hom}{Hom} / \DeclareMathOperator*{\Hom}{Hom}
    re.compile(
        r"\\DeclareMathOperator\*?\s*\{\s*\\(?P<name>[A-Za-z@]+)\s*\}\s*\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}",
        re.MULTILINE,
    ),
]


def extract_latex_macros(latex: str) -> Dict[str, str]:
    """Best-effort extraction of simple, argument-free LaTeX macros.

    Returns a mapping from macro name (without leading backslash) to its
    replacement body, e.g. {"cF": "\\mathcal{F}"}.

    This intentionally ignores macros with arguments ("#1", "#2", ...)
    to keep behaviour predictable and low-risk.
    """

    if not latex:
        return {}

    # Restrict search to the preamble for safety.
    doc_start = latex.find("\\begin{document}")
    search_region = latex if doc_start == -1 else latex[:doc_start]

    macros: Dict[str, str] = {}

    for pattern in _MACRO_PATTERNS:
        for match in pattern.finditer(search_region):
            name = match.group("name")  # e.g. "cF"
            body = (match.group("body") or "").strip()

            if not name or not body:
                continue

            # Skip macros whose body appears to take arguments; supporting
            # those correctly would require more TeX awareness than we want.
            if "#1" in body or "#2" in body or "#3" in body:
                continue

            macros[name] = body

    return macros
