import re
from typing import Dict

from loguru import logger


class NewTheoremScanner:
    """Utility for discovering custom theorem-like environments from LaTeX.

    It scans for ``\\newtheorem`` declarations and returns a mapping from
    environment name (e.g. ``thm1``) to canonical artifact type
    (e.g. ``"theorem"``) compatible with :class:`ArtifactType`.
    """

    # Keywords in the human-readable theorem name that map to canonical types.
    TITLE_KEYWORDS: Dict[str, str] = {
        "theorem": "theorem",
        "lemma": "lemma",
        "proposition": "proposition",
        "corollary": "corollary",
        "definition": "definition",
        "remark": "remark",
        "example": "example",
        "claim": "claim",
        "observation": "observation",
        "conjecture": "conjecture",
        "fact": "fact",
    }

    # Regex for the main \newtheorem forms we care about:
    #   \newtheorem{env}{Title}
    #   \newtheorem{env}{Title}[within]
    #   \newtheorem{env}[numbered_like]{Title}
    #   \newtheorem{env}[numbered_like]{Title}[within]
    _NEWTHEOREM_PATTERN = re.compile(
        r"\\newtheorem"  # command
        r"\s*"  # optional space
        r"\{([^}]+)\}"  # {env}
        r"\s*(?:\[[^\]]+\])?"  # optional [numbered_like]
        r"\s*\{([^}]+)\}"  # {Title}
    )

    @classmethod
    def scan(cls, content: str) -> Dict[str, str]:
        """Scan LaTeX content for ``\\newtheorem`` declarations.

        Returns a mapping ``env_name -> canonical_type`` where ``canonical_type``
        is a lowercase string like ``"theorem"`` or ``"lemma"`` that should be
        present in ``BaseGraphBuilder.ARTIFACT_TYPES``.
        """

        aliases: Dict[str, str] = {}

        for match in cls._NEWTHEOREM_PATTERN.finditer(content):
            env_name = match.group(1).strip()
            title = match.group(2).strip()
            title_norm = title.lower()

            canonical_type = None
            for keyword, mapped in cls.TITLE_KEYWORDS.items():
                if keyword in title_norm:
                    canonical_type = mapped
                    break

            if not canonical_type:
                # Unknown/new type; skip but log for debugging.
                logger.debug(
                    f"NewTheoremScanner: could not infer type for env '{env_name}' with title '{title}'."
                )
                continue

            # Only set if we haven't already seen this env_name; first definition wins.
            aliases.setdefault(env_name, canonical_type)

        if aliases:
            logger.debug(
                "NewTheoremScanner: discovered env aliases: "
                + ", ".join(f"{k}->{v}" for k, v in aliases.items())
            )

        return aliases
