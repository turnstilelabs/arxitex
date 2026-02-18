"""Citation tooling (OpenAlex + arXiv matching + dataset scripts)."""

from arxitex.tools.citations.arxiv_backfill import (
    backfill_external_reference_arxiv_matches,
)
from arxitex.tools.citations.arxiv_matcher import match_external_reference_to_arxiv
from arxitex.tools.citations.backfill import run_backfill as run_citations_backfill
from arxitex.tools.citations.openalex import (
    backfill_citations_openalex,
    strip_arxiv_version,
)

__all__ = [
    "backfill_citations_openalex",
    "backfill_external_reference_arxiv_matches",
    "match_external_reference_to_arxiv",
    "run_citations_backfill",
    "strip_arxiv_version",
]
