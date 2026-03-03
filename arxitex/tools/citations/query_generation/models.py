from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MentionContext:
    openalex_id: Optional[str]
    arxiv_id: Optional[str]
    context_sentence: Optional[str]
    cite_label: Optional[str]
    location_type: Optional[str]
    context_prev: Optional[str]
    context_next: Optional[str]
    context_paragraph: Optional[str]
    context_html: Optional[str]
    section_title: Optional[str]
    reference_precision: Optional[str]
    bib_entry: Optional[str]
    explicit_refs: Optional[list]

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "MentionContext":
        return cls(
            openalex_id=row.get("openalex_id"),
            arxiv_id=row.get("arxiv_id"),
            context_sentence=row.get("context_sentence"),
            cite_label=row.get("cite_label"),
            location_type=row.get("location_type"),
            context_prev=row.get("context_prev"),
            context_next=row.get("context_next"),
            context_paragraph=row.get("context_paragraph"),
            context_html=row.get("context_html"),
            section_title=row.get("section_title"),
            reference_precision=row.get("reference_precision"),
            bib_entry=row.get("bib_entry"),
            explicit_refs=row.get("explicit_refs"),
        )
