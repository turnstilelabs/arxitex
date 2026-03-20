"""Mention extraction helpers for the mentions pipeline stage 2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text

from arxitex.tools.mentions.acquisition.target_resolution import (
    TargetWorkProfile,
    classify_bib_entry,
)
from arxitex.tools.mentions.extraction.mention_utils import (
    build_label_regex,
    derive_labels_from_entry,
    find_sentence_index,
    normalize_for_match,
    split_sentences,
    title_matches_entry,
)
from arxitex.tools.mentions.utils import extract_refs


def _window_text(sentences: List[str], idx: int, window: int = 1) -> str:
    if not sentences:
        return ""
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(s for s in sentences[start:end] if s)


@dataclass
class MentionExtractor:
    target_profile: TargetWorkProfile

    def extract_from_paragraph(
        self,
        text: str,
        section_title: Optional[str],
        location_type: str,
        source: str,
        source_url: str,
        base: Dict[str, Any],
        labels: List[str],
        context_html: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        mentions: List[Dict[str, Any]] = []
        sentences = split_sentences(text)
        if not labels:
            return mentions
        for label in labels:
            label_re = build_label_regex(label)
            idx = find_sentence_index(sentences, label_re)
            if not sentences or idx < 0:
                continue
            sentence_text = sentences[idx] if sentences else ""
            window_text = _window_text(sentences, idx, window=1)
            explicit_refs = extract_refs(sentence_text)
            explicit_ref_source = "cite_sentence" if explicit_refs else None
            if not explicit_refs:
                explicit_refs = extract_refs(window_text)
                explicit_ref_source = "window" if explicit_refs else None
            if not explicit_refs:
                explicit_refs = extract_refs(text)
                explicit_ref_source = "paragraph" if explicit_refs else None
            mentions.append(
                {
                    **base,
                    "match_text": label,
                    "location_type": location_type,
                    "section_title": section_title,
                    "context_prev": sentences[idx - 1] if idx - 1 >= 0 else None,
                    "context_sentence": sentences[idx],
                    "context_next": (
                        sentences[idx + 1] if idx + 1 < len(sentences) else None
                    ),
                    "context_paragraph": text,
                    "context_html": context_html,
                    "source": source,
                    "source_url": source_url,
                    "explicit_refs": explicit_refs,
                    "explicit_ref_source": explicit_ref_source,
                    "reference_precision": "explicit" if explicit_refs else "implicit",
                }
            )
        return mentions

    def extract_from_html(
        self,
        html_path: str,
        source_url: str,
        base: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        mentions: List[Dict[str, Any]] = []
        bib_targets: Dict[str, Dict[str, Any]] = {}
        for bib in soup.select(".ltx_bibliography .ltx_bibitem, .ltx_bibitem"):
            bib_id = bib.get("id")
            if not bib_id:
                continue
            text = bib.get_text(" ", strip=True)
            if not text:
                continue
            match_status = classify_bib_entry(text, self.target_profile)
            if match_status not in {"exact_target", "same_work_alt_version"}:
                continue
            if not self.target_profile.doi and not title_matches_entry(
                text, self.target_profile.title
            ):
                # Keep strictness when no DOI anchor is available.
                continue
            tag = bib.select_one(".ltx_bibtag")
            label = tag.get_text(" ", strip=True) if tag else ""
            labels = [label] if label else []
            if not labels:
                labels = derive_labels_from_entry(text)
            bib_targets[bib_id] = {
                "labels": labels,
                "text": text,
                "target_match_status": match_status,
            }

        if bib_targets:
            seen: set = set()
            for a in soup.select("a.ltx_ref, a.ltx_cite"):
                href = a.get("href") or ""
                if not href.startswith("#"):
                    continue
                bib_id = href[1:]
                if bib_id not in bib_targets:
                    continue

                if a.find_parent(class_="ltx_bibliography") is not None:
                    continue

                container = (
                    a.find_parent(class_="ltx_para")
                    or a.find_parent("p")
                    or a.find_parent("li")
                )
                if container is None:
                    continue

                section = None
                heading = container.find_previous(["h1", "h2", "h3", "h4", "h5"])
                if heading is not None:
                    section = heading.get_text(" ", strip=True) or None

                marker = "__CITE_MARKER__"
                container_copy = BeautifulSoup(str(container), "lxml")
                marker_anchor = container_copy.find("a", href=f"#{bib_id}")
                if marker_anchor is not None:
                    marker_anchor.replace_with(marker)
                para_text = container_copy.get_text(" ", strip=True)
                if not para_text:
                    continue
                context_html = str(container)

                sentences = split_sentences(para_text)
                labels = bib_targets[bib_id].get("labels") or []
                label = labels[0] if labels else a.get_text(" ", strip=True)
                idx = 0
                for i, s in enumerate(sentences):
                    if marker in s:
                        idx = i
                        sentences[i] = s.replace(marker, label or "citation")
                        break
                else:
                    if label:
                        label_re = build_label_regex(label)
                        for i, s in enumerate(sentences):
                            if label_re.search(normalize_for_match(s)):
                                idx = i
                                break

                key = (
                    base.get("arxiv_id"),
                    bib_id,
                    sentences[idx] if sentences else "",
                )
                if key in seen:
                    continue
                seen.add(key)

                sentence_text = sentences[idx] if sentences else ""
                window_text = _window_text(sentences, idx, window=1)
                explicit_refs = extract_refs(sentence_text)
                explicit_ref_source = "cite_sentence" if explicit_refs else None
                if not explicit_refs:
                    explicit_refs = extract_refs(window_text)
                    explicit_ref_source = "window" if explicit_refs else None
                if not explicit_refs:
                    explicit_refs = extract_refs(para_text)
                    explicit_ref_source = "paragraph" if explicit_refs else None
                mentions.append(
                    {
                        **base,
                        "match_text": label or "citation",
                        "location_type": "body_citation",
                        "section_title": section,
                        "context_prev": sentences[idx - 1] if idx - 1 >= 0 else None,
                        "context_sentence": sentences[idx] if sentences else para_text,
                        "context_next": (
                            sentences[idx + 1] if idx + 1 < len(sentences) else None
                        ),
                        "context_paragraph": para_text,
                        "context_html": context_html,
                        "source": "ar5iv",
                        "source_url": source_url,
                        "cite_target": bib_id,
                        "cite_label": label or None,
                        "bib_entry": bib_targets[bib_id].get("text"),
                        "target_match_status": bib_targets[bib_id].get(
                            "target_match_status"
                        ),
                        "explicit_refs": explicit_refs,
                        "explicit_ref_source": explicit_ref_source,
                        "reference_precision": (
                            "explicit" if explicit_refs else "implicit"
                        ),
                    }
                )

        return mentions

    def extract_from_pdf(
        self,
        pdf_path: str,
        source_url: str,
        base: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        text = pdf_extract_text(pdf_path) or ""
        text = text.replace("\x0c", "\n")

        lower = text.lower()
        bib_idx = None
        for term in ["references", "bibliography"]:
            idx = lower.find(term)
            if idx != -1 and (bib_idx is None or idx < bib_idx):
                bib_idx = idx

        body_text = text if bib_idx is None else text[:bib_idx]
        bib_text = "" if bib_idx is None else text[bib_idx:]

        labels: List[str] = []
        target_match_status = "unknown"
        if bib_text and self.target_profile.title:
            entry_start_re = re.compile(r"^\s*(?:\[[^\]]+\]|\([^\)]+\)|\d+\.)\s+")
            author_year_start_re = re.compile(
                r"^\s*[A-Z][A-Za-z'`-]+(?:,|\s)\s+.*\b(19|20)\d{2}\b"
            )
            entries: List[str] = []
            current: List[str] = []
            for line in bib_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if entry_start_re.match(line) or author_year_start_re.match(line):
                    if current:
                        entries.append(" ".join(current))
                    current = [line]
                else:
                    if current:
                        current.append(line)
            if current:
                entries.append(" ".join(current))

            for entry in entries:
                match_status = classify_bib_entry(entry, self.target_profile)
                if match_status in {"exact_target", "same_work_alt_version"}:
                    if not self.target_profile.doi and not title_matches_entry(
                        entry, self.target_profile.title
                    ):
                        continue
                    labels.extend(derive_labels_from_entry(entry))
                    target_match_status = match_status

        mentions: List[Dict[str, Any]] = []
        for m in re.finditer(r"\S.*?(?:\n{2,}|\Z)", body_text, flags=re.S):
            para = m.group(0).strip()
            if not para:
                continue
            mentions.extend(
                self.extract_from_paragraph(
                    text=para,
                    section_title=None,
                    location_type="body",
                    source="pdf",
                    source_url=source_url,
                    base=base,
                    labels=labels,
                )
            )
        for mention in mentions:
            mention["target_match_status"] = target_match_status

        return mentions
