
import re
from pathlib import Path
from typing import Dict, List
from loguru import logger
from arxitex.extractor.utils import ArtifactNode, Citation


class CitationEnricher:
    """
    A self-contained helper class responsible for finding, parsing, and attaching
    bibliography information to a list of artifacts.
    """
    def enrich(self, project_dir: Path, nodes: List[ArtifactNode]):
        """
        Main entry point. Orchestrates the entire citation enrichment process.
        This method mutates the `nodes` list.
        """
        logger.info("Starting citation enrichment process...")
        # Step 1: Find and parse the bibliography to create the lookup map.
        bib_map = self._find_and_parse_bibliography(project_dir)
        
        # Step 2: Use the map to attach citation data to the nodes.
        self._attach_citations_to_nodes(nodes, bib_map)
        logger.info("Citation enrichment complete.")

    def _find_and_parse_bibliography(self, project_dir: Path) -> Dict[str, Dict]:
        """Finds and parses bibliography, prioritizing .bbl files."""
        bbl_files = list(project_dir.rglob('*.bbl'))
        if bbl_files:
            logger.info(f"Found .bbl file: {bbl_files[0].name}. Parsing it.")
            bbl_content = bbl_files[0].read_text(encoding='utf-8', errors='ignore')
            return self._parse_bbl_content(bbl_content)

        bib_files = list(project_dir.rglob('*.bib'))
        if bib_files:
            logger.info(f"No .bbl file found. Found .bib file: {bib_files[0].name}. Parsing it.")
            bib_content = bib_files[0].read_text(encoding='utf-8', errors='ignore')
            return self._parse_bib_content(bib_content)
        
        logger.warning("No .bbl or .bib file found. Cannot parse bibliography.")
        return {}

    def _parse_bbl_content(self, bbl_content: str) -> Dict[str, Dict]:
        bib_map = {}
        pattern = re.compile(r'\\bibitem\{(.*?)\}(.*?)(?=\\bibitem|\s*\\end)', re.DOTALL)
        for match in pattern.finditer(bbl_content):
            cite_key, ref_text = match.groups()
            ref_text = re.sub(r'\s+', ' ', ref_text).strip()
            arxiv_match = re.search(r'(?:arxiv[:\s]*|eprint\s*=\s*\{s*)([\d\.\/v-]+)', ref_text, re.IGNORECASE)
            bib_map[cite_key.strip()] = { "full_reference": ref_text, "arxiv_id": arxiv_match.group(1).strip() if arxiv_match else None }
        return bib_map

    def _parse_bib_content(self, bib_content: str) -> Dict[str, Dict]:
        bib_map = {}
        # Improved regex to handle entries at the end of the file.
        pattern = re.compile(r'@\w+\s*\{(.*?),(.*?)(?=\n@|\Z)', re.DOTALL)
        for match in pattern.finditer(bib_content):
            cite_key, fields_str = match.groups()
            ref_text = f"{cite_key}: {fields_str.strip()}"
            arxiv_match = re.search(r'(?:archivePrefix|eprint)\s*=\s*.*?([\d\.\/v-]+)', fields_str)
            bib_map[cite_key.strip()] = { "full_reference": ref_text, "arxiv_id": arxiv_match.group(1).strip() if arxiv_match else None }
        return bib_map
    
    def _attach_citations_to_nodes(self, nodes: List[ArtifactNode], bib_map: Dict[str, Dict]):
        """Attaches citation data to nodes based on the bibliography map."""
        if not bib_map:
            return

        cite_pattern = re.compile(r'\\cite[pt]?\*?(?:\[([^\]]*)\])?\{([^}]+)\}')
        for node in nodes:
            full_content = node.content + (node.proof or "")
            for match in cite_pattern.finditer(full_content):
                note, cite_keys_str = match.groups()
                cite_keys = [key.strip() for key in cite_keys_str.split(',')]
                for key in cite_keys:
                    if key in bib_map and not any(c.cite_key == key for c in node.citations):
                        bib_entry = bib_map[key]
                        node.citations.append(
                            Citation(
                                cite_key=key,
                                full_reference=bib_entry["full_reference"],
                                arxiv_id=bib_entry["arxiv_id"],
                                note=note.strip() if note else None
                            )
                        )