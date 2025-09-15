
import re
from pathlib import Path
from typing import Dict, List
from loguru import logger
from arxitex.extractor.utils import ArtifactNode, Reference, ReferenceType

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
        bib_map = self._find_and_parse_bibliography(project_dir)
        
        if bib_map:
            self._attach_citations_to_nodes(nodes, bib_map)
            logger.info("Citation enrichment complete.")

    def _find_and_parse_bibliography(self, project_dir: Path) -> Dict[str, Dict]:
        """
        Finds and parses ALL bibliography files in the project, prioritizing .bbl files
        and merging the contents of all found files.
        """
        # --- STRATEGY 1: Look for .bbl files first ---
        bbl_files = list(project_dir.rglob('*.bbl'))
        if bbl_files:
            logger.info(f"Found {len(bbl_files)} .bbl file(s). Parsing all of them.")
            final_bib_map = {}
            for bbl_file in bbl_files:
                try:
                    bbl_content = bbl_file.read_text(encoding='utf-8', errors='ignore')
                    parsed_map = self._parse_bbl_content(bbl_content)
                    final_bib_map.update(parsed_map)
                except Exception as e:
                    logger.warning(f"Could not parse .bbl file {bbl_file.name}: {e}")
            return final_bib_map

        # --- STRATEGY 2: Fallback to .bib files ---
        bib_files = list(project_dir.rglob('*.bib'))
        if bib_files:
            logger.info(f"No .bbl files found. Found {len(bib_files)} .bib file(s). Parsing all of them.")
            final_bib_map = {}
            for bib_file in bib_files:
                try:
                    bib_content = bib_file.read_text(encoding='utf-8', errors='ignore')
                    parsed_map = self._parse_bib_content(bib_content)
                    final_bib_map.update(parsed_map)
                except Exception as e:
                    logger.warning(f"Could not parse .bib file {bib_file.name}: {e}")
            return final_bib_map
        
        logger.warning("No .bbl or .bib files found in the project directory. Cannot parse bibliography.")
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
            logger.warning("Bibliography map is empty. No citations will be attached.")
            return

        cite_pattern = re.compile(r'\\cite[pt]?\*?(?:\[([^\]]*)\])?\{([^}]+)\}')
        for node in nodes:
            full_content = node.content + (node.proof or "")
            for match in cite_pattern.finditer(full_content):
                note, cite_keys_str = match.groups()
                cite_keys = [key.strip() for key in cite_keys_str.split(',')]
                
                for key in cite_keys:
                    if key in bib_map and not any(r.target_id == key for r in node.references if r.reference_type == ReferenceType.EXTERNAL):
                        bib_entry = bib_map[key]
                        node.references.append(
                            Reference(
                                target_id=key,
                                reference_type=ReferenceType.EXTERNAL,
                                full_reference=bib_entry["full_reference"],
                                arxiv_id=bib_entry["arxiv_id"],
                                note=note.strip() if note else None
                                # Context and Position can be extracted here as well if needed
                            )
                        )