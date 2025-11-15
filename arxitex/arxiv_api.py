import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from loguru import logger


class ArxivAPI:
    """Handles communication with the ArXiv API"""

    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ArxivConjectureScraper/1.0 (For academic research)",
            }
        )
        self.ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

    def fetch_papers(self, search_query, start=0, batch_size=100):
        """Fetch papers from ArXiv API"""
        logger.info(
            f"Fetching papers with query: {search_query} (start={start}, count={batch_size})"
        )

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                break
            except (requests.RequestException, requests.Timeout) as e:
                retry_count += 1
                wait_time = 2 * retry_count  # Exponential backoff
                logger.warning(
                    f"Error fetching papers (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count < max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Failed to fetch papers after multiple attempts")
                    return None

        return response.text if response.status_code == 200 else None

    def fetch_by_ids(self, ids):
        """Fetch specific papers by arXiv id_list (comma-separated)."""
        if not ids:
            return None
        logger.info(f"Fetching by id_list with {len(ids)} ids")

        params = {
            "id_list": ",".join(ids),
        }

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                break
            except (requests.RequestException, requests.Timeout) as e:
                retry_count += 1
                wait_time = 2 * retry_count  # Exponential backoff
                logger.warning(
                    f"Error fetching id_list (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count < max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Failed to fetch id_list after multiple attempts")
                    return None

        return response.text if response.status_code == 200 else None

    def parse_response(self, response_text):
        """Parse XML response from ArXiv API"""
        if not response_text:
            logger.warning("Received empty response text from API.")
            return 0, 0, []

        try:
            root = ET.fromstring(response_text)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            logger.debug(f"Response content: {response_text[:500]}...")
            return 0, 0, []

        error_entry = root.find(".//atom:entry[atom:title='Error']", self.ns)
        if error_entry is not None:
            error_summary = error_entry.find(".//atom:summary", self.ns)
            if error_summary is not None and "start_index" in error_summary.text:
                logger.warning(
                    f"ArXiv API returned a pagination limit error: {error_summary.text}"
                )
                return 0, 0, []

        total_results_elem = root.find(".//opensearch:totalResults", self.ns)
        if total_results_elem is None:
            logger.error("Could not find totalResults element in response")
            return 0, 0, []

        total_results = int(total_results_elem.text)
        if total_results == 0:
            logger.warning(
                "ArXiv API returned 0 total results. Check your search query."
            )
            return 0, 0, []

        entries = root.findall(".//atom:entry", self.ns)
        if not entries:
            logger.info(
                f"Pagination complete: No entries found in response despite total_results={total_results}"
            )
            return 0, total_results, []

        return len(entries), total_results, entries

    def extract_arxiv_id(self, id_url):
        """Extract ArXiv ID from a URL or ID string"""
        if "arxiv.org" in id_url or "/abs/" in id_url:
            return id_url.split("/abs/")[-1]
        else:
            # For cases where the ID might be in different format
            parsed = urlparse(id_url)
            path = parsed.path
            return path.split("/")[-1]

    def entry_to_paper(self, entry):
        """Convert an ArXiv API entry to a paper dictionary"""
        id_elem = entry.find("atom:id", self.ns)
        if id_elem is None:
            logger.warning("Entry missing ID element, skipping")
            return None

        arxiv_id = self.extract_arxiv_id(id_elem.text)

        title_elem = entry.find("atom:title", self.ns)
        abstract_elem = entry.find("atom:summary", self.ns)

        if title_elem is None or abstract_elem is None:
            logger.warning(f"Paper {arxiv_id} missing title or abstract, skipping")
            return None

        author_elements = entry.findall("atom:author/atom:name", self.ns)
        authors = [name.text.strip() for name in author_elements]
        title = title_elem.text.replace("\n", " ").strip()
        abstract = abstract_elem.text.replace("\n", " ").strip()

        primary_category = "unknown"
        all_categories = []

        ARXIV_CATEGORY_SCHEME = "http://arxiv.org/schemas/atom"
        CATEGORY_PATTERN = re.compile(r"^[a-z-]+(\.[A-Z]{2,}|-[a-zA-Z]{2,})$")
        all_cat_elems = entry.findall(".//atom:category", self.ns)
        for cat_elem in all_cat_elems:
            term = cat_elem.get("term")
            scheme = cat_elem.get("scheme")

            if (
                term
                and scheme == ARXIV_CATEGORY_SCHEME
                and CATEGORY_PATTERN.match(term)
            ):
                all_categories.append(term)

        primary_cat_elem = entry.find(".//arxiv:primary_category", self.ns)
        if primary_cat_elem is not None and primary_cat_elem.get("term"):
            primary_category = primary_cat_elem.get("term")
        elif all_categories:
            primary_category = all_categories[0]

        comment_elem = entry.find("arxiv:comment", self.ns)
        comment = (
            comment_elem.text.strip()
            if comment_elem is not None and comment_elem.text
            else None
        )

        paper = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "comment": comment,
            "primary_category": primary_category,
            "all_categories": all_categories,
        }

        return paper

    def close(self):
        logger.info("Closing ArxivAPI session...")
        self.session.close()
