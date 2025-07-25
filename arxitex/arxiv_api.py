import requests
import time
import xml.etree.ElementTree as ET
from loguru import logger
from urllib.parse import urlparse


class ArxivAPI:
    """Handles communication with the ArXiv API"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ArxivConjectureScraper/1.0 (For academic research)',
        })
        self.ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
        }
    
    def fetch_papers(self, search_query, start=0, batch_size=100):
        """Fetch papers from ArXiv API"""
        logger.info(f"Fetching papers with query: {search_query} (start={start}, count={batch_size})")

        params = {
            'search_query': search_query,
            'start': start,
            'max_results': batch_size,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        # Add timeout and retry mechanism for API robustness
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()  # Raise exception for HTTP errors
                break
            except (requests.RequestException, requests.Timeout) as e:
                retry_count += 1
                wait_time = 2 * retry_count  # Exponential backoff
                logger.warning(f"Error fetching papers (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Failed to fetch papers after multiple attempts")
                    return None, 0

        return response.text if response.status_code == 200 else None, 0
    
    def parse_response(self, response_text):
        """Parse XML response from ArXiv API"""
        try:
            root = ET.fromstring(response_text)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            logger.debug(f"Response content: {response_text[:500]}...")  # Log start of response
            return 0, 0, []

        total_results_elem = root.find('.//opensearch:totalResults', self.ns)
        if total_results_elem is None:
            logger.error("Could not find totalResults element in response")
            return 0, 0, []
            
        total_results = int(total_results_elem.text)
        if total_results == 0:
            logger.warning("ArXiv API returned 0 total results. Check your search query.")
            return 0, 0, []

        entries = root.findall('.//atom:entry', self.ns)
        if not entries:
            logger.warning(f"No entries found in response despite total_results={total_results}")
            return 0, total_results, []
            
        return len(entries), total_results, entries
    
    def extract_arxiv_id(self, id_url):
        """Extract ArXiv ID from a URL or ID string"""
        if 'arxiv.org' in id_url or '/abs/' in id_url:
            return id_url.split('/abs/')[-1]
        else:
            # For cases where the ID might be in different format
            parsed = urlparse(id_url)
            path = parsed.path
            return path.split('/')[-1]
    
    def entry_to_paper(self, entry):
        """Convert an ArXiv API entry to a paper dictionary"""
        id_elem = entry.find('atom:id', self.ns)
        if id_elem is None:
            logger.warning("Entry missing ID element, skipping")
            return None
            
        arxiv_id = self.extract_arxiv_id(id_elem.text)
        
        title_elem = entry.find('atom:title', self.ns)
        abstract_elem = entry.find('atom:summary', self.ns)
        
        if title_elem is None or abstract_elem is None:
            logger.warning(f"Paper {arxiv_id} missing title or abstract, skipping")
            return None
            
        author_elements = entry.findall('atom:author/atom:name', self.ns)
        authors = [name.text.strip() for name in author_elements]
        title = title_elem.text.replace('\n', ' ').strip()
        abstract = abstract_elem.text.replace('\n', ' ').strip()        

        paper = {
            'id': id_elem.text,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            'source_url': f"https://arxiv.org/e-print/{arxiv_id}",
            'arxiv_id': arxiv_id
        }

        authors = entry.findall('atom:author/atom:name', self.ns)
        paper['authors'] = [author.text for author in authors if author.text]

        return paper