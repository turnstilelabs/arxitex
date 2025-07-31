
import abc
import asyncio
import os
import re
from loguru import logger
from typing import Optional, Tuple, List, Dict
import json
from datetime import datetime, timezone
from arxitex.indices.discover import DiscoveryIndex
from arxitex.indices.processed import ProcessedIndex
from arxitex.indices.skipped import SkippedIndex
from arxitex.arxiv_api import ArxivAPI
from arxitex.search_cursor import SearchCursorManager


class ArxivPipelineComponents:
    """A container for shared services used in ArXiv processing pipelines."""
    def __init__(self, output_dir="output"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.db_path = os.path.join(self.output_dir, "arxitex_indices.db")

        self.arxiv_api = ArxivAPI()
        self.search_cursors = SearchCursorManager(self.output_dir)
        self.discovery_index = DiscoveryIndex(self.db_path)
        self.processing_index = ProcessedIndex(self.db_path)
        self.skipped_index = SkippedIndex(self.db_path)
        
class AsyncWorkflowRunnerBase(abc.ABC):
    """
    A generic abstract base class for any asynchronous workflow.
    """
    def __init__(self, components: ArxivPipelineComponents, max_concurrent_tasks: int = 10, **kwargs):
        self.components = components
        self.max_concurrent_tasks = max_concurrent_tasks
        self.force = kwargs.get('force', False) 
        self.results = []

    async def _process_and_handle_paper(self, paper: dict, semaphore: asyncio.Semaphore) -> dict:
        """
        A generic wrapper to manage concurrency and exceptions for single item processing.
        """
        item_id = paper.get('arxiv_id')
        async with semaphore:
            try:
                return await self._process_single_item(paper)
            except Exception as e:
                reason = f"Unhandled exception in workflow for item {item_id}: {e}"
                logger.error(reason, exc_info=True)
                return {"status": "failure", "id": item_id, "reason": reason}

    def _write_summary_report(self):
        """
        Categorizes results from the current run and writes a summary report.
        This method is shared by all workflows.
        """
        summary = {
            "workflow_name": self.__class__.__name__,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "papers_attempted_in_this_run": len(self.results),
            "successful_in_this_run": [r for r in self.results if r and r.get('status') == 'success'],
            "failed_in_this_run": [r for r in self.results if r and r.get('status') == 'failure']
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_filename = f"summary_{self.__class__.__name__.lower()}_{timestamp}.json"
        report_path = os.path.join(self.components.output_dir, report_filename)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Successfully wrote this run's summary report to {report_path}")
        except Exception as e:
            logger.error(f"Could not write summary report: {e}")

    @abc.abstractmethod
    async def run(self, **kwargs):
        """The main entry point for the workflow. Must be implemented by subclasses."""
        pass

    @abc.abstractmethod
    async def _process_single_item(self, item: dict) -> dict:
        """Processes a single item (e.g., a paper). Must be implemented by subclasses."""
        pass
    
class AsyncArxivWorkflowRunner(AsyncWorkflowRunnerBase):
    """A workflow runner specifically for fetching and processing papers from the ArXiv API."""

    def __init__(self, components: ArxivPipelineComponents, **kwargs):
        super().__init__(components, **kwargs)
        
        self.max_pages = kwargs.get('max_pages', 50)
        self.title_exclude_keywords = kwargs.get('title_exclude_keywords', ["lecture", "course", "notes on", "introduction to"])
        logger.info(f"Workflow initialized with filtering: max_pages={self.max_pages}, exclude_keywords={self.title_exclude_keywords}")
        
    def _is_title_disqualified(self, title: str) -> Optional[str]:
        """Checks if the title contains any disqualifying keywords. Returns the keyword if found."""
        lower_title = title.lower()
        for keyword in self.title_exclude_keywords:
            if keyword.lower() in lower_title:
                return keyword
        return None

    def _is_page_count_excessive(self, comment: Optional[str]) -> Optional[int]:
        """
        Parses the page count from the comment string using a robust regex.
        Returns the page count *only if* it exceeds the configured maximum.
        """
        if not comment:
            return None
        
        match = re.search(r'(\d+)\s*pages?', comment, re.IGNORECASE)
        if match:
            try:
                page_count = int(match.group(1))
                if page_count > self.max_pages:
                    return page_count
            except (ValueError, IndexError):
                logger.warning(f"Regex found a non-integer page count in comment: '{comment}'")
                return None
        return None
    
    def _fetch_and_filter_batch_sync(self, search_query: str, start: int, batch_size: int) -> Optional[Tuple[List[Dict], bool]]:
        query_to_run = search_query
        if start == 0:
            query_to_run = self.components.search_cursors.get_query_with_cursor(search_query)

        response_text, _ = self.components.arxiv_api.fetch_papers(query_to_run, start, batch_size)
        if not response_text: return None

        _, _, entries = self.components.arxiv_api.parse_response(response_text)
        if not entries: return ([], False)

        self.components.search_cursors.update_cursor(search_query, entries, self.components.arxiv_api.ns)

        papers_to_process = []
        for entry in entries:
            paper = self.components.arxiv_api.entry_to_paper(entry)
            if not paper: continue

            paper_id = paper['arxiv_id']

            if paper_id in self.components.skipped_index and not self.force:
                logger.debug(f"Skipping {paper_id}: already in skipped index.")
                continue
            
            disqualifying_keyword = self._is_title_disqualified(paper['title'])
            if disqualifying_keyword:
                reason = f"skipped_title_keyword: '{disqualifying_keyword}'"
                self.components.skipped_index.add(paper_id, reason)
                continue

            excessive_page_count = self._is_page_count_excessive(paper.get('comment'))
            if excessive_page_count:
                reason = f"skipped_too_long: {excessive_page_count} pages (limit {self.max_pages})"
                self.components.skipped_index.add(paper_id, reason)
                continue

            if self.components.processing_index.is_successfully_processed(paper_id):
                if self.force:
                    papers_to_process.append(paper)
                else:
                    logger.info(f"Skipping {paper_id}: already in processing index. Use --force to override.")
            else:
                papers_to_process.append(paper)

        logger.info(f"Found {len(papers_to_process)} new papers to consider in this batch of {len(entries)}.")
        return (papers_to_process, True)

    async def _get_papers_to_process(self, search_query: str, start: int, batch_size: int):
        return await asyncio.to_thread(
            self._fetch_and_filter_batch_sync, search_query, start, batch_size
        )

    async def run(self, search_query: str, max_papers: int, batch_size: int = 20):
        session_success_count = 0
        start_index = 0
        
        logger.info(f"Starting async workflow '{self.__class__.__name__}'.")
        logger.info(f"Goal for this run: Successfully process {max_papers} new papers.")
        
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        while session_success_count < max_papers:
            result_tuple = await self._get_papers_to_process(search_query, start_index, batch_size)
            if result_tuple is None:
                logger.warning("API did not return papers. Ending run.")
                break
            
            papers_to_process, api_had_entries = result_tuple

            if not api_had_entries:
                logger.info("No more papers found from the API for this query. Ending run.")
                break

            if not papers_to_process:
                logger.info("This batch contained no new papers to process. Fetching next batch.")
                start_index += batch_size
                continue 
            
            tasks = [self._process_and_handle_paper(paper, semaphore) for paper in papers_to_process]
            
            batch_results = await asyncio.gather(*tasks)
            self.results.extend(filter(None, batch_results))

            successes_in_batch = sum(1 for r in batch_results if r and r.get('status') == 'success')
            session_success_count += successes_in_batch

            logger.info(
                f"Batch complete. Successful in this run: {session_success_count}/{max_papers}"
            )

            if session_success_count >= max_papers:
                logger.info(f"Session goal of {max_papers} successful papers met.")
                break

            start_index += batch_size
            
        logger.info(f"Workflow finished. Processed {session_success_count} new papers successfully in this run.")
        logger.info("Writing summary report for this run...")
        self._write_summary_report()

    async def _process_and_handle_paper(self, paper: dict, semaphore: asyncio.Semaphore):
        """A wrapper to manage concurrency and exceptions for single paper processing."""
        paper_id = paper.get('arxiv_id', 'unknown')
        async with semaphore:
            try:
                return await self._process_single_item(paper)
            except Exception as e:
                reason = f"Unhandled exception in workflow: {str(e)}"
                logger.error(f"UNHANDLED_FAILURE for {paper_id}: {reason}", exc_info=True)
                self.components.processing_index.update_processed_papers_status(paper_id, status='failure', reason=reason)
                return {"status": "failure", "arxiv_id": paper_id, "reason": reason}
    
    def _write_summary_report(self):
        """Categorizes results from the current run and writes a summary report."""
        summary = {
            "workflow_name": self.__class__.__name__,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "papers_attempted_in_this_run": len(self.results),
            "successful_in_this_run": [r for r in self.results if r.get('status') == 'success'],
            "failed_in_this_run": [r for r in self.results if r.get('status') == 'failure']
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"workflow_summary_{timestamp}.json"
        report_path = os.path.join(self.components.output_dir, report_filename)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Successfully wrote this run's summary report to {report_path}")
        except Exception as e:
            logger.error(f"Could not write summary report: {e}")

