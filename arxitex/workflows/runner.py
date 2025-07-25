
import abc
import asyncio
import os
from loguru import logger
import json
from datetime import datetime, timezone
from arxitex.paper_index import PaperIndex
from arxitex.discover_index import DiscoveryIndex
from arxitex.arxiv_api import ArxivAPI
from arxitex.search_cursor import SearchCursorManager


class ArxivPipelineComponents:
    """A container for shared services used in ArXiv processing pipelines."""
    def __init__(self, output_dir="output"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.arxiv_api = ArxivAPI()
        self.search_cursors = SearchCursorManager(self.output_dir)
        self.discovery_index = DiscoveryIndex(self.output_dir)
        self.processing_index = PaperIndex(self.output_dir)
        
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

    def _get_papers_to_process(self, search_query: str, start: int, batch_size: int):
        """Fetches papers using the date cursor and filters out those already processed."""
        query_to_run = search_query
        if start == 0:
            query_to_run = self.components.search_cursors.get_query_with_cursor(search_query)

        response_text, _ = self.components.arxiv_api.fetch_papers(query_to_run, start, batch_size)
        if not response_text: return None

        _, _, entries = self.components.arxiv_api.parse_response(response_text)
        if not entries: return ([], False)

        self.components.search_cursors.update_cursor(
            search_query, entries, self.components.arxiv_api.ns
        )

        papers_to_process = []
        for entry in entries:
            paper = self.components.arxiv_api.entry_to_paper(entry)
            if not paper:
                continue

            paper_id = paper['arxiv_id']
            if self.components.processing_index.is_paper_processed(paper_id):
                if self.force:
                    logger.warning(f"Re-processing {paper_id} due to --force flag.")
                    papers_to_process.append(paper)
                else:
                    logger.info(f"Skipping {paper_id}: already in processing index. Use --force to override.")
            else:
                papers_to_process.append(paper)

        logger.info(f"Found {len(papers_to_process)} new papers to consider in this batch of {len(entries)}.")
        return (papers_to_process, True)

    async def run(self, search_query: str, max_papers: int, batch_size: int = 20):
        """Main entry point to run the async workflow"""
        session_success_count = 0
        start_index = 0
        
        total_in_index = len(self.components.processing_index.processed_papers)
        logger.info(f"Starting async workflow '{self.__class__.__name__}'.")
        logger.info(f"Goal for this run: Successfully process {max_papers} new papers.")
        logger.info(f"Persistent processing index already contains {total_in_index} entries.")
        
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        while session_success_count < max_papers:
            result_tuple = self._get_papers_to_process(search_query, start_index, batch_size)
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
                self.components.processing_index.update_processed_papers_index(paper_id, status='failure', reason=reason)
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

