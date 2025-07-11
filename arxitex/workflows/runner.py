
import abc
import asyncio
import os
from loguru import logger
import json
from datetime import datetime, timezone
from arxitex.paper_index import PaperIndex
from arxitex.arxiv_api import ArxivAPI


class ArxivPipelineComponents:
    """A container for shared services used in ArXiv processing pipelines."""
    def __init__(self, output_dir="output"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.arxiv_api = ArxivAPI()
        self.paper_index = PaperIndex(self.output_dir)

class AsyncArxivWorkflowRunner(abc.ABC):
    """Abstract base class for running an ASYNC processing workflow on ArXiv papers."""

    def __init__(self, components: ArxivPipelineComponents, max_concurrent_tasks: int = 10, force: bool = False):
        self.components = components
        self.max_concurrent_tasks = max_concurrent_tasks
        self.force = force
        self.results = []

    def _get_papers_to_process(self, search_query: str, start: int, batch_size: int):
        """Fetches papers and filters out those already processed, with logging."""
        response_text, _ = self.components.arxiv_api.fetch_papers(search_query, start, batch_size)
        if not response_text: return None

        _, _, entries = self.components.arxiv_api.parse_response(response_text)
        if not entries: return []

        papers_to_process = []
        for entry in entries:
            paper = self.components.arxiv_api.entry_to_paper(entry)
            if not paper: continue

            paper_id = paper['arxiv_id']
            if self.components.paper_index.is_paper_processed(paper_id):
                if self.force:
                    logger.warning(f"Re-processing {paper_id} due to --force flag.")
                    papers_to_process.append(paper)
                else:
                    logger.info(f"Skipping {paper_id}: already processed. Use --force to override.")
            else:
                papers_to_process.append(paper)
        
        logger.info(f"Found {len(papers_to_process)} new papers to process in this batch.")
        return papers_to_process

    async def run(self, search_query: str, max_papers: int, batch_size: int = 20):
        """Main entry point to run the async workflow"""
        session_success_count = 0
        start_index = 0
        
        total_in_index = len(self.components.paper_index.processed_papers)
        logger.info(f"Starting async workflow '{self.__class__.__name__}'.")
        logger.info(f"Goal for this run: Successfully process {max_papers} new papers.")
        logger.info(f"Persistent index already contains {total_in_index} processed papers.")
        
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        while session_success_count < max_papers:
            papers = self._get_papers_to_process(search_query, start_index, batch_size)
            if papers is None:
                logger.warning("API did not return papers. Ending run.")
                break
            if not papers:
                logger.info("No more new papers found in query range. Ending run.")
                break

            tasks = [self._process_and_handle_paper(paper, semaphore) for paper in papers]
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
            
            start_index += len(papers)
            
        logger.info(f"Workflow finished. Processed {session_success_count} new papers successfully in this run.")
        logger.info("Writing summary report for this run...")
        self._write_summary_report()

    async def _process_and_handle_paper(self, paper: dict, semaphore: asyncio.Semaphore):
        """A wrapper to manage concurrency and exceptions for single paper processing."""
        paper_id = paper.get('arxiv_id', 'unknown')
        async with semaphore:
            try:
                return await self._process_single_paper(paper)
            except Exception as e:
                reason = f"Unhandled exception in workflow: {str(e)}"
                logger.error(f"UNHANDLED_FAILURE for {paper_id}: {reason}", exc_info=True)
                self.components.paper_index.update_processed_papers_index(paper_id, status='failure', reason=reason)
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

    @abc.abstractmethod
    async def _process_single_paper(self, paper: dict) -> dict:
        """Process a single paper. Must return a dictionary with status and details."""
        pass
