import os

from loguru import logger

from arxitex.downloaders.async_downloader import AsyncSourceDownloader
from arxitex.workflows.runner import ArxivPipelineComponents, AsyncArxivWorkflowRunner


class DownloaderWorkflow(AsyncArxivWorkflowRunner):
    """
    A workflow that finds new papers via the ArXiv API and downloads their source.
    It updates the PaperIndex with status 'downloaded' or 'download_failed'.
    """

    def __init__(
        self,
        components: ArxivPipelineComponents,
        max_concurrent_tasks: int = 20,
        force: bool = False,
    ):
        super().__init__(components, max_concurrent_tasks, force)
        self.source_cache_dir = os.path.join(self.components.output_dir, "source_files")
        os.makedirs(self.source_cache_dir, exist_ok=True)

    async def _process_single_item(self, item: dict) -> dict:
        arxiv_id = item["arxiv_id"]

        try:
            async with AsyncSourceDownloader(
                cache_dir=self.persistent_source_dir
            ) as downloader:
                extracted_path = await downloader.download_and_extract_source(arxiv_id)

            self.components.processing_index.update_processed_papers_status(
                arxiv_id, status="downloaded", source_path=str(extracted_path)
            )

            logger.info(
                f"SUCCESS: Downloaded source for {arxiv_id} to {extracted_path}"
            )

            return {
                "status": "success",
                "arxiv_id": arxiv_id,
                "source_path": str(extracted_path),
            }
        except Exception as e:
            self.components.processing_index.update_processed_papers_status(
                arxiv_id, status="download_failed", reason=str(e)
            )
            raise e


# python run_downloader.py --query "cat:cs.AI AND all:conjecture" --max-papers 100
