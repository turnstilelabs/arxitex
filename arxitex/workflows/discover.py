import asyncio

from loguru import logger

from arxitex.workflows.runner import AsyncArxivWorkflowRunner


class DiscoveryWorkflow(AsyncArxivWorkflowRunner):
    """
    Finds new paper IDs from ArXiv by querying the API and adds them to the
    DiscoveryIndex queue for later processing.
    """

    async def _process_single_item(self, item: dict) -> dict:
        arxiv_id = item.get("arxiv_id")
        if not arxiv_id:
            return {"status": "failure", "reason": "item_missing_arxiv_id"}

        newly_added_count = await asyncio.to_thread(
            self.components.discovery_index.add_papers, [item]
        )

        if newly_added_count > 0:
            logger.debug(f"Added new ID to discovery queue: {arxiv_id}")
            return {
                "status": "success",
                "arxiv_id": arxiv_id,
                "action": "added_to_queue",
            }
        else:
            logger.debug(f"{arxiv_id} was already in the discovery queue.")
            return {
                "status": "skipped",
                "arxiv_id": arxiv_id,
                "reason": "already_in_queue",
            }
