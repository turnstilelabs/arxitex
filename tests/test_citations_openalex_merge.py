import asyncio

import httpx

from arxitex.tools.citations_openalex import fetch_openalex_citation


class DummyResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.request = httpx.Request("GET", "https://api.openalex.org/works")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status={self.status_code}", request=self.request, response=None
            )


def test_fetch_openalex_citation_picks_max_cited_by_count_for_same_title():
    """If OpenAlex returns both arXiv (0 cites) and journal (8 cites) works for the
    same paper, we should store 8.
    """

    title = "Recovery of a distributed order fractional derivative in an unknown medium"
    authors = ["Bangti Jin", "Yavar Kian"]

    # Simulate OpenAlex search response with two near-identical title matches.
    arxiv_work = {
        "id": "https://openalex.org/W_ARXIV",
        "title": title,
        "cited_by_count": 0,
        "authorships": [
            {"author": {"display_name": "Bangti Jin"}},
            {"author": {"display_name": "Yavar Kian"}},
        ],
    }
    journal_work = {
        "id": "https://openalex.org/W_JOURNAL",
        "title": title,
        "cited_by_count": 8,
        "authorships": [
            {"author": {"display_name": "Jin, Bangti"}},
            {"author": {"display_name": "Kian, Yavar"}},
        ],
    }

    payload = {"results": [arxiv_work, journal_work]}

    async def fake_get(self, url, params=None):
        assert url == "https://api.openalex.org/works"
        # Ensure we are searching by title.
        assert params and "search" in params
        return DummyResp(200, payload)

    client = type("C", (), {"get": fake_get})()

    rec = asyncio.run(
        fetch_openalex_citation(
            client,
            base_arxiv_id="2207.12929",
            title=title,
            authors=authors,
            mailto=None,
        )
    )

    assert rec.paper_id == "2207.12929"
    assert rec.citation_count == 8
    assert rec.source_work_id == "https://openalex.org/W_JOURNAL"
    # We no longer store raw OpenAlex JSON in the DB to save space.
