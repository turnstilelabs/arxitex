import json

from arxitex.arxiv_api import ArxivAPI
from arxitex.tools.external_reference_arxiv_matcher import (
    extract_title_and_authors,
    match_external_reference_to_arxiv,
)


def make_feed(arxiv_id: str, title: str, authors: list[str]) -> str:
    auth_xml = "\n".join([f"<author><name>{a}</name></author>" for a in authors])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <title>{title}</title>
    <summary>Abstract</summary>
    {auth_xml}
    <category term="math.GR" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="math.GR"/>
  </entry>
</feed>"""


def test_extract_title_and_authors_quotes():
    ref = 'J. Doe, "A Great Paper", Journal of Things, 2021.'
    title, authors = extract_title_and_authors(ref)
    assert title == "A Great Paper"
    assert authors and "Doe" in authors[0]


def test_extract_title_and_authors_emph():
    ref = r"J. Doe and A. Roe, \emph{A Great Paper}, 2021."
    title, authors = extract_title_and_authors(ref)
    assert title == "A Great Paper"
    assert len(authors) >= 1


def test_matcher_direct_regex_fast_path():
    api = ArxivAPI()
    ref = "J. Doe, Some paper, arXiv:1234.5678v2"
    res = match_external_reference_to_arxiv(api=api, full_reference=ref)
    assert res.match_method == "direct_regex"
    assert res.matched_arxiv_id == "1234.5678"


def test_matcher_search_picks_best(monkeypatch, tmp_path):
    api = ArxivAPI()
    # Two entries: one exact title, one far.
    feed_good = make_feed("1111.2222v1", "A Great Paper", ["John Doe", "Alice Roe"])

    def fake_fetch(search_query, start=0, batch_size=100):
        assert 'ti:"A Great Paper"' in search_query
        return feed_good

    monkeypatch.setattr(api, "fetch_papers", fake_fetch)

    db_path = tmp_path / "t.sqlite"
    # Use cache tables (created by ensure_schema) to exercise codepaths.
    res = match_external_reference_to_arxiv(
        api=api,
        full_reference='J. Doe, "A Great Paper", 2021.',
        db_path_for_cache=str(db_path),
        refresh_days=30,
    )
    assert res.match_method == "search"
    assert res.matched_arxiv_id == "1111.2222"
    assert (res.title_score or 0) > 0.95
    assert json.dumps(res.matched_authors)


def test_matcher_cache_avoids_requery(monkeypatch, tmp_path):
    api = ArxivAPI()
    feed = make_feed("1111.2222v1", "A Great Paper", ["John Doe"])
    calls = {"n": 0}

    def fake_fetch(search_query, start=0, batch_size=100):
        calls["n"] += 1
        return feed

    monkeypatch.setattr(api, "fetch_papers", fake_fetch)
    db_path = tmp_path / "t.sqlite"

    ref = 'J. Doe, "A Great Paper", 2021.'
    r1 = match_external_reference_to_arxiv(
        api=api, full_reference=ref, db_path_for_cache=str(db_path)
    )
    r2 = match_external_reference_to_arxiv(
        api=api, full_reference=ref, db_path_for_cache=str(db_path)
    )
    assert r1.matched_arxiv_id == r2.matched_arxiv_id
    assert calls["n"] == 1
