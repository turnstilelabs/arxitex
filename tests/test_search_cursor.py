import xml.etree.ElementTree as ET
from datetime import datetime

from arxitex.search_cursor import BackfillStateManager, SearchCursorManager

ATOM_NS = "http://www.w3.org/2005/Atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
NS = {
    "atom": ATOM_NS,
    "opensearch": OPENSEARCH_NS,
    "arxiv": "http://arxiv.org/schemas/atom",
}


def make_feed_with_published(published_iso: str):
    return f"""<?xml version="1.0"?>
    <feed xmlns="{ATOM_NS}" xmlns:opensearch="{OPENSEARCH_NS}">
        <opensearch:totalResults>1</opensearch:totalResults>
        <entry>
            <id>http://arxiv.org/abs/0000.00000v1</id>
            <title>Title</title>
            <published>{published_iso}</published>
        </entry>
    </feed>
    """


def test_get_query_with_no_cursor(tmp_path):
    scm = SearchCursorManager(str(tmp_path))
    q = "cat:math.GR"
    assert scm.get_query_with_cursor(q) == q


def test_update_cursor_and_get_query_with_cursor(tmp_path):
    scm = SearchCursorManager(str(tmp_path))
    q = "cat:math.GR"

    # Create a feed with a known published timestamp and parse entries
    published = "2020-01-02T03:04:05Z"
    feed = make_feed_with_published(published)
    root = ET.fromstring(feed)
    entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    # Update cursor with the entries
    scm.update_cursor(q, entries, NS)

    # Now retrieving the query should add a date filter ending with the published time in arXiv format
    modified = scm.get_query_with_cursor(q)
    assert "submittedDate" in modified
    # The published time should be formatted as YYYYMMDDHHMMSS inside the query
    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    expected = dt.strftime("%Y%m%d%H%M%S")
    assert expected in modified


def test_backfill_state_get_next_and_complete(tmp_path):
    bm = BackfillStateManager(str(tmp_path))
    q = "cat:math.GR"

    year, month = bm.get_next_interval(q)
    # Should return valid year/month integers
    assert isinstance(year, int) and isinstance(month, int)

    # Complete the returned interval and ensure state moves back one month
    original = (year, month)
    bm.complete_interval(q, year, month)
    new_year, new_month = bm.get_next_interval(q)

    # Calculate expected previous month
    prev_month = original[1] - 1
    prev_year = original[0]
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1

    assert (new_year, new_month) == (prev_year, prev_month)
