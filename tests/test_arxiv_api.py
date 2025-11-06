from arxitex.arxiv_api import ArxivAPI


def make_sample_feed():
    return """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry>
        <id>http://arxiv.org/abs/1234.5678v1</id>
        <title>Example Title</title>
        <summary>Abstract text with\nnewline</summary>
        <author><name>Alice</name></author>
        <author><name>Bob</name></author>
        <category term="math.GR" scheme="http://arxiv.org/schemas/atom"/>
        <arxiv:primary_category term="math.GR"/>
        <arxiv:comment>Short comment</arxiv:comment>
        <published>2020-01-02T03:04:05Z</published>
      </entry>
    </feed>
    """


def test_parse_response_and_entry_to_paper():
    api = ArxivAPI()
    feed = make_sample_feed()
    entries_count, total_results, entries = api.parse_response(feed)

    assert entries_count == 1
    assert total_results == 1
    assert len(entries) == 1

    entry = entries[0]
    paper = api.entry_to_paper(entry)
    assert paper is not None
    assert paper["arxiv_id"].endswith("1234.5678v1")
    assert paper["title"] == "Example Title"
    assert "Abstract text" in paper["abstract"]
    assert "Alice" in paper["authors"] and "Bob" in paper["authors"]
    assert paper["primary_category"] == "math.GR"
    assert paper["comment"] == "Short comment"


def test_extract_arxiv_id_variants():
    api = ArxivAPI()
    assert api.extract_arxiv_id("http://arxiv.org/abs/1234.5678v1") == "1234.5678v1"
    assert api.extract_arxiv_id("https://arxiv.org/abs/2101.00001") == "2101.00001"
    # path-only formats
    assert api.extract_arxiv_id("arxiv.org/abs/9999.99999") == "9999.99999"
    # other URL forms
    assert api.extract_arxiv_id("http://example.com/abs/xyz") == "xyz"


def test_parse_response_handles_empty_and_malformed():
    api = ArxivAPI()
    zero_entries = "<feed xmlns='http://www.w3.org/2005/Atom'><opensearch:totalResults xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>0</opensearch:totalResults></feed>"
    cnt, total, entries = api.parse_response(zero_entries)
    assert cnt == 0
    assert total == 0
    assert entries == []

    # malformed xml
    cnt2, total2, entries2 = api.parse_response("<feed><bad></feed>")
    assert cnt2 == 0
    assert total2 == 0
    assert entries2 == []
