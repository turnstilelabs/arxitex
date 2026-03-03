from arxitex.arxiv_utils import extract_arxiv_id_from_urls, parse_arxiv_id


def test_parse_arxiv_id_variants():
    assert parse_arxiv_id("https://arxiv.org/abs/1901.01234v2") == "1901.01234"
    assert parse_arxiv_id("1901.01234") == "1901.01234"
    assert parse_arxiv_id("http://arxiv.org/abs/math.AG/0601001") == "math.AG/0601001"


def test_extract_arxiv_id_from_urls():
    urls = [
        "https://example.com/paper.pdf",
        "https://arxiv.org/abs/1901.01234v2",
        "https://arxiv.org/pdf/1901.01234.pdf",
    ]
    assert extract_arxiv_id_from_urls(urls) == "1901.01234"
