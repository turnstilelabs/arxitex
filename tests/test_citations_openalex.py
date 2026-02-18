from arxitex.tools.citations.openalex import strip_arxiv_version


def test_strip_arxiv_version_modern():
    assert strip_arxiv_version("2501.01234v3") == "2501.01234"
    assert strip_arxiv_version("2501.01234") == "2501.01234"


def test_strip_arxiv_version_legacy():
    assert strip_arxiv_version("math.AG/0601001v2") == "math.AG/0601001"
    assert strip_arxiv_version("math.AG/0601001") == "math.AG/0601001"
