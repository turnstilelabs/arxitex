from arxitex.arxiv_utils import normalize_arxiv_id


def test_normalize_arxiv_id_modern():
    assert normalize_arxiv_id("2501.01234v3") == "2501.01234"
    assert normalize_arxiv_id("2501.01234") == "2501.01234"


def test_normalize_arxiv_id_legacy():
    assert normalize_arxiv_id("math.AG/0601001v2") == "math.AG/0601001"
    assert normalize_arxiv_id("math.AG/0601001") == "math.AG/0601001"
