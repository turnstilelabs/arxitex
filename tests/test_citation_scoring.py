from arxitex.tools.matching.scoring import (
    author_overlap,
    best_match_index,
    normalize_author,
    normalize_title,
    title_similarity,
)


def test_normalize_title_and_author():
    assert normalize_title("A \\textbf{Test} $x$-Theorem") == "a test theorem"
    assert normalize_author("Doe, John") == "john doe"


def test_author_overlap_last_name():
    wanted = ["John Doe", "Alice Smith"]
    cand = ["Doe, J.", "Smith, Alice"]
    assert author_overlap(wanted, cand, use_last_name=True) == 1.0


def test_title_similarity_exact():
    assert title_similarity("My Title", "My Title") == 1.0


def test_best_match_index():
    candidates = [
        {
            "title": "A Study on Perfectoid Spaces",
            "authors": ["Alice"],
            "cited_by_count": 5,
        },
        {
            "title": "A Study on Perfectoid Spaces",
            "authors": ["Bob"],
            "cited_by_count": 50,
        },
    ]
    idx = best_match_index(
        candidates,
        title="A Study on Perfectoid Spaces",
        authors=["Alice"],
        min_title_similarity=0.9,
        min_author_overlap=0.1,
        require_author_overlap=True,
        use_last_name=False,
    )
    assert idx == 0
