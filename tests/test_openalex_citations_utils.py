from arxitex.tools.mentions.acquisition.openalex_citations import (
    normalize_openalex_work_id,
)


def test_normalize_openalex_work_id_variants():
    assert normalize_openalex_work_id("W123") == "https://openalex.org/W123"
    assert (
        normalize_openalex_work_id("https://openalex.org/W123")
        == "https://openalex.org/W123"
    )
    assert (
        normalize_openalex_work_id("https://openalex.org/works/W123")
        == "https://openalex.org/W123"
    )
