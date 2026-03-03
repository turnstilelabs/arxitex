from arxitex.tools.citations.target_resolution import OpenAlexTargetResolver


def test_select_openalex_work_id_perfectoid():
    results = [
        {
            "id": "https://openalex.org/W0000000000",
            "title": "Perfectoid spaces and diamonds",
            "cited_by_count": 999,
            "authorships": [{"author": {"display_name": "Alice Example"}}],
        },
        {
            "id": "https://openalex.org/W4255501032",
            "title": "Perfectoid Spaces",
            "cited_by_count": 200,
            "authorships": [{"author": {"display_name": "Peter Scholze"}}],
        },
    ]
    picked = OpenAlexTargetResolver.select_openalex_work_id(
        results, "Perfectoid Spaces", ["Peter Scholze"]
    )
    assert picked == "https://openalex.org/W4255501032"
