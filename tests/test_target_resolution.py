from arxitex.tools.mentions.acquisition.target_resolution import (
    OpenAlexTargetResolver,
    TargetWorkProfile,
    classify_bib_entry,
)


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


def _profile() -> TargetWorkProfile:
    return TargetWorkProfile(
        title="Perfectoid Spaces",
        doi="10.1007/s10240-012-0042-x",
        year=2011,
    )


def test_classify_exact_target_by_doi():
    match = classify_bib_entry(
        "Scholze, Perfectoid Spaces, Publ. Math. IHES, doi:10.1007/s10240-012-0042-x",
        _profile(),
    )
    assert match == "exact_target"


def test_classify_non_target_survey():
    match = classify_bib_entry(
        "Perfectoid spaces: A survey, lecture notes, 2015",
        TargetWorkProfile(title="Perfectoid Spaces", year=2011),
    )
    assert match == "non_target"


def test_classify_unknown_with_weak_signal():
    match = classify_bib_entry(
        "Scholze, Perfectoid Spaces",
        TargetWorkProfile(title="Perfectoid Spaces", year=2011),
    )
    assert match in {"unknown", "same_work_alt_version", "non_target"}


def test_classify_same_work_from_title_phrase_and_pmihes():
    match = classify_bib_entry(
        "[Sch12] P. Scholze: Perfectoid spaces, Publ. Math. Inst. Hautes Études Sci. 116 (2012)",
        TargetWorkProfile(title="Perfectoid Spaces", year=2011),
    )
    assert match == "same_work_alt_version"
