from arxitex.tools.citations.utils import extract_named, extract_refs


def test_extract_refs_allows_abbrev_with_dot() -> None:
    refs = extract_refs("See Thm. 2.1 for details.")
    assert refs
    assert refs[0]["kind"] == "theorem"
    assert refs[0]["number"] == "2.1"


def test_extract_refs_allows_prop_abbrev_with_dot() -> None:
    refs = extract_refs("By Prop. 3 we obtain the claim.")
    assert refs
    assert refs[0]["kind"] == "proposition"
    assert refs[0]["number"] == "3"


def test_extract_refs_allows_letter_only_numbers() -> None:
    refs = extract_refs("Theorem A is classical.")
    assert refs
    assert refs[0]["kind"] == "theorem"
    assert refs[0]["number"] == "A"


def test_extract_refs_does_not_match_plural_kind() -> None:
    refs = extract_refs("theorems 2.1 are discussed elsewhere.")
    assert refs == []


def test_extract_named_captures_multiword_capitalized() -> None:
    named = extract_named("We use Theorem Hahn Banach in the proof.")
    assert named
    assert named[0]["kind"] == "theorem"
    assert named[0]["name"] == "Hahn Banach"


def test_extract_named_skips_lowercase_phrase() -> None:
    named = extract_named("This is Theorem general result.")
    assert named == []
