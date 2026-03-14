from pathlib import Path

from arxitex.tools.retrieval.msc2020 import MSCDictionary, context_similarity

CSV_CONTENT = """code,description
32Qxx,Complex manifolds
20Fxx,Combinatorial group theory
20-XX,Group results
14J32,Abelian varieties and moduli
"""


def test_msc_dictionary_loads_prefixes(tmp_path):
    csv_path = Path(tmp_path) / "msc.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")

    msc = MSCDictionary.from_csv(csv_path)
    assert "14J32" in msc.codes_5_digit
    assert "32Qxx" in msc.codes_3_digit
    assert "20Fxx" in msc.codes_3_digit
    assert "20-XX" in msc.codes_2_digit


def test_msc_dictionary_loads_tab_latin1_csv(tmp_path):
    csv_path = Path(tmp_path) / "msc.tsv"
    csv_bytes = (
        b"code\tdescription\r\n"
        b"32Qxx\tComplex manifolds; K\xf6hler geometry\r\n"
        b"20-XX\tGroup results\r\n"
    )
    csv_path.write_bytes(csv_bytes)

    msc = MSCDictionary.from_csv(csv_path)
    assert "32Qxx" in msc.codes_3_digit
    assert "20-XX" in msc.codes_2_digit


def test_msc_match_context_iteratively_strips_adjectives(tmp_path):
    csv_path = Path(tmp_path) / "msc.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")
    msc = MSCDictionary.from_csv(csv_path)

    m1 = msc.match_context("smooth complex manifolds")
    assert m1.code == "32Qxx"
    assert m1.level == 3

    m2 = msc.match_context("abstract group results")
    assert m2.code == "20-XX"
    assert m2.level == 2


def test_msc_match_prefers_five_digit(tmp_path):
    csv_path = Path(tmp_path) / "msc.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")
    msc = MSCDictionary.from_csv(csv_path)

    m = msc.match_context("abelian varieties moduli")
    assert m.code == "14J32"
    assert m.level == 5


def test_context_similarity_scores():
    assert context_similarity("32Qxx", "32Qxx") == 1.0
    assert context_similarity("14J32", "14J32") == 1.0
    assert context_similarity("14J32", "14Jxx") == 0.75
    assert context_similarity("20Fxx", "20-XX") == 0.5
    assert context_similarity("32Qxx", "20-XX") == 0.0
    assert context_similarity(None, "20-XX") == 0.0
