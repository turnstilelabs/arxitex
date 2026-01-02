from arxitex.extractor.graph_building.newtheorem_scanner import NewTheoremScanner


def test_newtheorem_scanner_simple_definitions():
    content = r"""
    % Simple newtheorem declarations
    \newtheorem{thm}{Theorem}
    \newtheorem{lem}{Lemma}
    \newtheorem{defn}{Definition}
    \newtheorem{prop}{Proposition}
    \newtheorem{cor}{Corollary}
    \newtheorem{rem}{Remark}
    \newtheorem{ex}{Example}
    """

    aliases = NewTheoremScanner.scan(content)

    assert aliases["thm"] == "theorem"
    assert aliases["lem"] == "lemma"
    assert aliases["defn"] == "definition"
    assert aliases["prop"] == "proposition"
    assert aliases["cor"] == "corollary"
    assert aliases["rem"] == "remark"
    assert aliases["ex"] == "example"


def test_newtheorem_scanner_shared_counter_family():
    content = r"""
    % Complex shared-counter style definitions
    \newtheorem{thm1}{Theorem}[section]
    \newtheorem{lem1}[thm1]{Lemma}
    \newtheorem{rem1}[thm1]{Remark}
    \newtheorem{def1}[thm1]{Definition}
    \newtheorem{cor1}[thm1]{Corollary}
    \newtheorem{defn1}[thm1]{Definition}
    \newtheorem{prop1}[thm1]{Proposition}
    \newtheorem{ex1}[thm1]{Example}
    """

    aliases = NewTheoremScanner.scan(content)

    assert aliases["thm1"] == "theorem"
    assert aliases["lem1"] == "lemma"
    assert aliases["rem1"] == "remark"
    assert aliases["def1"] == "definition"
    assert aliases["cor1"] == "corollary"
    assert aliases["defn1"] == "definition"
    assert aliases["prop1"] == "proposition"
    assert aliases["ex1"] == "example"


def test_newtheorem_scanner_ignores_unknown_titles():
    content = r"""
    % This title does not clearly map to a known artifact type
    \newtheorem{weird}{Foobar}
    """

    aliases = NewTheoremScanner.scan(content)

    assert "weird" not in aliases
