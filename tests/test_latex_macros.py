from arxitex.symdef.utils import extract_latex_macros


def test_extract_latex_macros_basic_newcommand_and_def():
    latex = r"""
    % Preamble
    \newcommand{\cF}{\mathcal{F}}
    \def\Hom{\operatorname{Hom}}
    \newcommand{\foo}[1]{#1^2}  % has argument, should be ignored

    \begin{document}
    Body where macros are used: $\cF$ and $\Hom$.
    \end{document}
    """

    macros = extract_latex_macros(latex)

    # Should capture simple, argument-free macros from the preamble only.
    assert macros.get("cF") == "\\mathcal{F}"
    assert macros.get("Hom") == "\\operatorname{Hom}"

    # Macros with arguments (#1, #2, ...) should be skipped for safety.
    assert "foo" not in macros
