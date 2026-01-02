import asyncio
import time

from arxitex.extractor.models import ArtifactNode, ArtifactType, Position
from arxitex.symdef.definition_bank import DefinitionBank
from arxitex.symdef.document_enhancer import DocumentEnhancer
from arxitex.symdef.utils import ContextFinder


class DummyBuilder:
    """A deterministic, async stub replacing the real DefinitionBuilder/LLM."""

    def __init__(
        self, *, sleep_s: float = 0.0, synthesized_definition: str | None = None
    ):
        self.sleep_s = sleep_s
        self.synthesized_definition = synthesized_definition
        self.synthesis_calls: list[str] = []

    async def aextract_document_terms(self, full_document_content: str):
        # pretend the LLM found a single term "TermOne"
        return ["TermOne"]

    async def aextract_single_artifact_terms(self, artifact_content: str):
        # not used in this test (we use global extraction), but provide for completeness
        return ["TermOne"]

    async def aextract_definition(self, artifact_content: str):
        # If called, return an ExtractedDefinition-like object
        class R:
            defined_term = "ExplicitTerm"
            definition_text = "An explicit definition."
            aliases = []

        return R()

    async def asynthesize_definition(
        self, term: str, context_snippets: str, base_definition
    ):
        self.synthesis_calls.append(term)
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)

        if self.synthesized_definition is not None:
            return self.synthesized_definition

        # Default: Return a deterministic synthesized definition
        return f"SYNTHETIC_DEF_FOR_{term}"


def make_latex_and_artifact():
    # Build a small latex doc where the term "TermOne" appears on line 4
    latex_lines = [
        "Title\n",
        "Intro paragraph\n",
        "More text\n",
        "Here we use TermOne in a sentence.\n",
        "End.\n",
    ]
    latex = "".join(latex_lines)

    # Artifact that corresponds to the line with TermOne
    # line_start is 4, we set col_start/col_end to cover the occurrence
    art = ArtifactNode(
        id="a1",
        type=ArtifactType.THEOREM,
        content="Here we use TermOne in a sentence.",
        label="lbl1",
        position=Position(line_start=4, line_end=4, col_start=1, col_end=40),
    )
    return latex, [art]


def test_document_enhancer_synthesizes_and_registers(tmp_path):
    latex, artifacts = make_latex_and_artifact()
    dummy_builder = DummyBuilder()
    context_finder = ContextFinder()
    bank = DefinitionBank()
    enhancer = DocumentEnhancer(
        llm_enhancer=dummy_builder, context_finder=context_finder, definition_bank=bank
    )

    results = asyncio.run(
        enhancer.enhance_document(artifacts, latex, use_global_extraction=True)
    )

    # results should contain definition bank and artifact->terms map
    assert "definitions_map" in results
    assert "artifact_to_terms_map" in results
    assert "definition_bank" in results

    artifact_terms = results["artifact_to_terms_map"]
    assert "a1" in artifact_terms
    # term should be discovered
    assert "TermOne" in artifact_terms["a1"]

    # The bank should contain the synthesized definition
    bank_dict = asyncio.run(results["definition_bank"].to_dict())
    # The normalized key for "TermOne" should exist
    assert any(
        "TermOne".lower() in k.lower() or "TermOne" in v.get("term", "")
        for k, v in bank_dict.items()
    )

    # The definitions_map should include the artifact id and the synthesized definition text
    definitions_map = results["definitions_map"]
    assert "a1" in definitions_map
    # synthesized def appears in the bank and in definitions_map values
    found = False
    for _, deftext in definitions_map["a1"].items():
        if "SYNTHETIC_DEF_FOR" in deftext or "SYNTHETIC_DEF_FOR_TermOne" in deftext:
            found = True
    assert found


def test_document_enhancer_synthesis_is_concurrent_across_terms():
    # Two terms, two artifacts; synthesis sleep should overlap.
    latex_lines = [
        "Title\n",
        "Intro paragraph\n",
        "More text\n",
        "Here we use TermOne in a sentence.\n",
        "Here we use TermTwo in a sentence.\n",
        "End.\n",
    ]
    latex = "".join(latex_lines)

    artifacts = [
        ArtifactNode(
            id="a1",
            type=ArtifactType.THEOREM,
            content="Here we use TermOne in a sentence.",
            label="lbl1",
            position=Position(line_start=4, line_end=4, col_start=1, col_end=40),
        ),
        ArtifactNode(
            id="a2",
            type=ArtifactType.THEOREM,
            content="Here we use TermTwo in a sentence.",
            label="lbl2",
            position=Position(line_start=5, line_end=5, col_start=1, col_end=40),
        ),
    ]

    dummy_builder = DummyBuilder(sleep_s=0.25)

    async def aextract_document_terms(_full_document_content: str):
        return ["TermOne", "TermTwo"]

    # Monkeypatch the document terms method for this test instance.
    dummy_builder.aextract_document_terms = aextract_document_terms  # type: ignore

    enhancer = DocumentEnhancer(
        llm_enhancer=dummy_builder,
        context_finder=ContextFinder(),
        definition_bank=DefinitionBank(),
    )

    start = time.perf_counter()
    asyncio.run(enhancer.enhance_document(artifacts, latex, use_global_extraction=True))
    elapsed = time.perf_counter() - start

    # If serialized, we'd expect ~0.50s (2 * 0.25). With concurrency, closer to ~0.25.
    # Give a generous bound for CI variance.
    assert elapsed < 0.45
    assert sorted(dummy_builder.synthesis_calls) == ["TermOne", "TermTwo"]


def test_document_enhancer_validation_flag_blocks_untrusted_definitions():
    # Term appears in context, but synthesized definition is *not* in context.
    latex_lines = [
        "Title\n",
        "Intro paragraph\n",
        "More text\n",
        "Here we use TermOne in a sentence.\n",
        "End.\n",
    ]
    latex = "".join(latex_lines)
    art = ArtifactNode(
        id="a1",
        type=ArtifactType.THEOREM,
        content="Here we use TermOne in a sentence.",
        label="lbl1",
        position=Position(line_start=4, line_end=4, col_start=1, col_end=40),
    )

    dummy_builder = DummyBuilder(
        synthesized_definition="This sentence is not in the context."
    )
    enhancer = DocumentEnhancer(
        llm_enhancer=dummy_builder,
        context_finder=ContextFinder(),
        definition_bank=DefinitionBank(),
    )

    results = asyncio.run(
        enhancer.enhance_document(
            [art],
            latex,
            use_global_extraction=True,
            validate_synthesized_definitions=True,
        )
    )

    bank_dict = asyncio.run(results["definition_bank"].to_dict())
    # No definition should have been registered.
    assert bank_dict == {}
