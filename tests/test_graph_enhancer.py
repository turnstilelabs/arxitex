import asyncio

from arxitex.extractor.graph_building.graph_enhancer import GraphEnhancer
from arxitex.extractor.models import ArtifactNode, ArtifactType, DocumentGraph, Position
from arxitex.symdef.utils import Definition


def test_is_subword_of():
    ge = GraphEnhancer()
    assert ge._is_subword_of("union closed", "approximate union closed")
    assert not ge._is_subword_of("term", "term")  # identical should be False
    assert not ge._is_subword_of("", "anything")
    assert not ge._is_subword_of("a", "")  # empty other


def make_graph_two_nodes():
    g = DocumentGraph()
    n1 = ArtifactNode(
        id="n1",
        type=ArtifactType.THEOREM,
        content="This node will use something",
        position=Position(line_start=1),
    )
    n2 = ArtifactNode(
        id="n2",
        type=ArtifactType.LEMMA,
        content="This is the target content",
        position=Position(line_start=20),
    )
    g.add_node(n1)
    g.add_node(n2)
    return g


class DummyDefinitionBank:
    def __init__(self, mapping=None):
        # mapping: term -> Definition or None
        self._mapping = mapping or {}

    async def find(self, term):
        # mimic async find returning Definition or None
        return self._mapping.get(term)


class DummyLLMChecker:
    class Result:
        def __init__(self, has_dependency, dependency_type=None, justification=None):
            self.has_dependency = has_dependency
            self.dependency_type = dependency_type
            self.justification = justification

    async def ainfer_dependency(self, src, tgt):
        # mark dependency if one side uses 'use' and the other contains 'target'
        src_content = src.get("content", "")
        tgt_content = tgt.get("content", "")
        if ("use" in src_content and "target" in tgt_content) or (
            "use" in tgt_content and "target" in src_content
        ):
            return DummyLLMChecker.Result(
                True, dependency_type="uses_result", justification="Detected usage"
            )
        return DummyLLMChecker.Result(False)


def test_infer_and_add_dependencies_with_enrichment():
    ge = GraphEnhancer()
    # Replace llm checker with dummy that deterministically returns a dependency for our pair
    ge.llm_dependency_checker = DummyLLMChecker()

    graph = make_graph_two_nodes()

    # artifact_to_terms_map gives overlapping conceptual footprint ('common')
    artifact_to_terms_map = {"n1": ["common_term"], "n2": ["common_term"]}

    # create a dummy bank whose find returns Definitions (with no extra dependencies)
    mapping = {
        "common_term": Definition(
            term="common_term",
            definition_text="d",
            source_artifact_id="n2",
            aliases=[],
            dependencies=[],
        )
    }
    bank = DummyDefinitionBank(mapping=mapping)

    new_graph = asyncio.run(
        ge._infer_and_add_dependencies(graph, artifact_to_terms_map, bank)
    )

    # After running, we expect at least one edge added
    assert len(new_graph.edges) == 1
    e = new_graph.edges[0]
    assert e.source_id in {"n1", "n2"}
    assert e.target_id in {"n1", "n2"}
    assert e.dependency_type is not None
    assert e.dependency is not None


def test_infer_and_add_dependencies_fallback_all_pairs():
    ge = GraphEnhancer()

    # LLM that never finds a dependency
    class NoDepLLM:
        class Result:
            def __init__(self):
                self.has_dependency = False
                self.dependency_type = None
                self.justification = None

        async def ainfer_dependency(self, a, b):
            return NoDepLLM.Result()

    ge.llm_dependency_checker = NoDepLLM()
    graph = make_graph_two_nodes()

    # call with no enrichment data (bank None, empty artifact_to_terms_map) -> fallback to all pairs
    new_graph = asyncio.run(ge._infer_and_add_dependencies(graph, {}, None))

    # No edges should be added because LLM returns no dependency
    assert len(new_graph.edges) == 0
