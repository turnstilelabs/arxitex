import asyncio

from arxitex.extractor.graph_building.graph_enhancer import GraphEnhancer
from arxitex.extractor.models import ArtifactNode, ArtifactType, DocumentGraph, Position
from arxitex.symdef.utils import Definition


def make_graph_with_terms():
    g = DocumentGraph()
    n1 = ArtifactNode(
        id="n1",
        type=ArtifactType.THEOREM,
        content="content1",
        position=Position(line_start=1),
    )
    n2 = ArtifactNode(
        id="n2",
        type=ArtifactType.LEMMA,
        content="content2",
        position=Position(line_start=10),
    )
    n3 = ArtifactNode(
        id="n3",
        type=ArtifactType.DEFINITION,
        content="content3",
        position=Position(line_start=20),
    )
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    return g


def test_candidate_generation_overlaps_and_subwords(monkeypatch):
    ge = GraphEnhancer()
    graph = make_graph_with_terms()

    # artifact_to_terms_map: n1 and n2 share 'common', n2 and n3 have subword relation
    artifact_to_terms_map = {
        "n1": ["common_term"],
        "n2": ["common_term", "union closed"],
        "n3": ["approximate union closed"],
    }

    # Prepare a dummy bank where find_many returns a Definition with dependencies
    # for 'union closed' term. Also ensure find() is never called.
    class DummyBank:
        def __init__(self):
            self.find_calls = 0
            self.find_many_calls = 0
            self._map = {
                "union closed": Definition(
                    term="union closed",
                    definition_text="d",
                    source_artifact_id="n2",
                    aliases=[],
                    dependencies=["group"],
                )
            }

        def _normalize_term(self, term: str) -> str:
            return term.strip().lower()

        async def find(self, term):
            self.find_calls += 1
            return self._map.get(term)

        async def find_many(self, terms):
            self.find_many_calls += 1
            out = []
            for t in terms:
                d = self._map.get(t)
                if d is not None:
                    out.append(d)
            return out

    bank = DummyBank()

    # Capture calls made to the LLM checker by replacing the llm_dependency_checker
    calls = []

    class DummyLLM:
        async def ainfer_dependency(self, src, tgt):
            calls.append((src["id"], tgt["id"]))

            # return no dependency to avoid modifying graph
            class R:
                has_dependency = False

            return R()

    ge.llm_dependency_checker = DummyLLM()

    new_graph = asyncio.run(
        ge._infer_and_add_dependencies_pairwise(
            graph, artifact_to_terms_map, bank, cfg=None
        )
    )

    # Regression: footprint expansion should use a single find_many, not repeated find()
    assert bank.find_many_calls == 1
    assert bank.find_calls == 0

    # The function returns the (possibly mutated) DocumentGraph; ensure we got the same object back.
    assert new_graph is graph

    # Ensure candidate pairs were generated and LLM was invoked at least once
    assert len(calls) >= 1
    # For our mapping, at least pair between n1 and n2 should be considered
    assert any(("n1" in (a, b) and "n2" in (a, b)) for a, b in calls)
