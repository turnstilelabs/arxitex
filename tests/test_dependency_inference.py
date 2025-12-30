import asyncio

from arxitex.extractor.dependency_inference.dependency_inference import (
    GraphDependencyInference,
)
from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
)
from arxitex.extractor.dependency_inference.dependency_models import (
    PairwiseDependencyCheck,
)
from arxitex.extractor.dependency_inference.global_dependency_inference import (
    GlobalGraphDependencyInference,
)
from arxitex.extractor.models import ArtifactNode, ArtifactType, Position


def test_infer_dependency_sync_monkeypatch(monkeypatch):
    gdi = GraphDependencyInference()

    # Monkeypatch the llms.execute_prompt used inside infer_dependency
    def fake_execute_prompt(prompt, output_class, model):
        # Return a pydantic model instance or object compatible with PairwiseDependencyCheck
        return PairwiseDependencyCheck(
            has_dependency=True, dependency_type="used_in", justification="evidence"
        )

    monkeypatch.setattr("arxitex.llms.llms.execute_prompt", fake_execute_prompt)

    src = {"id": "s1", "content": "source content", "type": "theorem"}
    tgt = {"id": "t1", "content": "target content", "type": "lemma"}

    res = gdi.infer_dependency(src, tgt)
    assert hasattr(res, "has_dependency")
    assert res.has_dependency is True
    assert res.dependency_type is not None


def test_infer_dependency_async_monkeypatch(monkeypatch):
    gdi = GraphDependencyInference()

    async def fake_aexecute_prompt(prompt, output_class, model):
        # Return an object that has the expected attributes
        return PairwiseDependencyCheck(
            has_dependency=False, dependency_type=None, justification=None
        )

    monkeypatch.setattr("arxitex.llms.llms.aexecute_prompt", fake_aexecute_prompt)

    src = {"id": "s1", "content": "source content", "type": "theorem"}
    tgt = {"id": "t1", "content": "target content", "type": "lemma"}

    res = asyncio.run(gdi.ainfer_dependency(src, tgt))
    assert hasattr(res, "has_dependency")
    assert res.has_dependency is False


def test_global_dependency_inference_monkeypatch(monkeypatch):
    ggi = GlobalGraphDependencyInference()

    async def fake_aexecute_prompt(prompt, output_class, model):
        # Minimal object compatible with GlobalDependencyGraph (pydantic will validate)
        return output_class.model_validate(
            {
                "edges": [
                    {
                        "source_id": "s1",
                        "target_id": "t1",
                        "dependency_type": "used_in",
                        "justification": "x",
                    }
                ]
            }
        )

    monkeypatch.setattr("arxitex.llms.llms.aexecute_prompt", fake_aexecute_prompt)

    nodes = [
        ArtifactNode(
            id="t1",
            type=ArtifactType.LEMMA,
            content="t",
            position=Position(line_start=1),
        ),
        ArtifactNode(
            id="s1",
            type=ArtifactType.THEOREM,
            content="s",
            position=Position(line_start=2),
        ),
    ]
    cfg = DependencyInferenceConfig()
    res = asyncio.run(ggi.ainfer_dependencies(nodes, cfg))
    assert len(res.edges) == 1
    assert res.edges[0].dependency_type.value == "used_in"
