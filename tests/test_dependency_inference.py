import asyncio

from arxitex.extractor.dependency_inference.dependency_inference import (
    GraphDependencyInference,
)
from arxitex.extractor.dependency_inference.dependency_models import (
    PairwiseDependencyCheck,
)


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
