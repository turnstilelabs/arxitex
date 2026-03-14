from pathlib import Path

from arxitex.tools.retrieval.logic_decomposition import service as decomp_service
from arxitex.tools.retrieval.logic_decomposition.models import (
    LogicGoal,
    LogicHypothesis,
)
from arxitex.tools.retrieval.logic_decomposition.prompt import (
    LogicDecompositionPromptGenerator,
)


def test_logic_decomposition_prompt_contract():
    gen = LogicDecompositionPromptGenerator()
    prompt = gen.make_prompt("Let G be an abelian group.", "logic-abc")
    assert prompt.id == "logic-abc"
    assert "Return JSON only" in prompt.system
    assert "hypotheses" in prompt.system
    assert "Statement:" in prompt.user


def test_extract_logic_decompositions_cache_hit(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def fake_aexecute_prompt(prompt, output_class, model):
        calls["n"] += 1
        return output_class(
            context="group theory",
            hypotheses=[
                LogicHypothesis(
                    id="h1",
                    predicate="is_group",
                    args=["G"],
                    polarity="pos",
                    quantifier="none",
                    scope="global",
                    raw="G is a group",
                )
            ],
            goal=LogicGoal(
                raw="G is abelian", canonical_latex="G\\text{ abelian}", ops=[]
            ),
        )

    monkeypatch.setattr(decomp_service, "aexecute_prompt", fake_aexecute_prompt)

    cache_path = Path(tmp_path) / "logic_cache.jsonl"
    out1 = decomp_service.extract_logic_decompositions(
        texts=["Let G be a group."],
        ids=["a1"],
        model="test-model",
        cache_path=cache_path,
        concurrency=1,
    )
    assert "a1" in out1
    assert out1["a1"].context == "group theory"
    assert calls["n"] == 1

    async def fail_aexecute_prompt(prompt, output_class, model):
        raise AssertionError("cache should be used on second call")

    monkeypatch.setattr(decomp_service, "aexecute_prompt", fail_aexecute_prompt)

    out2 = decomp_service.extract_logic_decompositions(
        texts=["Let G be a group."],
        ids=["a1"],
        model="test-model",
        cache_path=cache_path,
        concurrency=1,
    )
    assert "a1" in out2
    assert out2["a1"].context == "group theory"


def test_extract_logic_decompositions_fallback_is_deterministic(tmp_path, monkeypatch):
    async def fail_aexecute_prompt(prompt, output_class, model):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(decomp_service, "aexecute_prompt", fail_aexecute_prompt)

    cache_path = Path(tmp_path) / "logic_cache.jsonl"
    text = "If x+y=z then z-x=y"

    out_a = decomp_service.extract_logic_decompositions(
        texts=[text],
        ids=["q1"],
        model="test-model",
        cache_path=cache_path,
        concurrency=1,
    )
    out_b = decomp_service.extract_logic_decompositions(
        texts=[text],
        ids=["q1b"],
        model="test-model",
        cache_path=cache_path,
        concurrency=1,
    )

    assert out_a["q1"].hypotheses == []
    assert out_a["q1"].goal.raw == text
    assert out_a["q1"].goal.canonical_latex
    assert out_a["q1"].goal.canonical_latex == out_b["q1b"].goal.canonical_latex
