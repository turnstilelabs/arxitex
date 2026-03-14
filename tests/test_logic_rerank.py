from arxitex.tools.retrieval.logic_decomposition.models import (
    LogicDecomposition,
    LogicGoal,
    LogicHypothesis,
)
from arxitex.tools.retrieval.logic_rerank import service as rerank_service
from arxitex.tools.retrieval.msc2020 import MSCMatch


def _decomp(context, hyps, goal):
    return LogicDecomposition(
        context=context,
        hypotheses=hyps,
        goal=LogicGoal(raw=goal, canonical_latex=goal, ops=[]),
    )


def test_goal_score_debruijn_fallback_hits_equivalent_forms():
    s = rerank_service.compute_goal_score("x+y=z", "a+b=c", use_debruijn=True)
    assert s == 1.0


def test_logic_rerank_reorders_and_filters_contradiction(monkeypatch):
    async def fake_aexecute_prompt(prompt, output_class, model):
        user = prompt.user
        if "polarity=neg" in user:
            return output_class(shyp=0.0, contradiction=True, rationale="contradiction")
        return output_class(shyp=1.0, contradiction=False, rationale="match")

    monkeypatch.setattr(rerank_service, "aexecute_prompt", fake_aexecute_prompt)

    query_id = "q1"
    query_logic = {
        query_id: _decomp(
            "group theory",
            [
                LogicHypothesis(
                    id="hq1",
                    predicate="is_group",
                    args=["G"],
                    polarity="pos",
                    quantifier="none",
                    scope="global",
                    raw="G is a group",
                )
            ],
            "x+y=z",
        )
    }
    artifact_logic = {
        "a1": _decomp(
            "group theory",
            [
                LogicHypothesis(
                    id="hr1",
                    predicate="is_group",
                    args=["G"],
                    polarity="pos",
                    quantifier="none",
                    scope="global",
                    raw="G is a group",
                )
            ],
            "a+b=c",
        ),
        "a2": _decomp(
            "group theory",
            [
                LogicHypothesis(
                    id="hr2",
                    predicate="is_group",
                    args=["G"],
                    polarity="neg",
                    quantifier="none",
                    scope="global",
                    raw="G is not a group",
                )
            ],
            "u+v=w",
        ),
    }

    query_msc = {query_id: MSCMatch(code="20Fxx", level=3)}
    artifact_msc = {
        "a1": MSCMatch(code="20Fxx", level=3),
        "a2": MSCMatch(code="20Fxx", level=3),
    }

    reranker = rerank_service.LogicReranker(model="test", top_n=2)
    reranked = rerank_service.apply_logic_rerank(
        results={
            query_id: {
                "query_id": query_id,
                "query_text": "group query",
                "indices": [0, 1],
                "scores": [0.2, 0.9],
            }
        },
        query_ids=[query_id],
        id_lookup={0: "a1", 1: "a2"},
        query_logic=query_logic,
        artifact_logic=artifact_logic,
        query_msc=query_msc,
        artifact_msc=artifact_msc,
        reranker=reranker,
    )

    row = reranked[query_id]
    assert row["artifact_ids"][0] == "a1"
    assert "a2" not in row["artifact_ids"]
    assert row["logic_rerank"][0]["artifact_id"] == "a1"


def test_apply_logic_rerank_noop_when_disabled():
    initial = {
        "q1": {
            "query_id": "q1",
            "query_text": "t",
            "indices": [0, 1],
            "scores": [0.9, 0.8],
        }
    }
    out = rerank_service.apply_logic_rerank(
        results=initial,
        query_ids=["q1"],
        id_lookup={0: "a1", 1: "a2"},
        query_logic={},
        artifact_logic={},
        query_msc={},
        artifact_msc={},
        reranker=None,
    )
    assert out == initial
