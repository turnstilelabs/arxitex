from arxitex.tools.retrieval.agentic.models import AgentDecision
from arxitex.tools.retrieval.agentic.service import run_agentic_search


def test_agentic_search_final_selection() -> None:
    candidates = [
        {
            "statement_id": "lemma:test",
            "type": "lemma",
            "number": "1.1",
            "score": 0.8,
            "text_preview": "Lemma statement",
        }
    ]

    def search_fn(query: str, k: int):
        return candidates

    decisions = iter(
        [AgentDecision(action="final", selected_id="lemma:test", confidence=0.9)]
    )

    def fake_llm(prompt, output_class, model: str):
        return next(decisions)

    result = run_agentic_search(
        mention="find lemma",
        search_fn=search_fn,
        model="fake-model",
        max_steps=2,
        top_k=5,
        llm_executor=fake_llm,
    )
    assert result.selected_id == "lemma:test"
    assert result.candidates[0] == "lemma:test"
