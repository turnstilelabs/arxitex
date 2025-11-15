import os
import time

import pytest

from arxitex.experiments.generate_then_verify import MultiRetrievalOutput
from arxitex.experiments.prompts_generate_then_verify import prompt_closed_book
from arxitex.llms.llms import execute_prompt


@pytest.mark.live_openai
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_chat_search_live_minimal():
    """
    Live integration test that makes a real call to OpenAI using a search-enabled
    Chat Completions model and validates that we can parse a structured response
    into our Pydantic model (MultiRetrievalOutput).

    pytest -m live_openai -k openai_chat_search_live -q
    """
    model = os.getenv("ARXITEX_TEST_SEARCH_MODEL", "gpt-5-search-api")

    q = f"Positive science news story from today? {int(time.time())}"
    prompt = prompt_closed_book(q, k=1)
    out = execute_prompt(prompt, MultiRetrievalOutput, model=model)

    assert isinstance(out, MultiRetrievalOutput)
    assert isinstance(out.candidates, list)
    assert len(out.candidates) >= 1
    top = out.candidates[0]
    assert top.reference is not None
    assert (top.reference.title or "").strip() != ""

    print(
        f"[live_openai] model={model} top_title={top.reference.title!r} confidence={getattr(top, 'confidence', None)}"
    )
