from pydantic import BaseModel

from arxitex.llms.json_extractor import JSONExtractor, extract_after_think
from arxitex.llms.prompt import Prompt


class DummyModel(BaseModel):
    a: int
    b: str


def test_extract_after_think():
    s = 'some text</think>{"a":1, "b":"x"}'
    assert extract_after_think(s).startswith("{")
    assert '{"a":1' in extract_after_think(s)


def test_local_json_extraction_basic(monkeypatch):
    # Prevent JSONExtractor from constructing a real OpenAI client that requires keys
    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            class Beta:
                class Chat:
                    class Completions:
                        @staticmethod
                        def parse(*args, **kwargs):
                            raise RuntimeError(
                                "Should not be called in local extraction tests"
                            )

                    completions = Completions()

                chat = Chat()

            self.beta = Beta()

    monkeypatch.setattr("arxitex.llms.json_extractor.OpenAI", DummyOpenAI)

    extractor = JSONExtractor()
    text = 'Here is some output: {"a": 5, "b": "hello"} and trailing text.'
    result = extractor.extract_json(text, DummyModel)
    assert result is not None
    assert result.a == 5
    assert result.b == "hello"


def test_prompt_cache_save_and_get(tmp_path, monkeypatch):
    # point PROMPTS_CACHE_PATH to tmp_path
    monkeypatch.setenv("PROMPTS_CACHE_PATH", str(tmp_path))
    # re-import functions to pick up env change if necessary (they read at import time)
    from importlib import reload

    import arxitex.llms.prompt_cache as pc

    reload(pc)

    prompt = Prompt(id="p1", system="sys", user="usr")
    model = "m1"
    data = {"x": 1, "y": "z"}

    # save
    pc.save_prompt_result(prompt, model, data)
    # get
    loaded = pc.get_prompt_result(prompt, model)
    assert loaded is not None
    assert loaded["x"] == 1
    assert loaded["y"] == "z"
