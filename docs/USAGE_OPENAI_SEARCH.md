# Using OpenAI Chat Completions with Web Search

This project supports OpenAI search-enabled Chat Completions models (e.g., `gpt-5-search-api`) via `web_search_options`.

What’s implemented
- Registry: added search-capable models:
  - gpt-5-search-api
  - gpt-4o-search-preview
  - gpt-4o-mini-search-preview
- LLM calls: `arxitex/llms/llms.py` forwards `web_search_options` automatically when a search model is used (or when `ARXITEX_OPENAI_WEB_SEARCH=1` is set).
- Live test: `tests/test_openai_search_live.py` performs a real OpenAI call and checks structured parsing into Pydantic.

Requirements
- Python dependencies from `requirements.txt` (ensure `openai>=1.86.0`)
- Environment variable:
  - `OPENAI_API_KEY` set to your OpenAI API key

Optional environment
- `ARXITEX_TEST_SEARCH_MODEL` to change the test model (default: `gpt-5-search-api`)
- `ARXITEX_OPENAI_WEB_SEARCH=1` forces `web_search_options` for non-preview models (not required if you use a search-enabled model)
- `ARXITEX_OPENAI_WEB_SEARCH_OPTIONS` JSON string to pass options to `web_search_options` (default `{}`)

Run the live OpenAI test (explicit marker)
1) Install deps and export your key:
   - `pip install -r requirements.txt`
   - `export OPENAI_API_KEY="sk-..."`
2) Run only the live test:
   - `pytest -m live_openai -k openai_chat_search_live -q`

Notes
- The test is minimal-cost (k=1) and only asserts that a non-empty candidate title is returned and parsed, to avoid flakiness.
- By default, the rest of the test suite does NOT make network calls.

Run the experiment with web search
- Example:
  - `export OPENAI_API_KEY="sk-..."`
  - `python -m arxitex.experiments.generate_then_verify -m gpt-5-search-api`

Troubleshooting
- If your installed OpenAI SDK errors on `web_search_options` with `chat.beta.completions.parse`, update the `openai` package:
  - `pip install -U openai`
- If you need to force-enable `web_search_options`, set `ARXITEX_OPENAI_WEB_SEARCH=1`.
- To pass custom search options, set `ARXITEX_OPENAI_WEB_SEARCH_OPTIONS`, e.g.:
  - `export ARXITEX_OPENAI_WEB_SEARCH_OPTIONS='{"search_depth":"advanced"}'`
