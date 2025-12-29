import sqlite3

from arxitex.db.schema import ensure_schema
from arxitex.llms.metrics import TokenUsage, register_usage_sink
from arxitex.llms.usage_context import llm_usage_context
from arxitex.llms.usage_sink_sqlite import SQLiteUsageSink


def test_llm_usage_logging_writes_row(tmp_path):
    db_path = tmp_path / "arxitex_indices.db"
    ensure_schema(db_path)

    # Register a sink just for this test.
    register_usage_sink(SQLiteUsageSink(db_path))

    with llm_usage_context(paper_id="9999.99999v1", mode="defs"):
        # This simulates what metrics.py does internally
        # (we don't call an actual provider in unit tests).
        u = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="test-model",
            provider="test-provider",
            cached=False,
            context="unit-test",
        )

        # Call the sink directly to avoid depending on logger configuration.
        SQLiteUsageSink(db_path)(u)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "select paper_id, mode, provider, model, prompt_tokens, completion_tokens, total_tokens from llm_usage"
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "9999.99999v1"
        assert rows[0][1] == "defs"
        assert rows[0][2] == "test-provider"
        assert rows[0][3] == "test-model"
        assert rows[0][4] == 10
        assert rows[0][5] == 5
        assert rows[0][6] == 15
    finally:
        conn.close()
