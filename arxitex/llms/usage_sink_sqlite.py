from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from arxitex.db.connection import connect
from arxitex.db.schema import ensure_schema
from arxitex.llms.metrics import TokenUsage
from arxitex.llms.usage_context import get_usage_context


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteUsageSink:
    """Persist TokenUsage events into SQLite.

    This is best-effort and must never raise (metrics should not break the pipeline).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        # Ensure table exists.
        try:
            ensure_schema(self.db_path)
        except Exception as e:
            logger.warning(f"Could not ensure schema for usage sink: {e}")

    def __call__(self, u: TokenUsage) -> None:
        ctx = get_usage_context()
        paper_id = ctx.get("paper_id")
        mode = ctx.get("mode")
        stage = ctx.get("stage")

        try:
            conn = connect(self.db_path)
            try:
                # Ensure FK target exists (papers row) even if the main pipeline
                # isn't persisting normalized data.
                if paper_id:
                    conn.execute(
                        "INSERT OR IGNORE INTO papers (paper_id) VALUES (?)",
                        (paper_id,),
                    )

                conn.execute(
                    """
                    INSERT INTO llm_usage (
                        created_at_utc, paper_id, mode, stage,
                        provider, model, prompt_id, context, cached,
                        prompt_tokens, completion_tokens, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _utc_now_iso(),
                        paper_id,
                        mode,
                        stage,
                        u.provider,
                        u.model,
                        # We don't currently have prompt_id on TokenUsage; keep null.
                        None,
                        u.context,
                        1 if u.cached else 0,
                        u.prompt_tokens,
                        u.completion_tokens,
                        u.total_tokens,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as e:
            # Best-effort: don't spam; but log once in debug.
            logger.debug(f"Failed to write llm_usage row: {e} usage={asdict(u)}")
        except Exception as e:
            logger.debug(f"Failed to write llm_usage row: {e} usage={asdict(u)}")
