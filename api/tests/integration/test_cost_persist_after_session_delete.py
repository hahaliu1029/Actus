"""Repro: session deleted while its task is still finishing → cost persist
must skip quietly instead of hitting fk_cost_records_session_id_sessions twice.

Prod evidence (2026-07-10 17:24:06 actus-api-1):

    WARNING CostCallbackHandler: persister failed for session_id=a8aff5aa...
        ForeignKeyViolationError ... fk_cost_records_session_id_sessions
    WARNING CostCallbackHandler: degraded marker insert also failed ...

The degraded-marker fallback re-used the deleted ``session_id`` via
``replace(record, ...)`` so it hit the exact same FK — the fallback itself
could never work. And counting the FK hit as a persist failure made the
terminal drain chase it with a session-level marker (same FK again).

Contract locked here (best-effort, session-gone aware):
- nothing raises out of the callback path;
- the deleted-session FK hit is NOT counted as a persist failure;
- the degraded-marker fallback is NOT attempted against the same dead FK;
- no WARNING-level noise — this is an expected lifecycle race, logged INFO.

Run: cd api && uv run pytest tests/integration/test_cost_persist_after_session_delete.py -v
Requires: pgvector-enabled Postgres reachable via SQLALCHEMY_DATABASE_URL.
"""

from __future__ import annotations

import logging
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from sqlalchemy import text

pytestmark = pytest.mark.anyio

_HANDLER_LOGGER = "app.domain.services.cost_callback_handler"


def _llm_result(usage: dict | None) -> LLMResult:
    return LLMResult(
        generations=[
            [ChatGeneration(message=AIMessage(content="hi", usage_metadata=usage))]
        ]
    )


async def test_persist_after_session_delete_skips_quietly(
    async_session_factory, uow_factory, caplog
) -> None:
    """Session row deleted mid-run → persist path stays silent + marker-free."""
    from app.application.services.cost_callback_factory import (
        build_cost_callback_handler,
    )

    user_id = str(uuid.uuid4())
    session_id = f"sess-fkgone-{uuid.uuid4().hex[:12]}"

    # Committed rows on an independent connection: the handler's persister
    # opens its own UoW and cannot see uncommitted fixture state (see the
    # transaction-isolation warning in tests/integration/conftest.py).
    async with async_session_factory() as setup:
        await setup.execute(
            text("INSERT INTO users (id) VALUES (:uid)"), {"uid": user_id}
        )
        await setup.execute(
            text("INSERT INTO sessions (id, user_id) VALUES (:sid, :uid)"),
            {"sid": session_id, "uid": user_id},
        )
        await setup.commit()

    try:
        handler = build_cost_callback_handler(session_id, user_id, uow_factory)

        # The session is deleted while the task is still winding down.
        async with async_session_factory() as s:
            await s.execute(
                text("DELETE FROM sessions WHERE id = :sid"), {"sid": session_id}
            )
            await s.commit()

        run_id = uuid.uuid4()
        with caplog.at_level(logging.DEBUG, logger=_HANDLER_LOGGER):
            await handler.on_chat_model_start(
                serialized={},
                messages=[[HumanMessage(content="hi")]],
                run_id=run_id,
                metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
                invocation_params={
                    "model": "gpt-4o",
                    "provider_id": "openai_official",
                },
            )
            await handler.on_llm_end(
                _llm_result(
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                ),
                run_id=run_id,
            )
            flush = await handler.flush_pending()

        assert flush.drained is True

        # NOT a persist failure: counting it would make the terminal drain
        # write a session-level degraded marker → same dead FK again.
        assert flush.persist_failures == 0
        assert handler.persist_failure_count == 0

        # The per-record degraded-marker fallback must not re-hit the FK.
        assert "degraded marker insert also failed" not in caplog.text

        handler_warnings = [
            r.getMessage()
            for r in caplog.records
            if r.name == _HANDLER_LOGGER and r.levelno >= logging.WARNING
        ]
        assert handler_warnings == [], (
            "deleted-session persist must not WARN (expected lifecycle race), "
            f"got: {handler_warnings}"
        )

        # The skip is still observable at INFO for ops.
        assert any(
            r.name == _HANDLER_LOGGER
            and r.levelno == logging.INFO
            and "session row gone" in r.getMessage()
            for r in caplog.records
        ), "expected one INFO 'session row gone' log"

        # And of course nothing landed for the deleted session.
        async with async_session_factory() as s:
            count = (
                await s.execute(
                    text(
                        "SELECT count(*) FROM cost_records WHERE session_id = :sid"
                    ),
                    {"sid": session_id},
                )
            ).scalar_one()
        assert count == 0
    finally:
        async with async_session_factory() as s:
            await s.execute(
                text("DELETE FROM users WHERE id = :uid"), {"uid": user_id}
            )
            await s.commit()
