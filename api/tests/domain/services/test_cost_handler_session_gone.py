"""Session row deleted mid-run → handler must skip persistence quietly.

Prod evidence (2026-07-10 actus-api-1): after a session was deleted while
its task was still finishing, every cost persist hit
``fk_cost_records_session_id_sessions`` AND the degraded-marker fallback hit
the exact same FK again — two WARNINGs per LLM call, zero signal, and the
fallback could never succeed by construction (``replace(record, ...)`` keeps
the deleted ``session_id``).

Contract under test — persisters signal the condition by raising
``SessionRowGoneError``; the handler then treats it as a benign lifecycle
race, NOT a persist failure:

- the degraded-marker fallback is NOT attempted (same dead FK);
- ``persist_failure_count`` does NOT tick (otherwise the terminal drain
  chases it with a session-level marker → same FK again);
- every subsequent persist short-circuits without touching the persister;
- ``write_session_degraded_marker`` stops touching the persister too;
- nothing raises out of the callback path; noise level is one INFO.
"""

from __future__ import annotations

import logging
from typing import List
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.domain.models.cost_record import CostRecord
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    SessionRowGoneError,
)

pytestmark = pytest.mark.anyio

_HANDLER_LOGGER = "app.domain.services.cost_callback_handler"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_llm_result(usage: dict | None):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    return LLMResult(
        generations=[
            [ChatGeneration(message=AIMessage(content="hi", usage_metadata=usage))]
        ]
    )


async def _drive_one_llm_call(handler: CostCallbackHandler) -> None:
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result(
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        ),
        run_id=run_id,
    )


class _SessionGonePersister:
    """Models the DB persister once the session row has been deleted."""

    def __init__(self) -> None:
        self.calls: int = 0

    async def __call__(self, record: CostRecord) -> None:
        self.calls += 1
        raise SessionRowGoneError(
            f"cost_records parent row gone for session_id={record.session_id}"
        )


async def test_session_gone_skips_marker_and_failure_count(caplog) -> None:
    """First persist detects session-gone → no marker retry, no failure tick."""
    persister = _SessionGonePersister()
    handler = CostCallbackHandler(
        session_id="sess-gone", user_id="u", persister=persister
    )

    with caplog.at_level(logging.DEBUG, logger=_HANDLER_LOGGER):
        await _drive_one_llm_call(handler)
        flush = await handler.flush_pending()

    # Exactly ONE persister call: the degraded-marker fallback must not
    # be attempted against the same dead FK.
    assert persister.calls == 1

    # Not a persist failure — the ledger for a deleted session is gone by
    # CASCADE; there is nothing to mark as degraded.
    assert handler.persist_failure_count == 0
    assert flush.persist_failures == 0
    assert flush.drained is True

    handler_warnings = [
        r for r in caplog.records
        if r.name == _HANDLER_LOGGER and r.levelno >= logging.WARNING
    ]
    assert handler_warnings == [], (
        f"expected no WARNINGs, got: {[r.getMessage() for r in handler_warnings]}"
    )
    infos = [
        r for r in caplog.records
        if r.name == _HANDLER_LOGGER
        and r.levelno == logging.INFO
        and "session row gone" in r.getMessage()
    ]
    assert len(infos) == 1, "session-gone must log exactly one INFO"


async def test_session_gone_short_circuits_subsequent_persists() -> None:
    """Once known gone, later LLM calls never reach the persister again."""
    persister = _SessionGonePersister()
    handler = CostCallbackHandler(
        session_id="sess-gone", user_id="u", persister=persister
    )

    await _drive_one_llm_call(handler)
    await handler.flush_pending()
    assert persister.calls == 1

    # Task keeps winding down: more LLM calls settle after the delete.
    await _drive_one_llm_call(handler)
    await _drive_one_llm_call(handler)
    await handler.flush_pending()

    assert persister.calls == 1, (
        "post-detection persists must short-circuit without touching the DB"
    )
    assert handler.persist_failure_count == 0


async def test_session_gone_marker_writer_short_circuits(caplog) -> None:
    """Session-level marker writer detects + stops re-hitting the dead FK."""
    persister = _SessionGonePersister()
    handler = CostCallbackHandler(
        session_id="sess-gone", user_id="u", persister=persister
    )

    with caplog.at_level(logging.DEBUG, logger=_HANDLER_LOGGER):
        first = await handler.write_session_degraded_marker(reason="drain_timeout")
        second = await handler.write_session_degraded_marker(reason="drain_timeout")

    assert first is False
    assert second is False
    # First call may touch the persister (that's how it learns); the second
    # must not.
    assert persister.calls == 1

    handler_warnings = [
        r for r in caplog.records
        if r.name == _HANDLER_LOGGER and r.levelno >= logging.WARNING
    ]
    assert handler_warnings == [], (
        f"expected no WARNINGs, got: {[r.getMessage() for r in handler_warnings]}"
    )


async def test_generic_failure_then_marker_detects_session_gone(caplog) -> None:
    """DB blip first, session deleted before the marker retry.

    The generic failure still counts (real ledger gap while the session
    lived), but the marker's SessionRowGoneError must flip the handler into
    skip mode instead of logging 'degraded marker insert also failed'.
    """
    calls: List[str] = []

    async def persister(record: CostRecord) -> None:
        calls.append(record.node_name)
        if len(calls) == 1:
            raise RuntimeError("db blip")
        raise SessionRowGoneError("session row gone")

    handler = CostCallbackHandler(
        session_id="sess-gone", user_id="u", persister=persister
    )

    with caplog.at_level(logging.DEBUG, logger=_HANDLER_LOGGER):
        await _drive_one_llm_call(handler)
        await handler.flush_pending()

    # Primary attempt + marker attempt, then nothing more.
    assert len(calls) == 2
    # The generic blip is a real persist failure.
    assert handler.persist_failure_count == 1
    # But the marker's FK hit is the session-gone signal, not a second noise line.
    assert "degraded marker insert also failed" not in caplog.text

    # Handler is now in skip mode.
    await _drive_one_llm_call(handler)
    await handler.flush_pending()
    assert len(calls) == 2
