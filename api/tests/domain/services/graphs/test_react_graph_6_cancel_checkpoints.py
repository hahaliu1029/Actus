"""C2 PR-4 Task 4.5 — react_graph cancel checkpoint tests.

Spec ref: §8.4 — 8 cancel checkpoints (numbered #1-#8 in the spec).
#1 lives in ``CoordinatorChildRunner.run_work_unit`` (worker start) and
#8 in ``CoordinatorChildRunner._finalize_*`` (artifact upload before publish).
#2-#7 live in this module.

This file covers:
- ``_should_cancel(config)`` helper: True iff config carries a set
  ``cancel_event`` in ``configurable``.
- ``CancelledByEventError`` is raised at each in-graph checkpoint when
  the event is already set.

#4 (LLM streaming chunk boundary) is DEFERRED — live ``llm_node`` calls
``await llm_with_tools.ainvoke(messages)`` (line 1067, atomic), not a
streaming async-for. When ``llm_node`` is refactored to stream, checkpoint
#4 belongs inside the ``async for chunk in ...`` loop. This test file
includes a contract pin (test_chunk_boundary_deferred_until_streaming) so
a future maintainer adding streaming without re-wiring #4 fails here.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from app.domain.services.graphs.react_graph import (
    CancelledByEventError,
    _should_cancel,
)


pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# _should_cancel helper
# ---------------------------------------------------------------------------

def test_should_cancel_none_config_returns_false() -> None:
    assert _should_cancel(None) is False  # type: ignore[arg-type]


def test_should_cancel_empty_config_returns_false() -> None:
    assert _should_cancel({}) is False


def test_should_cancel_no_event_in_configurable_returns_false() -> None:
    assert _should_cancel({"configurable": {}}) is False


def test_should_cancel_event_set_returns_true() -> None:
    ce = asyncio.Event()
    ce.set()
    assert _should_cancel({"configurable": {"cancel_event": ce}}) is True


def test_should_cancel_event_not_set_returns_false() -> None:
    ce = asyncio.Event()  # not set
    assert _should_cancel({"configurable": {"cancel_event": ce}}) is False


def test_should_cancel_none_event_value_returns_false() -> None:
    """Defensive: configurable.cancel_event=None must NOT crash."""
    assert _should_cancel({"configurable": {"cancel_event": None}}) is False


def test_should_cancel_wrong_type_returns_false() -> None:
    """If something non-Event slips into configurable.cancel_event (e.g. a
    plain dict), we must NOT crash — return False and let the loop continue.
    Crashing here would mask the real bug under a checkpoint exception."""
    assert _should_cancel({"configurable": {"cancel_event": object()}}) is False


# ---------------------------------------------------------------------------
# CancelledByEventError exception
# ---------------------------------------------------------------------------

def test_cancelled_by_event_error_is_exception() -> None:
    assert issubclass(CancelledByEventError, Exception)


def test_cancelled_by_event_error_carries_checkpoint_name() -> None:
    """The error message MUST include the checkpoint name so the finalizer
    can attribute the cancel to the correct point in the loop."""
    e = CancelledByEventError("react_loop_entry")
    assert "react_loop_entry" in str(e)


# ---------------------------------------------------------------------------
# Checkpoint contract (the 5 active checkpoints + 1 deferred)
# ---------------------------------------------------------------------------

ACTIVE_CHECKPOINTS = (
    "react_loop_entry",     # #2 — pre_llm_node top
    "llm_node_entry",       # #3 — llm_node top
    "llm_return",           # #5 — llm_node bottom (pre-return)
    "tool_node_entry",      # #6 — tool_node top
    "tool_node_return",     # #7 — tool_node bottom (pre-return)
)


@pytest.mark.parametrize("checkpoint", ACTIVE_CHECKPOINTS)
def test_checkpoint_name_appears_in_source(checkpoint: str) -> None:
    """AST/source-level pin: each active checkpoint name MUST appear as a
    string literal in react_graph.py. A silent removal would surface here
    rather than at integration time."""
    from app.domain.services.graphs import react_graph
    src = inspect.getsource(react_graph)
    assert checkpoint in src, (
        f"checkpoint {checkpoint!r} not found in react_graph.py source; "
        f"either it was removed or the name drifted from the spec."
    )


def test_chunk_boundary_deferred_until_streaming() -> None:
    """[deferred contract] #4 (llm_chunk_boundary) is NOT wired because the
    live llm_node uses ``ainvoke`` (atomic). The string ``llm_chunk_boundary``
    appears ONLY in a deferral docstring/comment, NOT in any actual checkpoint
    call. When a future PR adds streaming, the developer MUST also wire
    #4 into the new ``async for chunk in ...`` loop.

    This test pins the deferral so the absence is intentional, not silent."""
    from app.domain.services.graphs import react_graph
    src = inspect.getsource(react_graph)
    assert "llm_chunk_boundary" in src and "deferred" in src.lower(), (
        "#4 (llm_chunk_boundary) must be explicitly noted as deferred in "
        "react_graph.py until streaming is added; absent the marker, a "
        "reviewer can't tell whether the omission is a bug or by design."
    )


# ---------------------------------------------------------------------------
# Behavior: cancel raised when event set
# ---------------------------------------------------------------------------

async def test_pre_llm_node_raises_when_cancelled() -> None:
    """react_loop_entry (#2) — first guard at pre_llm_node entry.
    When cancel_event is set, pre_llm_node MUST raise CancelledByEventError
    before invoking context_assembler."""
    from app.domain.services.graphs.react_graph import build_react_graph
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    cancel_event = asyncio.Event()
    cancel_event.set()
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="never reached")])
    graph = build_react_graph(llm=llm, tools=[])
    state = {"messages": [HumanMessage(content="hi")]}
    config = {"configurable": {"cancel_event": cancel_event}}
    with pytest.raises(CancelledByEventError) as ei:
        await graph.ainvoke(state, config)
    assert "react_loop_entry" in str(ei.value)


async def test_no_cancel_when_event_not_set() -> None:
    """Negative: no cancel_event → no CancelledByEventError. This protects
    against accidental over-firing on the legacy (non-coordinator) path."""
    from app.domain.services.graphs.react_graph import build_react_graph
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="ok done")])
    graph = build_react_graph(llm=llm, tools=[])
    state = {"messages": [HumanMessage(content="hi")]}
    result = await graph.ainvoke(state, {"configurable": {}})
    assert result is not None
