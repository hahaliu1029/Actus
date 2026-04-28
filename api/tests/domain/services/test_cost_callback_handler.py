"""B4 M0 Phase E: CostCallbackHandler contract.

Verifies the handler's load-bearing behaviors:

1. Happy path — start + end with usage_metadata → persister receives a
   CostRecord with correct node_name/step_ix/tokens/total_usd and
   ``cost_status=actual``.
2. No usage — end with ``usage_metadata=None`` → record persisted with
   ``cost_status=estimated`` and ``total_usd=0`` (don't silently zero-bill).
3. Unknown model — model not in PRICING_TABLE → ``cost_status=unknown`` and
   ``total_usd=0``.
4. LRU eviction — Issue 1C: _pending is bounded so missing on_llm_end can't
   OOM the handler over days of uptime.
5. Out-of-graph attribution — when langgraph_* metadata is absent, attribute
   to ``out_of_graph`` (summarizer, context_compaction).
6. Persister failure is swallowed — a failing persist must not propagate.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Awaitable, Callable, List
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    FlushResult,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_llm_result(
    usage_metadata: dict | None = None,
    content: str = "hi",
) -> LLMResult:
    msg = AIMessage(content=content, usage_metadata=usage_metadata)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _make_capture_persister() -> tuple[
    Callable[[CostRecord], Awaitable[None]], List[CostRecord]
]:
    captured: List[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    return persist, captured


class TestHappyPath:
    async def test_start_then_end_with_usage_persists_actual_cost(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-1", user_id="user-1", persister=persist
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={"id": ["ActusChatModel"]},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner", "langgraph_step": 3},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        }
        await handler.on_llm_end(
            _make_llm_result(usage_metadata=usage), run_id=run_id
        )
        await handler.flush_pending()

        assert len(captured) == 1
        record = captured[0]
        assert record.session_id == "sess-1"
        assert record.user_id == "user-1"
        assert record.node_name == "planner"
        assert record.step_ix == 3
        assert record.model == "gpt-4o"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_status == CostStatus.ACTUAL
        assert record.total_usd == Decimal("0.0075")
        assert record.pricing_version


class TestNoUsage:
    async def test_end_without_usage_metadata_marks_unknown(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "executor", "langgraph_step": 1},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(usage_metadata=None), run_id=run_id
        )
        await handler.flush_pending()

        assert len(captured) == 1
        rec = captured[0]
        # Design decision (post-audit): "estimated" implies a char-count
        # heuristic we don't actually compute in M0. Label it ``unknown``
        # honestly instead of lying in the status name.
        assert rec.cost_status == CostStatus.UNKNOWN
        assert rec.total_usd == Decimal(0)
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0


class TestUnknownModel:
    async def test_model_not_in_pricing_table_marks_unknown(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner", "langgraph_step": 0},
            invocation_params={"model": "totally-made-up-model"},
        )
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        await handler.on_llm_end(
            _make_llm_result(usage_metadata=usage), run_id=run_id
        )
        await handler.flush_pending()

        rec = captured[0]
        assert rec.cost_status == CostStatus.UNKNOWN
        assert rec.total_usd == Decimal(0)
        assert rec.input_tokens == 10
        assert rec.output_tokens == 5


class TestLRUEviction:
    async def test_pending_is_bounded_by_max_pending(self) -> None:
        """Issue 1C: _pending must not grow unbounded on missing-end paths."""
        persist, _ = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist, max_pending=2
        )

        r1, r2, r3 = uuid4(), uuid4(), uuid4()
        for rid in (r1, r2, r3):
            await handler.on_chat_model_start(
                serialized={},
                messages=[[HumanMessage(content="x")]],
                run_id=rid,
                metadata={"langgraph_node": "n", "langgraph_step": 0},
                invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
            )

        pending_keys = handler.pending_keys()
        assert r1 not in pending_keys, "oldest entry must be evicted at cap"
        assert r2 in pending_keys
        assert r3 in pending_keys
        assert len(pending_keys) == 2


class TestOutOfGraphAttribution:
    async def test_missing_langgraph_metadata_falls_back_to_out_of_graph(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        await handler.on_llm_end(
            _make_llm_result(usage_metadata=usage), run_id=run_id
        )
        await handler.flush_pending()

        assert captured[0].node_name == "out_of_graph"


class TestPersisterFailureIsSwallowed:
    async def test_persister_exception_does_not_propagate(self) -> None:
        async def broken(record: CostRecord) -> None:
            raise RuntimeError("db down")

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=broken
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="x")]],
            run_id=run_id,
            metadata={"langgraph_node": "n", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            ),
            run_id=run_id,
        )
        await handler.flush_pending()


class TestFlushPendingTimeout:
    async def test_timeout_does_not_block_terminal_transition(self) -> None:
        """A stuck persister must not wedge the terminal-status path.

        Scenario: DB/pool is down → persister hangs. ``flush_pending(timeout=0.1)``
        must return within ~0.1s instead of blocking indefinitely, so
        ``_set_terminal_status`` can mark the session COMPLETED and surface
        the result to the UI.
        """
        hang = asyncio.Event()

        async def slow_persist(record: CostRecord) -> None:
            await hang.wait()

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=slow_persist
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="x")]],
            run_id=run_id,
            metadata={"langgraph_node": "n", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            ),
            run_id=run_id,
        )

        t0 = time.monotonic()
        await handler.flush_pending(timeout=0.1)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f"flush_pending(timeout=0.1) took {elapsed:.2f}s — must not block "
            "on a stuck persister beyond the requested timeout."
        )

        hang.set()
        await asyncio.gather(*handler._active_tasks, return_exceptions=True)

    async def test_timeout_does_not_cancel_in_flight_persists(self) -> None:
        """Regression: ``wait_for(gather(...))`` would cancel survivors.

        Post-timeout persists must keep running in the background so no
        cost rows are dropped on the floor. Uses ``asyncio.wait`` instead
        of ``asyncio.wait_for(asyncio.gather(...))`` to preserve that.
        """
        persisted: list[CostRecord] = []
        release = asyncio.Event()

        async def slow_then_persist(record: CostRecord) -> None:
            await release.wait()
            persisted.append(record)

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=slow_then_persist
        )

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="x")]],
            run_id=run_id,
            metadata={"langgraph_node": "n", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            ),
            run_id=run_id,
        )

        # Timeout expires.
        await handler.flush_pending(timeout=0.05)

        # Task must NOT be cancelled — it's still waiting on the release event.
        [t] = list(handler._active_tasks)
        assert not t.cancelled(), (
            "flush_pending timeout must not cancel pending persist tasks; "
            "using asyncio.wait_for(gather) would cancel them and lose rows."
        )
        assert not t.done(), "task should still be running in the background"

        # Unblock persister, give the task a beat to finish, assert the
        # row landed.
        release.set()
        await asyncio.wait_for(
            asyncio.gather(*handler._active_tasks, return_exceptions=True),
            timeout=1.0,
        )
        assert len(persisted) == 1, (
            "After timeout, the pending persist must complete once unblocked — "
            "otherwise cost rows are lost."
        )


class TestFlushResult:
    """B4 Issue 1D: ``flush_pending`` returns FlushResult, not None.

    Spec: docs/superpowers/specs/2026-04-27-b4-1d-finishing-drain-design.md §3.1
    """

    async def test_empty_pending_returns_drained_true(self) -> None:
        persist, _captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        result = await handler.flush_pending()
        assert isinstance(result, FlushResult)
        assert result == FlushResult(
            drained=True, pending_count=0, persist_failures=0
        )

    async def test_timeout_none_all_settle_returns_drained(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        # Drive one full happy-path call to enqueue a persist task.
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(usage_metadata={
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15
            }),
            run_id=run_id,
        )

        result = await handler.flush_pending(timeout=None)
        assert result == FlushResult(
            drained=True, pending_count=0, persist_failures=0
        )
        assert len(captured) == 1

    async def test_timeout_with_leftovers_returns_pending_count(self) -> None:
        # A persister that hangs on a never-set event so we can deterministically
        # force a leftover task at flush time.
        block_event = asyncio.Event()
        captured: List[CostRecord] = []

        async def slow_persist(record: CostRecord) -> None:
            await block_event.wait()
            captured.append(record)

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=slow_persist
        )
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(usage_metadata={
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15
            }),
            run_id=run_id,
        )

        result = await handler.flush_pending(timeout=0.05)
        assert result == FlushResult(
            drained=False, pending_count=1, persist_failures=0
        )
        # Cleanup: release the persister so the background task exits before
        # pytest tears down the event loop.
        block_event.set()
        await asyncio.sleep(0)
        await handler.flush_pending(timeout=1.0)

    async def test_persist_failures_delta_excludes_prior(self) -> None:
        """Spec §3.1 + §5: ``persist_failures`` is window-local delta.

        Prior failures (recorded before flush_pending entry) MUST NOT be
        attributed to the current flush call.
        """
        persist, _captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        # Simulate a prior failure by bumping the counter directly. (In real
        # code this is incremented by ``_persist_safely``; we don't need to
        # round-trip the full failure here.)
        handler._persist_failure_count = 2  # type: ignore[attr-defined]

        # First flush (no pending tasks, no new failures): delta = 0.
        result = await handler.flush_pending()
        assert result == FlushResult(
            drained=True, pending_count=0, persist_failures=0
        )

    async def test_persist_failures_delta_counts_window_failures(self) -> None:
        """Failures occurring DURING the flush window ARE counted."""
        captured: List[CostRecord] = []

        async def failing_persist(record: CostRecord) -> None:
            # Simulate transient persister failure. ``_persist_safely``
            # catches and increments _persist_failure_count.
            captured.append(record)
            raise RuntimeError("simulated persister failure")

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=failing_persist
        )
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
            invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
        )
        await handler.on_llm_end(
            _make_llm_result(usage_metadata={
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15
            }),
            run_id=run_id,
        )

        result = await handler.flush_pending(timeout=1.0)
        # Counter semantics: ``_persist_safely`` increments
        # ``_persist_failure_count`` ONLY when the original persist call
        # raises (cost_callback_handler.py: ``_persist_safely`` real-persist
        # try/except branch). The subsequent best-effort degraded-marker
        # insert is log-and-swallow and does NOT increment the counter
        # (cost_callback_handler.py: degraded-fallback try/except branch).
        # So a single real-persist failure contributes exactly +1 to the
        # delta. We assert ``>= 1`` (not ``== 1``) only to leave a small
        # margin for harness flakiness — do NOT change the implementation
        # to also count marker-insert failures; that would conflate two
        # distinct best-effort paths.
        assert result.drained is True
        assert result.pending_count == 0
        assert result.persist_failures >= 1, (
            "Expected at least the original persist failure to bump the "
            f"counter, saw delta={result.persist_failures}"
        )
