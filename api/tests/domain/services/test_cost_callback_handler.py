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
from app.domain.services.cost_callback_handler import CostCallbackHandler

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
