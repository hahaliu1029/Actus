"""Integration tests for the LLM quality gate path in _evaluate_flush_gate.

Gate off (memory_gate_llm=None) → legacy size-only passthrough, all chunks
emitted with category=None. Covered by test_flush_gate.py / test_flush_integration.py.

Gate on → three new branches to verify:
1. classify + filter_kept_decisions → kept chunks carry category + auto_promoted_at
2. classifier raises → breaker.record_failure() + drop all + pending_batch stays None
3. breaker already OPEN → short-circuit without LLM call
4. daily cap exhausted → drop all kept chunks
5. batch_cap truncates at most N chunks to the LLM
6. cursor still advances on "all dropped" path (so next flush doesn't retry same window)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import AgentConfig, MemoryConfig
from app.domain.models.memory_chunk import FlushBatch, RawChunk
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.services.memory_gate import (
    MemoryGateBreaker,
    _MemoryGateBatchDecision,
    _MemoryGateDecisionWire,
)

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_flow(
    gate_llm=None,
    gate_breaker=None,
    gate_cap=None,
    notification_emitter=None,
    **overrides,
):
    from app.domain.services.flows.planner_react import PlannerReActFlow

    kwargs = {
        "llm": MagicMock(),
        "agent_config": AgentConfig(
            memory=MemoryConfig(
                flush_enabled=True,
                flush_min_steps=1,
                flush_min_new_tokens=500,
            )
        ),
        "session_id": "s-gate",
        "user_id": TEST_USER_ID_FIXED,
        "uow_factory": MagicMock(),
        "browser_accessor": EagerBrowserAccessor(MagicMock()),
        "sandbox_accessor": EagerSandboxAccessor(MagicMock()),
        "search_engine": MagicMock(),
        "mcp_tool": MagicMock(),
        "a2a_tool": MagicMock(),
        "skill_tool": MagicMock(),
        "memory_gate_llm": gate_llm,
        "memory_gate_breaker": gate_breaker,
        "memory_gate_daily_cap": gate_cap,
        "memory_gate_threshold": 0.7,
        "memory_gate_batch_cap": 20,
        "memory_notification_emitter": notification_emitter,
    }
    kwargs.update(overrides)
    return PlannerReActFlow(**kwargs)


def _make_plan(num_completed: int = 2) -> Plan:
    steps = [
        Step(
            id=f"s{i}",
            description=f"Step {i}",
            status=ExecutionStatus.COMPLETED,
            result=f"Result {i}",
            success=True,
        )
        for i in range(num_completed)
    ]
    return Plan(title="P", language="zh", goal="g", steps=steps)


def _rich_messages(n: int):
    return [
        HumanMessage(content=f"User asks about topic {i}. " * 30)
        if i % 2 == 0
        else AIMessage(content=f"Agent replies with details {i}. " * 30)
        for i in range(n)
    ]


def _stub_gate_llm(decisions_wire: list[_MemoryGateDecisionWire]):
    """Build an LLM stub whose with_structured_output.ainvoke returns the
    given batch decision."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=_MemoryGateBatchDecision(decisions=decisions_wire)
    )
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


class _FakeCap:
    def __init__(self, granted: bool = True, cap: int = 100) -> None:
        self.granted = granted
        # Public ``cap`` attribute mirrors the real ``MemoryGateDailyCap.cap``
        # property so the gate's ``getattr(..., "cap", None)`` lookup for
        # notification payload resolution works on the fake too.
        self.cap = cap
        self.calls: list[tuple[str, int]] = []

    async def try_reserve(self, user_id: str, amount: int):
        self.calls.append((user_id, amount))
        return (self.granted, self.cap if self.granted else 0)


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    async def emit(self, *, user_id: str, event_type: str, payload: dict) -> None:
        self.events.append((user_id, event_type, payload))


class TestGateOn:

    async def test_empty_user_id_drops_batch_without_calling_llm(self) -> None:
        """Gate on + user_id='' is a construction error (daily-cap key
        would collapse across users, notification emitter is a no-op).
        Must drop silently and **not** call the LLM."""
        llm = _stub_gate_llm([])
        flow = _make_flow(gate_llm=llm, user_id="")
        dummy = [
            RawChunk(
                content="x", session_id="s", user_id="",
                source="session_flush", metadata={}, content_hash=f"h{i}",
            )
            for i in range(3)
        ]
        kept = await flow._apply_llm_gate(dummy)
        assert kept == []
        llm.with_structured_output.assert_not_called()

    async def test_kept_chunks_carry_category_and_promotion_ts(self) -> None:
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="keep", category="user", confidence=0.9
                ),
                _MemoryGateDecisionWire(
                    chunk_index=1, verdict="drop", category="user", confidence=0.2
                ),
            ]
        )
        flow = _make_flow(gate_llm=llm, gate_cap=_FakeCap())
        msgs = _rich_messages(6)
        await flow._evaluate_flush_gate(msgs, _make_plan())

        batch = flow._pending_flush_batch
        assert batch is not None, "at least one kept → FlushBatch present"
        # Only chunk_index=0 was kept; we can't assume N==2 chunks from
        # _chunk_messages, but what comes out must all be gate-enriched.
        assert len(batch.chunks) >= 1
        for chunk in batch.chunks:
            assert chunk.category in ("user", "rule", "fact")
            assert chunk.auto_promoted_at is not None

    async def test_all_drop_advances_cursor_without_batch(self) -> None:
        """If LLM drops everything, we still advance the cursor—otherwise
        the next flush would resurrect the same content into another LLM
        call. Prevent infinite re-evaluation."""
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="drop", category="user", confidence=0.1
                ),
                _MemoryGateDecisionWire(
                    chunk_index=1, verdict="drop", category="user", confidence=0.1
                ),
            ]
        )
        flow = _make_flow(gate_llm=llm)
        flow._flush_cursor = 0
        msgs = _rich_messages(6)
        await flow._evaluate_flush_gate(msgs, _make_plan())

        assert flow._pending_flush_batch is None
        assert flow._flush_cursor == 6  # advanced

    async def test_below_threshold_treated_as_drop(self) -> None:
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="keep", category="user", confidence=0.5
                ),
            ]
        )
        flow = _make_flow(gate_llm=llm, memory_gate_threshold=0.7)
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())
        assert flow._pending_flush_batch is None


class TestBreaker:

    async def test_llm_exception_records_failure_and_drops(self) -> None:
        breaker = MemoryGateBreaker(threshold=3)
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 500"))
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)

        flow = _make_flow(gate_llm=llm, gate_breaker=breaker)
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())

        assert flow._pending_flush_batch is None
        assert breaker.consecutive_failures == 1
        assert not breaker.is_open()  # below threshold

    async def test_open_breaker_short_circuits_without_llm_call(self) -> None:
        breaker = MemoryGateBreaker(threshold=1, recovery_seconds=3600)
        breaker.record_failure()  # open
        assert breaker.is_open()

        llm = _stub_gate_llm(
            [_MemoryGateDecisionWire(chunk_index=0, verdict="keep",
                                      category="user", confidence=0.9)]
        )
        flow = _make_flow(gate_llm=llm, gate_breaker=breaker)
        flow._flush_cursor = 0
        msgs = _rich_messages(6)
        await flow._evaluate_flush_gate(msgs, _make_plan())

        assert flow._pending_flush_batch is None
        # LLM must not have been called
        llm.with_structured_output.assert_not_called()
        # Cursor advances even on short-circuit drop
        assert flow._flush_cursor == 6

    async def test_success_resets_failure_counter(self) -> None:
        breaker = MemoryGateBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="keep", category="user", confidence=0.9
                ),
            ]
        )
        flow = _make_flow(gate_llm=llm, gate_breaker=breaker, gate_cap=_FakeCap())
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())

        assert breaker.consecutive_failures == 0


class TestDailyCap:

    async def test_cap_exhausted_drops_kept_chunks(self) -> None:
        cap = _FakeCap(granted=False)
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="keep", category="user", confidence=0.9
                ),
            ]
        )
        flow = _make_flow(gate_llm=llm, gate_cap=cap)
        flow._flush_cursor = 0
        msgs = _rich_messages(6)
        await flow._evaluate_flush_gate(msgs, _make_plan())

        assert flow._pending_flush_batch is None
        # Reserved exactly once for the single kept chunk's count
        assert len(cap.calls) == 1

    async def test_cap_not_called_when_everything_dropped(self) -> None:
        """No reservation attempted when LLM drops everything—avoid burning
        quota budget on chunks we weren't going to promote anyway."""
        cap = _FakeCap(granted=True)
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="drop", category="user", confidence=0.1
                ),
            ]
        )
        flow = _make_flow(gate_llm=llm, gate_cap=cap)
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())

        assert cap.calls == []


class TestNotificationEmission:

    async def test_breaker_rising_edge_emits_gate_paused(self) -> None:
        """First time breaker hits threshold → notification fires exactly
        once. Test calls ``_apply_llm_gate`` directly so we don't interact
        with cursor/size-gate state between failures — the emission
        contract we're verifying is local to the gate function."""
        breaker = MemoryGateBreaker(threshold=2)
        emitter = _FakeEmitter()

        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)

        flow = _make_flow(
            gate_llm=llm, gate_breaker=breaker, notification_emitter=emitter,
        )
        dummy_chunks = [
            RawChunk(
                content="some content",
                session_id="s-gate",
                user_id=TEST_USER_ID_FIXED,
                source="session_flush",
                metadata={},
                content_hash=f"h{i}",
            )
            for i in range(2)
        ]

        # Failure 1: under threshold → no emit
        await flow._apply_llm_gate(dummy_chunks)
        assert emitter.events == []

        # Failure 2: crosses threshold → rising edge → emit once
        await flow._apply_llm_gate(dummy_chunks)
        assert len(emitter.events) == 1
        uid, event_type, payload = emitter.events[0]
        assert uid == TEST_USER_ID_FIXED
        assert event_type == "memory_gate_paused"
        assert payload["consecutive_failures"] == 2
        assert "boom" in payload["last_error"]

        # Failure 3: breaker already OPEN → short-circuits without LLM
        # call, so record_failure is never invoked → no re-emission.
        await flow._apply_llm_gate(dummy_chunks)
        assert len(emitter.events) == 1  # unchanged

    async def test_quota_exhausted_emits_quota_exceeded(self) -> None:
        emitter = _FakeEmitter()
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="keep", category="user",
                    confidence=0.9,
                ),
            ]
        )
        flow = _make_flow(
            gate_llm=llm,
            gate_cap=_FakeCap(granted=False, cap=100),
            notification_emitter=emitter,
        )
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())

        assert len(emitter.events) == 1
        uid, event_type, payload = emitter.events[0]
        assert event_type == "quota_exceeded"
        assert payload["cap"] == 100
        assert payload["attempted"] >= 1

    async def test_no_emit_when_gate_succeeds(self) -> None:
        emitter = _FakeEmitter()
        breaker = MemoryGateBreaker(threshold=3)
        llm = _stub_gate_llm(
            [
                _MemoryGateDecisionWire(
                    chunk_index=0, verdict="keep", category="user",
                    confidence=0.95,
                ),
            ]
        )
        flow = _make_flow(
            gate_llm=llm,
            gate_breaker=breaker,
            gate_cap=_FakeCap(),
            notification_emitter=emitter,
        )
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())
        assert emitter.events == []

    async def test_emitter_exception_does_not_break_flush_path(self) -> None:
        """Emitter throws → exception swallowed, gate still returns
        kept chunks. Notifications are advisory, never load-bearing."""
        class _ExplodingEmitter:
            async def emit(self, **_):
                raise RuntimeError("DB is down")

        breaker = MemoryGateBreaker(threshold=1)
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)

        flow = _make_flow(
            gate_llm=llm,
            gate_breaker=breaker,
            notification_emitter=_ExplodingEmitter(),
        )
        # Should NOT raise even though emitter does
        await flow._evaluate_flush_gate(_rich_messages(6), _make_plan())
        assert flow._pending_flush_batch is None


class TestBatchCap:

    async def test_over_cap_truncates_to_cap(self) -> None:
        """Batch cap = 2, _chunk_messages yields >2 chunks → LLM sees only
        first 2. Rest are silently not evaluated this flush (cursor still
        advances past them so they don't re-enter)."""
        import re

        llm = _stub_gate_llm([])  # LLM returns empty; test just checks what was sent
        flow = _make_flow(gate_llm=llm, memory_gate_batch_cap=2)
        msgs = _rich_messages(20)  # long → many chunks after _chunk_messages
        await flow._evaluate_flush_gate(msgs, _make_plan())

        structured = llm.with_structured_output.return_value
        # 用 `\[chunk \d+\]` 精确匹配 real chunk marker，排除 prompt 头里
        # 说明文字"以 [chunk N] 开头"这种文本字面。
        call = structured.ainvoke.await_args_list[0]
        user_text = call[0][0][1][1]
        marker_count = len(re.findall(r"\[chunk \d+\]", user_text))
        assert marker_count <= 2, f"Expected ≤2 chunk markers, got {marker_count}"
