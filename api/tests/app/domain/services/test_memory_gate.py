"""Unit tests for MemoryGateClassifier, MemoryGateBreaker, MemoryGateDailyCap.

Real LLM 调用走 ``pytest.mark.slow`` 的 eval harness；这里只验证：
- batch prompt 结构（每 chunk 一行）
- 空 batch 不打 LLM
- structured output 正确映射到 domain decisions
- filter_kept_decisions 的阈值语义
- breaker 的 open/close 状态机 + rising-edge 信号
- daily cap 原子回滚 + 跨日 TTL
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.memory_gate import (
    MemoryGateBreaker,
    MemoryGateClassifier,
    MemoryGateDailyCap,
    MemoryGateDecision,
    MemoryGateInput,
    _MemoryGateBatchDecision,
    _MemoryGateDecisionWire,
    filter_kept_decisions,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _wire_decision(
    idx: int, verdict: str, category: str = "user", confidence: float = 0.9
) -> _MemoryGateDecisionWire:
    return _MemoryGateDecisionWire(
        chunk_index=idx,
        verdict=verdict,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _make_llm(wire_decisions: list[_MemoryGateDecisionWire]) -> MagicMock:
    """Build a MagicMock BaseChatModel whose with_structured_output binds
    to an AsyncMock returning the given batch decision."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=_MemoryGateBatchDecision(decisions=wire_decisions)
    )
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


class TestClassify:

    async def test_empty_batch_does_not_call_llm(self) -> None:
        llm = _make_llm([])
        clf = MemoryGateClassifier(llm)
        result = await clf.classify([])
        assert result == []
        llm.with_structured_output.assert_not_called()

    async def test_maps_wire_to_domain_decisions(self) -> None:
        llm = _make_llm(
            [
                _wire_decision(0, "keep", "user", 0.92),
                _wire_decision(1, "drop", "user", 0.55),
            ]
        )
        clf = MemoryGateClassifier(llm)
        chunks = [
            MemoryGateInput(chunk_index=0, text="记住我用中文"),
            MemoryGateInput(chunk_index=1, text="刚才 debug 打印 x=3"),
        ]
        result = await clf.classify(chunks)
        assert result == [
            MemoryGateDecision(
                chunk_index=0,
                verdict="keep",
                category="user",
                confidence=0.92,
            ),
            MemoryGateDecision(
                chunk_index=1,
                verdict="drop",
                category="user",
                confidence=0.55,
            ),
        ]

    async def test_prompt_includes_chunk_markers_and_text(self) -> None:
        """Prompt 必须把 chunk_index 和 text 喂到 LLM，让模型能在输出
        里按 index 引用——否则回来的 chunk_index 可能错位。"""
        llm = _make_llm([_wire_decision(7, "drop", "user", 0.1)])
        clf = MemoryGateClassifier(llm)
        chunks = [
            MemoryGateInput(chunk_index=7, text="用户说：以后别 mock 数据库"),
        ]
        await clf.classify(chunks)

        structured = llm.with_structured_output.return_value
        call_args = structured.ainvoke.call_args[0][0]
        # messages 形如 [("system", ...), ("user", ...)]
        assert call_args[0][0] == "system"
        assert call_args[1][0] == "user"
        user_text = call_args[1][1]
        assert "[chunk 7]" in user_text
        assert "以后别 mock 数据库" in user_text


class TestFilterKeptDecisions:

    def test_drops_verdict_drop(self) -> None:
        decisions = [
            MemoryGateDecision(0, "keep", "user", 0.9),
            MemoryGateDecision(1, "drop", "user", 0.95),  # 高置信度但 drop
        ]
        kept = filter_kept_decisions(decisions, threshold=0.7)
        assert [d.chunk_index for d in kept] == [0]

    def test_drops_below_threshold(self) -> None:
        decisions = [
            MemoryGateDecision(0, "keep", "user", 0.69),
            MemoryGateDecision(1, "keep", "fact", 0.71),
        ]
        kept = filter_kept_decisions(decisions, threshold=0.7)
        assert [d.chunk_index for d in kept] == [1]

    def test_boundary_at_threshold_is_kept(self) -> None:
        """>= threshold 语义——0.7 应保留，不要把 float 比较搞反。"""
        decisions = [MemoryGateDecision(0, "keep", "user", 0.7)]
        assert len(filter_kept_decisions(decisions, threshold=0.7)) == 1

    def test_empty_decisions_returns_empty(self) -> None:
        assert filter_kept_decisions([], threshold=0.7) == []


class TestMemoryGateBreaker:

    def test_starts_closed(self) -> None:
        br = MemoryGateBreaker()
        assert br.is_open() is False
        assert br.consecutive_failures == 0

    def test_opens_after_threshold_failures(self) -> None:
        br = MemoryGateBreaker(threshold=3)
        just_opened_1 = br.record_failure()  # 1
        just_opened_2 = br.record_failure()  # 2
        just_opened_3 = br.record_failure()  # 3 → rising edge
        assert (just_opened_1, just_opened_2, just_opened_3) == (False, False, True)
        assert br.is_open() is True

    def test_record_failure_rising_edge_only_fires_once(self) -> None:
        """Caller emits a notification on the rising edge; we don't want
        every subsequent failure to also trigger a notification or the
        user's notification tray fills up."""
        br = MemoryGateBreaker(threshold=2)
        assert br.record_failure() is False
        assert br.record_failure() is True
        # 4th / 5th / ... failures should NOT re-signal "just opened"
        assert br.record_failure() is False
        assert br.record_failure() is False

    def test_record_success_resets(self) -> None:
        br = MemoryGateBreaker(threshold=2)
        br.record_failure()
        br.record_failure()
        assert br.is_open() is True
        br.record_success()
        assert br.is_open() is False
        assert br.consecutive_failures == 0

    def test_post_cooldown_probe_allows_one_attempt(self) -> None:
        """After ``recovery_seconds`` passes is_open returns False so the
        next call can try; result of that call decides whether to stay
        closed or re-open."""
        br = MemoryGateBreaker(threshold=1, recovery_seconds=0.0)
        br.record_failure()
        assert br.is_open() is False  # recovery immediately over
        # Simulate a probe that succeeded
        br.record_success()
        assert br.is_open() is False

    def test_cooldown_until_only_set_when_open(self) -> None:
        br = MemoryGateBreaker(threshold=1, recovery_seconds=60.0)
        assert br.cooldown_until() is None
        br.record_failure()
        cu = br.cooldown_until()
        assert cu is not None
        # Should be within ~60s of now
        delta = (cu - datetime.now(tz=timezone.utc)).total_seconds()
        assert 0 <= delta <= 61


class _FakeRedis:
    """Tiny async fake — enough for try_reserve/INCRBY+EXPIRE semantics."""

    def __init__(self) -> None:
        self._store: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def incrby(self, key: str, amount: int) -> int:
        self._store[key] = self._store.get(key, 0) + amount
        return self._store[key]

    async def expire(self, key: str, seconds: int) -> bool:
        self.expire_calls.append((key, seconds))
        return True

    async def get(self, key: str) -> str | None:
        return str(self._store[key]) if key in self._store else None


class TestMemoryGateDailyCap:

    async def test_within_cap_grants(self) -> None:
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=100)
        granted, remaining = await cap.try_reserve("u1", 5)
        assert granted is True
        assert remaining == 95

    async def test_probe_amount_zero_does_not_increment(self) -> None:
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=100)
        granted, _ = await cap.try_reserve("u1", 0)
        assert granted is True
        # confirmed no writes
        assert not redis._store

    async def test_cap_reached_denies_and_rolls_back(self) -> None:
        """超限时必须把 counter 退回到 cap，否则同日后续调用读到 "已满"，
        等同于无声黑洞 —— 这是 gate quota 最容易写错的一点。"""
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=10)
        await cap.try_reserve("u1", 8)
        granted, remaining = await cap.try_reserve("u1", 5)  # overflows by 3
        assert granted is False
        assert remaining == 0
        # Post-rollback: counter == cap, not cap+3
        key = cap._key("u1")
        assert redis._store[key] == 10

    async def test_exact_boundary_granted(self) -> None:
        """用满最后一格不应 overflow。"""
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=10)
        granted, remaining = await cap.try_reserve("u1", 10)
        assert granted is True
        assert remaining == 0

    async def test_sets_ttl_on_increment(self) -> None:
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=100)
        await cap.try_reserve("u1", 1)
        assert redis.expire_calls == [(cap._key("u1"), cap.TTL_SECONDS)]

    async def test_different_users_isolated(self) -> None:
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=5)
        await cap.try_reserve("u1", 5)  # cap u1
        granted, remaining = await cap.try_reserve("u2", 3)
        assert granted is True
        assert remaining == 2

    async def test_key_is_date_scoped(self) -> None:
        redis = _FakeRedis()
        cap = MemoryGateDailyCap(redis, cap=5)
        day1 = datetime(2026, 4, 17, tzinfo=timezone.utc)
        day2 = datetime(2026, 4, 18, tzinfo=timezone.utc)
        k1 = cap._key("u1", today=day1)
        k2 = cap._key("u1", today=day2)
        assert k1 != k2
        assert "20260417" in k1 and "20260418" in k2
