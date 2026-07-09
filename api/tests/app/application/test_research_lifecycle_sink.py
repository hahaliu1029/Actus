"""C7 PR5 — ResearchLifecycleSink：AND 门/投递/可观测 skip/outcome 映射（spec §4.5 R15）。"""
from unittest.mock import AsyncMock

import pytest

from app.application.services.research_lifecycle_sink import ResearchLifecycleSink
from app.domain.models.app_config import LifecycleRuntimeConfig
from app.domain.models.lifecycle import LifecycleContractError
from app.interfaces.schemas.subagent import ChildOutcome


class _FakeCounter:
    def __init__(self) -> None:
        self.count = 0

    def add(self, n: int, attributes=None) -> None:
        self.count += n


class _FakeRepo:
    def __init__(self) -> None:
        self.persisted = []

    async def add_event(self, session_id, event) -> None:
        self.persisted.append((session_id, event))


class _FakeUoW:
    def __init__(self, repo) -> None:
        self.session = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _sink(*, master=True, sub=True, emit_returns="msg-1"):
    repo = _FakeRepo()
    emit = AsyncMock(return_value=emit_returns)
    counter = _FakeCounter()
    flags = LifecycleRuntimeConfig(
        lifecycle_events_enabled=master, lifecycle_subagent_events_enabled=sub,
    )
    sink = ResearchLifecycleSink(
        emit_event=emit,
        uow_factory=lambda: _FakeUoW(repo),
        flags_getter=lambda: flags,
        skip_counter=counter,
    )
    return sink, emit, repo, counter


@pytest.mark.asyncio
async def test_delivers_started_when_parent_task_active():
    sink, emit, repo, counter = _sink()
    ok = await sink.child_started(parent_session_id="parent-1", child_session_id="child-1")
    assert ok is True
    emit.assert_awaited_once()
    ev = emit.await_args.args[1]
    assert (ev.type, ev.lifecycle_type.value, ev.event.value, ev.unit_id) == (
        "lifecycle", "subagent", "started", "child-1",
    )
    assert ev.parent_unit_id == "parent-1"
    assert [sid for sid, _ in repo.persisted] == ["parent-1"]   # PG persist 跟随成功投递
    assert counter.count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,expected_kind,expected_reason",
    [
        (ChildOutcome.COMPLETED, "completed", None),
        (ChildOutcome.FAILED, "failed", "worker_failed"),
        (ChildOutcome.TIMED_OUT, "failed", "watchdog_timeout"),
        (ChildOutcome.CANCELLED, "cancelled", None),
        (ChildOutcome.WAITING, "failed", "waiting_unsupported"),
    ],
)
async def test_child_done_outcome_mapping(outcome, expected_kind, expected_reason):
    sink, emit, _, _ = _sink()
    await sink.child_done(parent_session_id="p", child_session_id="c", outcome=outcome)
    ev = emit.await_args.args[1]
    assert (ev.event.value, ev.reason) == (expected_kind, expected_reason)


@pytest.mark.asyncio
async def test_skip_is_observable_not_silent(caplog):
    # R15 核心：无活跃 task → 计数器 + 结构化日志；不 persist
    sink, emit, repo, counter = _sink(emit_returns=None)
    with caplog.at_level("WARNING"):
        ok = await sink.child_started(parent_session_id="p", child_session_id="c")
    assert ok is False
    assert counter.count == 1
    assert repo.persisted == []
    assert any("lifecycle_research_sink_skipped" in r.message for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("master,sub", [(False, True), (True, False), (False, False)])
async def test_and_gate_zero_construction(master, sub):
    # AND 门（R10#A9）：任一 off → 零构造零 emit 零计数（flag-off 不是 skip，是不存在）
    sink, emit, repo, counter = _sink(master=master, sub=sub)
    ok = await sink.child_started(parent_session_id="p", child_session_id="c")
    assert ok is False
    emit.assert_not_awaited()
    assert counter.count == 0 and repo.persisted == []


@pytest.mark.asyncio
async def test_sink_never_raises(caplog):
    sink, emit, _, _ = _sink()
    emit.side_effect = RuntimeError("redis down")
    with caplog.at_level("WARNING"):
        ok = await sink.child_started(parent_session_id="p", child_session_id="c")
    assert ok is False  # 观测面永不破坏 research 主流程


@pytest.mark.asyncio
async def test_sink_contains_event_build_failure(monkeypatch, caplog):
    # T15 flip 前硬化（final review R1）：构造期 LifecycleContractError（词表漂移等）
    # 同样不得穿透 research 主流程——「sink 永不 raise」覆盖构造+投递全程
    import app.application.services.research_lifecycle_sink as mod

    def _boom(*args, **kwargs):
        raise LifecycleContractError("vocab drift")

    monkeypatch.setattr(mod, "build_lifecycle_event", _boom)
    sink, emit, repo, counter = _sink()
    with caplog.at_level("WARNING"):
        ok_started = await sink.child_started(parent_session_id="p", child_session_id="c")
        ok_done = await sink.child_done(
            parent_session_id="p", child_session_id="c", outcome=ChildOutcome.COMPLETED,
        )
    assert (ok_started, ok_done) == (False, False)
    emit.assert_not_awaited()  # 构造失败 → 不投递不 persist，仅告警日志
    assert repo.persisted == []
    assert counter.count == 0  # skip 计数器语义保留给「无活跃 task」投递缺口
    assert sum("build failed" in r.message for r in caplog.records) == 2


def test_service_call_sites_pinned():
    # 3 处 yield 站点（started :839 / done :823+:877 区域）前都要调 sink
    from pathlib import Path
    import app.application.services.subagent_research_service as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert src.count("child_started(") >= 1
    assert src.count("child_done(") >= 2
