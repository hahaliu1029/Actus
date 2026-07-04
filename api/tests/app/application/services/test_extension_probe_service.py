"""B9 探测状态机/退避/并发防护（spec §3.3/§3.4/§13）。"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.extension_probe_service import (
    ExtensionProbeService,
    ProbeBusyError,
    ProbeDisabledError,
    ProbeGoneError,
    ProbeOutcome,
    ProbeRecord,
)
from app.domain.models.app_config import (
    A2AConfig, A2AServerConfig, AppConfig, MCPConfig, MCPServerConfig, MCPTransport,
)
from app.domain.models.runtime_extension import LivenessSnapshot

T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.now = T0
    def __call__(self) -> datetime:
        return self.now
    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeProber:
    def __init__(self) -> None:
        self.outcome = ProbeOutcome(ok=True, latency_ms=42, tool_count=3)
        self.calls: list[tuple[str, str]] = []
        self.seen_configs: list = []
        self.gate: asyncio.Event | None = None   # 设置后探测挂起直到 set（midflight 场景）
    async def probe_mcp(self, server_name, config):
        self.calls.append(("mcp", server_name))
        self.seen_configs.append(config)
        if self.gate:
            await self.gate.wait()
        return self.outcome
    async def probe_a2a(self, config):
        self.calls.append(("a2a", config.id))
        self.seen_configs.append(config)
        if self.gate:
            await self.gate.wait()
        return self.outcome


class FakeLiveness:
    def __init__(self) -> None:
        self.active: dict[tuple[str, str], set[str]] = {}
    def snapshot(self) -> LivenessSnapshot:
        return LivenessSnapshot(
            active={k: frozenset(v) for k, v in self.active.items()}, degraded=False)


class World:
    """可变 config 世界：测试原地改动模拟排队/探测期间的世界变化（R19#1/R18#3 场景）。"""
    def __init__(self) -> None:
        self.mcp: dict[str, MCPServerConfig] = {
            "srv": MCPServerConfig(transport=MCPTransport.STDIO, command="npx", enabled=True),
        }
        self.a2a = [A2AServerConfig(id="a2a-1", base_url="http://remote:9000", enabled=True)]
        self.flag = True
    def config(self) -> AppConfig:
        return AppConfig.model_construct(
            llm_config=MagicMock(), agent_config=MagicMock(),
            mcp_config=MCPConfig(mcpServers=dict(self.mcp)),
            a2a_config=A2AConfig(a2a_servers=list(self.a2a)),
        )


def _make(world=None, *, prober=None, clock=None, rng=lambda: 0.5, liveness=None):
    world = world or World()
    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(return_value=[])
    svc = ExtensionProbeService(
        config_provider=world.config,
        skill_repository=skill_repo,
        prober=prober or FakeProber(),
        probe_flag_provider=lambda: world.flag,
        liveness_view=liveness,
        clock=clock or FakeClock(),
        rng=rng,
    )
    return world, svc


async def _wait_until(predicate, timeout=1.0):
    """确定性同步（R8#1）：gated 场景禁止裸 sleep 假定并发推进——
    所有'等待探测已开探/已占满'的前置条件一律用本 helper bounded wait。"""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("条件在时限内未达成（测试前置失败，非被测行为失败）")
        await asyncio.sleep(0.005)


# —— 状态转移表逐行（spec §3.3）——

@pytest.mark.anyio
async def test_success_writes_reachable_and_next_probe_60s():
    clock = FakeClock()
    _, svc = _make(clock=clock)
    record = await svc.probe_one_manual("mcp", "srv")
    assert record.state == "reachable"
    assert record.consecutive_failures == 0
    assert record.latency_ms == 42
    assert record.tool_count == 3
    assert record.last_checked_at == T0                       # R1#7：成功必更新
    assert record.next_probe_at == T0 + timedelta(seconds=60)


@pytest.mark.anyio
async def test_first_failure_immediately_unreachable():
    prober = FakeProber()
    prober.outcome = ProbeOutcome(ok=False, latency_ms=10,
                                  error_code="connect_failed", error_message="refused")
    clock = FakeClock()
    _, svc = _make(prober=prober, clock=clock)
    record = await svc.probe_one_manual("mcp", "srv")
    assert record.state == "unreachable"       # 任何一次失败立即诚实标 unreachable
    assert record.consecutive_failures == 1
    assert record.error_code == "connect_failed"
    assert record.last_checked_at == T0
    assert record.next_probe_at == T0 + timedelta(seconds=5)   # 首败 5s、jitter 中性


@pytest.mark.anyio
async def test_backoff_sequence_5_10_20_40_80_160_300_capped():
    prober = FakeProber()
    prober.outcome = ProbeOutcome(ok=False, latency_ms=1,
                                  error_code="connect_failed", error_message="x")
    clock = FakeClock()
    _, svc = _make(prober=prober, clock=clock)
    await svc.probe_one_manual("mcp", "srv")           # 失败 1（起点）
    delays = []
    for _ in range(7):
        rec = svc.snapshot()[("mcp", "srv")]
        delta = (rec.next_probe_at - clock.now).total_seconds()
        delays.append(delta)
        clock.advance(delta)
        await svc.tick_once()                          # 后台连败累积（手动会清零 failures）
    assert delays == [5, 10, 20, 40, 80, 160, 300]     # 2**6*5=320 → cap 300


@pytest.mark.anyio
async def test_jitter_bounds():
    prober = FakeProber()
    prober.outcome = ProbeOutcome(ok=False, latency_ms=1,
                                  error_code="connect_failed", error_message="x")
    clock = FakeClock()
    _, svc_lo = _make(prober=prober, clock=clock, rng=lambda: 0.0)
    await svc_lo.probe_one_manual("mcp", "srv")
    rec = svc_lo.snapshot()[("mcp", "srv")]
    assert (rec.next_probe_at - T0).total_seconds() == pytest.approx(3.75)   # 5 * 0.75
    _, svc_hi = _make(prober=prober, clock=clock, rng=lambda: 0.999)
    await svc_hi.probe_one_manual("mcp", "srv")
    rec = svc_hi.snapshot()[("mcp", "srv")]
    assert (rec.next_probe_at - T0).total_seconds() == pytest.approx(6.2475) # 5 * 1.2495


@pytest.mark.anyio
async def test_auth_failed_terminal_no_next_probe():
    prober = FakeProber()
    prober.outcome = ProbeOutcome(ok=False, latency_ms=1,
                                  error_code="auth_failed", error_message="401")
    clock = FakeClock()
    _, svc = _make(prober=prober, clock=clock)
    record = await svc.probe_one_manual("mcp", "srv")
    assert record.state == "unreachable"
    assert record.next_probe_at is None                # 终态不自动重试
    clock.advance(3600)
    await svc.tick_once()
    assert len(prober.calls) == 1                      # tick 不再选中它


@pytest.mark.anyio
async def test_manual_probe_resets_auth_terminal():
    prober = FakeProber()
    prober.outcome = ProbeOutcome(ok=False, latency_ms=1,
                                  error_code="auth_failed", error_message="401")
    _, svc = _make(prober=prober)
    await svc.probe_one_manual("mcp", "srv")
    prober.outcome = ProbeOutcome(ok=True, latency_ms=5, tool_count=1)
    record = await svc.probe_one_manual("mcp", "srv")  # 手动解除终态
    assert record.state == "reachable"
    assert record.consecutive_failures == 0


# —— 二次复核四分支（R19#1）——

@pytest.mark.anyio
async def test_manual_recheck_flag_off_raises_probe_disabled():
    world, svc = _make()
    world.flag = False
    with pytest.raises(ProbeDisabledError) as exc:
        await svc.probe_one_manual("mcp", "srv")
    assert exc.value.reason == "probe_disabled"


@pytest.mark.anyio
async def test_manual_recheck_target_deleted_raises_gone():
    world, svc = _make()
    world.mcp.clear()
    with pytest.raises(ProbeGoneError):
        await svc.probe_one_manual("mcp", "srv")


@pytest.mark.anyio
async def test_manual_recheck_disabled_raises_extension_disabled():
    world, svc = _make()
    world.mcp["srv"] = MCPServerConfig(transport=MCPTransport.STDIO, command="npx", enabled=False)
    with pytest.raises(ProbeDisabledError) as exc:
        await svc.probe_one_manual("mcp", "srv")
    assert exc.value.reason == "extension_disabled"


@pytest.mark.anyio
async def test_manual_recheck_config_changed_probes_latest():
    prober = FakeProber()
    world, svc = _make(prober=prober)
    world.mcp["srv"] = MCPServerConfig(transport=MCPTransport.STDIO, command="node", enabled=True)
    await svc.probe_one_manual("mcp", "srv")
    assert prober.seen_configs[-1].command == "node"   # 以最新 config 执行


# —— midflight 回写复核（R18#3 / generation fencing）——

@pytest.mark.anyio
async def test_writeback_dropped_when_generation_bumped_midflight():
    prober = FakeProber()
    prober.gate = asyncio.Event()
    world, svc = _make(prober=prober)
    task = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))
    await _wait_until(lambda: len(prober.calls) == 1)  # 确定性等到已开探（R8#1）
    svc.invalidate("mcp", "srv", "update")             # generation+1 + 重置 unknown
    prober.gate.set()
    record = await task
    assert record.state == "unknown"                   # 旧结果被丢弃
    assert svc.snapshot()[("mcp", "srv")].state == "unknown"


@pytest.mark.anyio
async def test_writeback_dropped_and_evicted_when_deleted_midflight():
    prober = FakeProber()
    prober.gate = asyncio.Event()
    world, svc = _make(prober=prober)
    task = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))
    await _wait_until(lambda: len(prober.calls) == 1)  # 确定性等到已开探（R8#1）
    world.mcp.clear()                                  # legacy delete 不经 façade
    prober.gate.set()
    with pytest.raises(ProbeGoneError):
        await task
    assert ("mcp", "srv") not in svc.snapshot()


# —— 并发防护（R1#1 预算内化 + 锁语义）——

@pytest.mark.anyio
async def test_budget_exhausted_waiting_raises_probe_busy(monkeypatch):
    import app.application.services.extension_probe_service as mod
    monkeypatch.setattr(mod, "MANUAL_PROBE_BUDGET_SECONDS", 0.05)
    prober = FakeProber()
    prober.gate = asyncio.Event()
    _, svc = _make(prober=prober)
    holder = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))   # 持锁挂起
    await _wait_until(lambda: len(prober.calls) == 1)  # 确定性等到持锁已开探（R8#1）
    with pytest.raises(ProbeBusyError):
        await svc.probe_one_manual("mcp", "srv")       # 等锁超预算 → busy
    prober.gate.set()
    record = await holder
    assert record.state == "reachable"                 # 先行探测不受影响正常回写
    assert len(prober.calls) == 1                      # 排队者从未开探


@pytest.mark.anyio
async def test_budget_exhausted_waiting_for_slot_raises_probe_busy(monkeypatch):
    """R7#1：不同 key 等【全局 slot】超预算 → ProbeBusyError（区别于同 key 等锁超时）——
    钉死 _acquire_slot_wait 的 Condition.wait_for 必须带剩余预算 timeout。"""
    import app.application.services.extension_probe_service as mod
    monkeypatch.setattr(mod, "MANUAL_PROBE_BUDGET_SECONDS", 0.05)
    prober = FakeProber()
    prober.gate = asyncio.Event()
    world, svc = _make(prober=prober)
    for name in ("srv2", "srv3", "srv4"):
        world.mcp[name] = MCPServerConfig(transport=MCPTransport.STDIO, command="npx", enabled=True)
    running = [
        asyncio.create_task(svc.probe_one_manual("mcp", name))
        for name in ("srv", "srv2", "srv3")
    ]
    await _wait_until(lambda: len(prober.calls) == 3)  # R8#1：确定性等到三探测全部开探（_inflight=3）
    calls_before = len(prober.calls)
    with pytest.raises(ProbeBusyError):
        await svc.probe_one_manual("mcp", "srv4")      # 锁空闲、slot 满 → 等 slot 超预算
    assert len(prober.calls) == calls_before           # srv4 从未开探
    prober.gate.set()
    await asyncio.gather(*running)                     # 先行三探测正常收尾


@pytest.mark.anyio
async def test_started_probe_never_cancelled_and_lock_serializes():
    """R1#1 核心：开探后无取消源；同 key 第二请求只能等锁，绝不并发进入。"""
    prober = FakeProber()
    prober.gate = asyncio.Event()
    _, svc = _make(prober=prober)
    t1 = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))
    await _wait_until(lambda: len(prober.calls) == 1)  # 确定性等到 t1 已开探（R8#1）
    t2 = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))
    await asyncio.sleep(0.01)
    assert len(prober.calls) == 1                      # t2 在锁上等待，未开探
    prober.gate.set()
    r1, r2 = await asyncio.gather(t1, t2)
    assert len(prober.calls) == 2                      # 严格串行两次
    assert r1.state == "reachable" and r2.state == "reachable"


@pytest.mark.anyio
async def test_background_tick_skips_locked_key():
    prober = FakeProber()
    prober.gate = asyncio.Event()
    _, svc = _make(prober=prober)
    task = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))
    await _wait_until(lambda: len(prober.calls) == 1)  # 确定性等到持锁已开探（R8#1）
    await svc.tick_once()                              # 见锁被占 → 跳过
    assert len(prober.calls) == 1
    prober.gate.set()
    await task


@pytest.mark.anyio
async def test_background_tick_skips_when_inflight_full():
    _, svc = _make()
    svc._inflight = 3                                  # 满额（PROBE_SEMAPHORE_SIZE）
    await svc.tick_once()
    assert svc._prober.calls == []                     # 全跳，后台不排队（R18#4）


@pytest.mark.anyio
async def test_tick_skips_while_manual_probes_saturate_slots():
    """R6#3：3 个 gated 手动探测占满额度、第 4 个 manual 在排队时，
    tick_once 立即返回（不排队、不新增探测）——nowait helper 的结构保证。"""
    prober = FakeProber()
    prober.gate = asyncio.Event()
    world, svc = _make(prober=prober)
    for name in ("srv2", "srv3", "srv4"):
        world.mcp[name] = MCPServerConfig(transport=MCPTransport.STDIO, command="npx", enabled=True)
    running = [
        asyncio.create_task(svc.probe_one_manual("mcp", name))
        for name in ("srv", "srv2", "srv3")
    ]
    await _wait_until(lambda: len(prober.calls) == 3)   # R8#1：确定性等到占满 slot
    waiter = asyncio.create_task(svc.probe_one_manual("mcp", "srv4"))
    await asyncio.sleep(0.01)                           # 机会窗：给错误实现"waiter 直接开探"暴露机会
    assert len(prober.calls) == 3                       # R9#3：负断言——waiter 必须仍在等待、未开探
    await asyncio.wait_for(svc.tick_once(), timeout=1)  # 必须立即返回
    assert len(prober.calls) == 3                       # tick 零新增探测（字面 3，不吸收错误开探）
    prober.gate.set()
    await asyncio.gather(*running, waiter)              # 全部正常收尾（waiter 最终拿到 slot）


@pytest.mark.anyio
async def test_stdio_in_use_skipped_by_tick_preserves_state():
    liveness = FakeLiveness()
    liveness.active[("mcp", "srv")] = {"run-1"}
    prober = FakeProber()
    clock = FakeClock()
    _, svc = _make(prober=prober, clock=clock, liveness=liveness)
    await svc.tick_once()                              # stdio + in_use → 跳过（R4#1）
    assert prober.calls == []
    rec = svc.snapshot()[("mcp", "srv")]
    assert rec.state == "unknown"                      # 保持原态（含 unknown）
    assert rec.last_checked_at is None                 # 不动
    assert rec.next_probe_at == T0 + timedelta(seconds=60)
    record = await svc.probe_one_manual("mcp", "srv")  # 手动不受此限
    assert record.state == "reachable"


# —— invalidate / reconcile ——

@pytest.mark.anyio
async def test_invalidate_transitions():
    _, svc = _make()
    await svc.probe_one_manual("mcp", "srv")
    gen0 = svc.snapshot()[("mcp", "srv")].generation
    svc.invalidate("mcp", "srv", "disable")
    assert svc.snapshot()[("mcp", "srv")].state == "skipped"
    svc.invalidate("mcp", "srv", "enable")
    rec = svc.snapshot()[("mcp", "srv")]
    assert rec.state == "unknown" and rec.error_code is None
    svc.invalidate("mcp", "srv", "update")
    assert svc.snapshot()[("mcp", "srv")].generation > gen0
    svc.invalidate("mcp", "srv", "delete")
    assert ("mcp", "srv") not in svc.snapshot()


@pytest.mark.anyio
async def test_reconcile_fingerprint_change_resets_unknown():
    world, svc = _make()
    await svc.probe_one_manual("mcp", "srv")
    live_keys = {("mcp", "srv"), ("a2a", "a2a-1")}
    evicted = svc.reconcile(live_keys, {("mcp", "srv"): "different-fingerprint"})
    assert evicted == []
    assert svc.snapshot()[("mcp", "srv")].state == "unknown"


@pytest.mark.anyio
async def test_reconcile_missing_config_evicts_and_returns_key():
    world, svc = _make()
    await svc.probe_one_manual("mcp", "srv")
    evicted = svc.reconcile({("a2a", "a2a-1")}, {})
    assert evicted == [("mcp", "srv")]
    assert ("mcp", "srv") not in svc.snapshot()


# —— 结构性断言 ——

@pytest.mark.anyio
async def test_probe_one_manual_rejects_skill_kind():
    """R9#1/R10#1：skill 无网络探测语义——服务层结构性拒绝（endpoint 走短路重扫分支）。"""
    _, svc = _make()
    with pytest.raises(ValueError):
        await svc.probe_one_manual("skill", "any-skill")


def test_record_has_no_stored_stale_field():
    assert not hasattr(ProbeRecord(), "stale")         # stale 由消费方动态计算（R11#2）


def _called_names(func) -> set[str]:
    """收集函数体内全部 Call 目标名（Attribute.attr 与 Name.id）——AST 级，
    不受注释/docstring/字符串字面量干扰（R3#9）。"""
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def test_probe_flow_has_no_task_spawn_or_shield():
    """R1#1：probe_one_manual 内禁 create_task/shield 调用——探测必须留在锁的同协程帧内
    （wait_for 合法：仅用于锁/semaphore 等待段）。"""
    import app.application.services.extension_probe_service as mod
    called = _called_names(mod.ExtensionProbeService.probe_one_manual)
    assert "create_task" not in called
    assert "shield" not in called


def test_locked_probe_section_has_no_cancellation_wrappers():
    """R2#1：锁内段 _probe_locked 全程零取消包装调用——开探后无任何取消源（F4/INV-B9-5 地基）。"""
    import app.application.services.extension_probe_service as mod
    called = _called_names(mod.ExtensionProbeService._probe_locked)
    for banned in ("wait_for", "timeout", "shield", "create_task"):
        assert banned not in called, banned


@pytest.mark.anyio
async def test_budget_does_not_cancel_started_probe(monkeypatch):
    """R2#1 行为面：预算耗尽时探测已开始 → 不取消、正常完成回写。"""
    import app.application.services.extension_probe_service as mod
    monkeypatch.setattr(mod, "MANUAL_PROBE_BUDGET_SECONDS", 0.05)
    prober = FakeProber()
    prober.gate = asyncio.Event()
    _, svc = _make(prober=prober)
    task = asyncio.create_task(svc.probe_one_manual("mcp", "srv"))   # 锁空闲，预算内即开探
    await asyncio.sleep(0.15)                                        # 真实时间已超预算
    prober.gate.set()
    record = await task                                              # 不抛 ProbeBusyError
    assert record.state == "reachable"
    assert svc.snapshot()[("mcp", "srv")].state == "reachable"       # 回写完成


# —— Fix 2：run_loop cancel 路径不吃满 30s tick sleep ——


class _GatedTickProber:
    """probe_mcp 挂在 Event 上——让 cancel 落在 tick_once 的 mid-await 处。"""

    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate
        self.calls: list[tuple[str, str]] = []

    async def probe_mcp(self, server_name, config):
        self.calls.append(("mcp", server_name))
        await self._gate.wait()   # 永挂——cancel 必须在此帧落地
        return ProbeOutcome(ok=True, latency_ms=1, tool_count=0)

    async def probe_a2a(self, config):  # pragma: no cover - 本用例不触发
        await self._gate.wait()
        return ProbeOutcome(ok=True, latency_ms=1)


@pytest.mark.anyio
async def test_run_loop_cancel_skips_tick_interval_sleep():
    """Fix 2：sleep 移出 finally 后，tick_once mid-await 收到 cancel 时循环立即退出，
    绝不再跑一个完整 PROBE_TICK_INTERVAL_SECONDS(30s) sleep。**真 30s 常量不 monkeypatch**——
    task 必须在 _wait_until 的 1s 上限内结束，直接证明 30s sleep 没在 cancel 路径上执行。
    """
    gate = asyncio.Event()
    prober = _GatedTickProber(gate)
    world = World()
    clock = FakeClock()
    _, svc = _make(prober=prober, clock=clock, liveness=FakeLiveness())
    # 播种一条已到期 mcp 记录，让首个 tick 立即选中开探并挂在 gate 上。
    svc._records[("mcp", "srv")] = ProbeRecord(
        state="unknown", next_probe_at=T0 - timedelta(seconds=1)
    )

    task = asyncio.create_task(svc.run_loop())
    await _wait_until(lambda: len(prober.calls) == 1)   # 探测已开、挂在 gate 上（mid-await）

    task.cancel()                                       # cancel 落在 tick_once 的 await 上
    # 关键断言：1s 内结束——若 sleep 仍在 finally，cancel 会先被 tick 吞掉再进 30s sleep。
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
