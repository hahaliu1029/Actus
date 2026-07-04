"""B9 循环自愈 + shutdown + fail-open（spec §3.5/§13，Task 14）。

覆盖 Task 14 三块：
- ``ExtensionProbeService.run_loop`` 顶层 while-True + try/except/finally 自愈
  （单轮异常 warn + sleep 后继续，绝不被单 tick 异常杀死）与 ``shutdown``。
- ``main._start_b9_probe`` / ``_stop_b9_probe`` / ``_log_b9_probe_exit`` lifespan helper：
  无条件构造/起 task；启动失败 fail-open（只 warn，不冒泡）；shutdown 沿同 task
  cancel 传播、cleanup 闭环。
- ``tick_once`` 内 skill diagnostics 后台刷新（flag off 时零扫描；仅状态变化打日志）。
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.application.services.extension_probe_service as probe_mod
from app.application.services.extension_probe_service import (
    ExtensionProbeService,
    ProbeOutcome,
)
from app.domain.models.app_config import (
    A2AConfig, A2AServerConfig, AppConfig, MCPConfig, MCPServerConfig, MCPTransport,
)
from app.domain.models.runtime_extension import LivenessSnapshot
from app.domain.models.skill_diagnostic import SkillDiagnostic

T0 = datetime(2026, 7, 5, 12, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeProber:
    def __init__(self) -> None:
        self.outcome = ProbeOutcome(ok=True, latency_ms=7, tool_count=1)
        self.calls: list[tuple[str, str]] = []
        self.finally_flag = False   # cancel-cleanup 闭环观测点（用例 3）
        self.gate: asyncio.Event | None = None

    async def probe_mcp(self, server_name, config):
        self.calls.append(("mcp", server_name))
        try:
            if self.gate:
                await self.gate.wait()
            return self.outcome
        finally:
            self.finally_flag = True

    async def probe_a2a(self, config):
        self.calls.append(("a2a", config.id))
        try:
            if self.gate:
                await self.gate.wait()
            return self.outcome
        finally:
            self.finally_flag = True


class FakeLiveness:
    def snapshot(self) -> LivenessSnapshot:
        return LivenessSnapshot(active={}, degraded=False)


class World:
    def __init__(self) -> None:
        self.mcp: dict[str, MCPServerConfig] = {
            "srv": MCPServerConfig(
                transport=MCPTransport.STDIO, command="npx", enabled=True
            ),
        }
        self.a2a: list[A2AServerConfig] = []
        self.flag = True

    def config(self) -> AppConfig:
        return AppConfig.model_construct(
            llm_config=MagicMock(), agent_config=MagicMock(),
            mcp_config=MCPConfig(mcpServers=dict(self.mcp)),
            a2a_config=A2AConfig(a2a_servers=list(self.a2a)),
        )


def _make(world=None, *, prober=None, clock=None, skill_repo=None, liveness=None):
    world = world or World()
    if skill_repo is None:
        skill_repo = MagicMock()
        skill_repo.list_with_diagnostics = AsyncMock(return_value=[])
    svc = ExtensionProbeService(
        config_provider=world.config,
        skill_repository=skill_repo,
        prober=prober or FakeProber(),
        probe_flag_provider=lambda: world.flag,
        liveness_view=liveness,
        clock=clock or FakeClock(),
        rng=lambda: 0.5,
    )
    return world, svc


async def _wait_until(predicate, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("条件在时限内未达成（测试前置失败）")
        await asyncio.sleep(0.005)


# ============================================================
# 1. run_loop 单轮异常自愈——tick_once 首败二成，循环不退出
# ============================================================
@pytest.mark.anyio
async def test_run_loop_survives_tick_exception(monkeypatch):
    monkeypatch.setattr(probe_mod, "PROBE_TICK_INTERVAL_SECONDS", 0.01)
    _, svc = _make()

    ticks = {"n": 0}

    async def flaky_tick():
        ticks["n"] += 1
        if ticks["n"] == 1:
            raise RuntimeError("boom")   # 首轮抛错

    svc.tick_once = flaky_tick   # type: ignore[assignment]

    task = asyncio.create_task(svc.run_loop())
    await _wait_until(lambda: ticks["n"] >= 2)     # 两轮 tick 都发生
    assert not task.done()                          # 循环未被单 tick 异常杀死

    await svc.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ============================================================
# 2. flag off → 空转：prober 与 skill 扫描都不跑（R9#4 一体门控）
# ============================================================
@pytest.mark.anyio
async def test_run_loop_flag_off_idles():
    world, svc = _make()
    world.flag = False

    await svc.tick_once()

    assert svc._prober.calls == []
    svc._skill_repository.list_with_diagnostics.assert_not_awaited()


# ============================================================
# 3. shutdown cancel in-flight → prober finally 闭环（cancel 沿同 task 传播，F4）
# ============================================================
@pytest.mark.anyio
async def test_shutdown_cancels_inflight_cleanup_runs(monkeypatch):
    monkeypatch.setattr(probe_mod, "PROBE_TICK_INTERVAL_SECONDS", 0.01)
    prober = FakeProber()
    prober.gate = asyncio.Event()          # 探测挂起
    clock = FakeClock()
    world, svc = _make(prober=prober, clock=clock, liveness=FakeLiveness())

    # 播种一条已到期记录，让 tick_once 立即选中开探
    svc._records[("mcp", "srv")] = probe_mod.ProbeRecord(
        state="unknown", next_probe_at=T0 - timedelta(seconds=1)
    )

    task = asyncio.create_task(svc.run_loop())
    await _wait_until(lambda: len(prober.calls) == 1)   # 探测已开、挂在 gate 上

    task.cancel()                                       # cancel 沿同 task 传播进探测
    with pytest.raises(asyncio.CancelledError):
        await task
    assert prober.finally_flag is True                  # cleanup 闭环


# ============================================================
# 4. _start_b9_probe 构造失败 → fail-open（不冒泡 + started False + service None）
# ============================================================
def test_start_b9_probe_failure_is_fail_open(monkeypatch):
    import app.main as main_mod

    def _boom(*a, **k):
        raise RuntimeError("构造炸了")

    monkeypatch.setattr(
        probe_mod, "ExtensionProbeService", _boom, raising=True
    )

    fake_app = SimpleNamespace(state=SimpleNamespace())
    # 不抛
    main_mod._start_b9_probe(fake_app)

    assert fake_app.state.extension_probe_started is False
    assert fake_app.state.extension_probe_service is None


# ============================================================
# 5. flag off 仍无条件构造/起 task（R1#3 终态：flag 不参与构造条件）
# ============================================================
@pytest.mark.anyio
async def test_start_b9_probe_flag_off_still_constructs(monkeypatch):
    import app.main as main_mod

    # 让 flag off（不参与构造），但确保 run_loop 不真跑成无限循环卡住测试
    monkeypatch.setattr(probe_mod, "PROBE_TICK_INTERVAL_SECONDS", 0.01)

    from app.interfaces import service_dependencies as deps

    def _flag_off_config():
        return AppConfig.model_construct(
            llm_config=MagicMock(),
            agent_config=MagicMock(),
            mcp_config=MCPConfig(mcpServers={}),
            a2a_config=A2AConfig(a2a_servers=[]),
            tool_runtime=SimpleNamespace(
                extension_probe_enabled=False, extension_stats_enabled=False
            ),
        )

    monkeypatch.setattr(deps, "_load_app_config", _flag_off_config, raising=True)

    fake_app = SimpleNamespace(state=SimpleNamespace())
    main_mod._start_b9_probe(fake_app)

    try:
        assert fake_app.state.extension_probe_service is not None
        assert fake_app.state.extension_probe_started is True
        task = fake_app.state._extension_probe_task
        assert isinstance(task, asyncio.Task)
    finally:
        await main_mod._stop_b9_probe(fake_app)   # cancel 收尾


# ============================================================
# 5b. _stop_b9_probe cancel + await 收尾（shutdown 顺序基元）
# ============================================================
@pytest.mark.anyio
async def test_stop_b9_probe_cancels_task(monkeypatch):
    import app.main as main_mod

    async def _forever():
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(_forever())
    fake_app = SimpleNamespace(
        state=SimpleNamespace(_extension_probe_task=task)
    )
    await main_mod._stop_b9_probe(fake_app)
    assert task.cancelled() or task.done()

    # 无 task（构造失败路径）也不炸
    empty_app = SimpleNamespace(state=SimpleNamespace())
    await main_mod._stop_b9_probe(empty_app)


# ============================================================
# 5d. Fix 1：cancel 被底层吞掉时 _stop_b9_probe 仍 bounded 退出（不挂 shutdown）
# ============================================================
class _FakeStoppableService:
    """探测 service 假体：shutdown() 仅记录被调（对齐真 service 置 _stopping 语义）。"""

    def __init__(self) -> None:
        self.shutdown_called = False

    async def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.mark.anyio
async def test_stop_b9_probe_bounded_when_cancel_swallowed(monkeypatch, caplog):
    """底层 MCP init 的 except BaseException 可吞 CancelledError → task 永不退出。
    _stop_b9_probe 必须：(1) 调 service.shutdown()（cooperative stop）；
    (2) bounded asyncio.wait 超时后 fail-open 放弃并 warn，绝不无限挂起
    （D6：不用 wait_for——它 cancel 后等确认，对吞 cancel 任务同样会挂）。
    """
    import app.main as main_mod

    # 缩小等待上限，避免真等 5s（不用 pytest-timeout / 不真 sleep 30s）。
    monkeypatch.setattr(main_mod, "PROBE_SHUTDOWN_WAIT_SECONDS", 0.05)

    # 模拟底层 MCP init 的 except BaseException 反复吞掉 cancel：即便 lifespan
    # .cancel() 落地，也被吃回去、循环不退出——正是 Fix 1 要防的病灶。task 只在测试
    # 显式 set(release) 后才退出，模拟一个"完全不响应 cancel 的 in-flight 探测"。
    release = asyncio.Event()

    async def _swallows_cancel():
        while not release.is_set():
            try:
                await asyncio.wait_for(release.wait(), timeout=0.02)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # 吞掉外层 cancel（BaseException-swallow 病灶），继续等 release

    task = asyncio.create_task(_swallows_cancel())
    await asyncio.sleep(0)  # 让 task 起步
    service = _FakeStoppableService()
    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            extension_probe_service=service,
            _extension_probe_task=task,
        )
    )

    with caplog.at_level(logging.WARNING):
        # 关键断言：不挂起——bounded 等待到点即返回（wait_for 上限 2s 兜底捕挂起）。
        await asyncio.wait_for(main_mod._stop_b9_probe(fake_app), timeout=2.0)

    assert service.shutdown_called is True                  # cooperative stop 已触发
    assert any("5s 内退出" in r.message for r in caplog.records)  # fail-open warn

    # 收尾：释放 task 让它干净退出（避免 pending task 泄漏警告）。
    release.set()
    await asyncio.wait_for(task, timeout=1.0)


# ============================================================
# 5c. _log_b9_probe_exit：cancelled 静默；异常记 error
# ============================================================
@pytest.mark.anyio
async def test_log_b9_probe_exit_logs_unexpected(caplog):
    import app.main as main_mod

    async def _raise():
        raise ValueError("非预期退出")

    task = asyncio.create_task(_raise())
    with pytest.raises(ValueError):
        await task
    with caplog.at_level(logging.ERROR):
        main_mod._log_b9_probe_exit(task)
    assert any("非预期退出" in r.message or "B9" in r.message for r in caplog.records)

    # cancelled → 静默（无新 error）
    async def _forever():
        while True:
            await asyncio.sleep(0.01)

    ctask = asyncio.create_task(_forever())
    ctask.cancel()
    try:
        await ctask
    except asyncio.CancelledError:
        pass
    caplog.clear()
    main_mod._log_b9_probe_exit(ctask)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


# ============================================================
# 6. skill diagnostics 刷新：仅状态变化打日志；不入 probe 快照（R6#4）
# ============================================================
@pytest.mark.anyio
async def test_skill_diag_refresh_logs_only_on_change(caplog):
    world = World()
    world.mcp.clear()   # 无 mcp/a2a，隔离 skill 扫描的日志

    good = SkillDiagnostic(skill_key="s1", ok=True)
    broken = SkillDiagnostic(
        skill_key="s1", ok=False, error_code="parse_error",
        relative_file="meta.json",
    )

    skill_repo = MagicMock()
    seq = [
        [good],      # 轮 1：建基线，无变化日志
        [broken],    # 轮 2：新损坏 → warn 一次
        [broken],    # 轮 3：相同 → 零新日志
        [good],      # 轮 4：修复 → info 一次
    ]
    skill_repo.list_with_diagnostics = AsyncMock(side_effect=seq)
    _, svc = _make(world=world, skill_repo=skill_repo)

    with caplog.at_level(logging.INFO, logger="app.application.services.extension_probe_service"):
        # 轮 1：建基线，无变化日志
        caplog.clear()
        await svc.tick_once()
        assert caplog.records == []
        # 轮 2：新损坏 → warn
        caplog.clear()
        await svc.tick_once()
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        assert "s1" in warns[0].message
        # 轮 3：相同 → 零新日志
        caplog.clear()
        await svc.tick_once()
        assert caplog.records == []
        # 轮 4：修复 → info
        caplog.clear()
        await svc.tick_once()
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        assert "s1" in infos[0].message

    # 全程不进 probe 快照（snapshot 无 skill key）
    assert all(k[0] != "skill" for k in svc.snapshot().keys())


# ============================================================
# 7. Fix 4：skill integrity 刷新 1s 防御超时——挂死扫描不拖垮整 tick
# ============================================================
@pytest.mark.anyio
async def test_skill_diag_refresh_defensive_timeout(monkeypatch, caplog):
    """list_with_diagnostics 挂死（await 永不 set 的 Event）时，_refresh_skill_diagnostics
    必须在 SKILL_DIAG_REFRESH_TIMEOUT_SECONDS 后放弃 + warn；mcp/a2a 探测不受影响。
    """
    monkeypatch.setattr(probe_mod, "SKILL_DIAG_REFRESH_TIMEOUT_SECONDS", 0.01)

    world = World()
    world.mcp.clear()  # 隔离：只测 skill 刷新超时，不牵扯 mcp/a2a 探测

    hang = asyncio.Event()  # 永不 set → 扫描挂死

    async def _hang():
        await hang.wait()
        return []

    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(side_effect=_hang)
    _, svc = _make(world=world, skill_repo=skill_repo)

    with caplog.at_level(logging.WARNING, logger="app.application.services.extension_probe_service"):
        # tick_once 必须整体在时限内返回（不因扫描挂死无限阻塞）。
        await asyncio.wait_for(svc.tick_once(), timeout=1.0)

    assert any("超时" in r.message for r in caplog.records)
    # 基线未更新（下轮重试）；skill 未进 probe 快照。
    assert svc._last_skill_diag_state is None
    assert all(k[0] != "skill" for k in svc.snapshot().keys())


@pytest.mark.anyio
async def test_skill_diag_timeout_does_not_block_mcp_probe(monkeypatch):
    """Fix 4 隔离性：skill 扫描超时后，同一 tick 的 mcp 探测照常执行（不被拖死）。"""
    monkeypatch.setattr(probe_mod, "SKILL_DIAG_REFRESH_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(probe_mod, "PROBE_TICK_INTERVAL_SECONDS", 0.01)

    world = World()  # 保留默认 mcp "srv"（enabled stdio）
    clock = FakeClock()

    hang = asyncio.Event()

    async def _hang():
        await hang.wait()
        return []

    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(side_effect=_hang)
    prober = FakeProber()
    _, svc = _make(
        world=world, prober=prober, clock=clock, skill_repo=skill_repo,
        liveness=FakeLiveness(),
    )
    # 播种一条已到期 mcp 记录，让 tick 在 skill 扫描超时后仍选中它开探。
    svc._records[("mcp", "srv")] = probe_mod.ProbeRecord(
        state="unknown", next_probe_at=T0 - timedelta(seconds=1)
    )

    await asyncio.wait_for(svc.tick_once(), timeout=1.0)
    assert prober.calls == [("mcp", "srv")]  # skill 挂死不挡 mcp 探测


# ============================================================
# 8. Fix 5：skill_key 过脱敏——含换行的坏 skill_key 不以原文进日志
# ============================================================
@pytest.mark.anyio
async def test_skill_diag_change_log_sanitizes_skill_key(caplog):
    """skill_key 攻击者可控（目录名）——变化日志里的 %s 必须过 _safe_log_id，
    含换行/控制字符的 key 不以原文出现在任何 record.message 中。
    """
    world = World()
    world.mcp.clear()  # 隔离 skill 变化日志

    bad_key = "bad\nID"
    good = SkillDiagnostic(skill_key=bad_key, ok=True)
    broken = SkillDiagnostic(
        skill_key=bad_key, ok=False, error_code="parse_error",
        relative_file="meta.json",
    )
    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(side_effect=[[good], [broken]])
    _, svc = _make(world=world, skill_repo=skill_repo)

    with caplog.at_level(logging.INFO, logger="app.application.services.extension_probe_service"):
        await svc.tick_once()   # 轮 1：建基线，无变化日志
        caplog.clear()
        await svc.tick_once()   # 轮 2：新损坏 → warn（走脱敏）

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    # 原文换行不得出现；脱敏后 "badID" 出现（换行被剥离）。
    assert bad_key not in warns[0].message
    assert "\n" not in warns[0].message
    assert "badID" in warns[0].message
