"""B9 GET 聚合：PR-1 stub 语义 + 角色投影 + 四路降级（spec §6/§12/§13）
+ PR-2 真实 probe/liveness 接入 + 惰性 reconcile + INV-B9-4 零网络门（Task 17）。"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.extension_probe_service import ProbeRecord
from app.application.services.runtime_extension_service import RuntimeExtensionService
from app.domain.external.extension_stats import ExtensionStatsData
from app.domain.models.app_config import (
    A2AConfig, A2AServerConfig, AppConfig, MCPConfig, MCPServerConfig, MCPTransport,
)
from app.domain.models.runtime_extension import LivenessSnapshot
from app.domain.models.skill_diagnostic import SkillDiagnostic


def _app_config() -> AppConfig:
    """最小 AppConfig：2 mcp（1 disabled）+ 1 a2a + skill 走 repo。

    llm_config/agent_config 必填——用 MagicMock 绕开或用 AppConfig.model_construct
    跳过校验（推荐 model_construct，纯读场景安全）。
    """
    return AppConfig.model_construct(
        llm_config=MagicMock(), agent_config=MagicMock(),
        mcp_config=MCPConfig(mcpServers={
            "srv-on": MCPServerConfig(transport=MCPTransport.STDIO, command="npx", enabled=True),
            "srv-off": MCPServerConfig(transport=MCPTransport.STDIO, command="npx", enabled=False),
        }),
        a2a_config=A2AConfig(a2a_servers=[
            A2AServerConfig(id="a2a-1111-2222", base_url="http://remote:9000", enabled=True),
        ]),
    )


def _skill_diags() -> list[SkillDiagnostic]:
    good = MagicMock()
    good.id = "skill-good"; good.name = "Good Skill"; good.description = "d"
    good.enabled = True; good.runtime_type = MagicMock(value="native")
    good.source_type = MagicMock(value="local")
    good.manifest = {"bundle_file_count": 2}
    return [
        SkillDiagnostic(skill_key="skill-good", ok=True, skill=good),
        SkillDiagnostic(skill_key="broken-dir", ok=False,
                        error_code="parse_error", relative_file="meta.json"),
    ]


def _service(*, enablements=None, diags=None) -> RuntimeExtensionService:
    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(return_value=diags if diags is not None else _skill_diags())
    enablement = MagicMock()
    enablement.list_user_enablements = AsyncMock(return_value=enablements or [])
    return RuntimeExtensionService(
        config_provider=_app_config,
        skill_repository=skill_repo,
        enablement_service=enablement,
    )


@pytest.mark.anyio
async def test_pr1_stub_full_assertions_admin():
    snap = await _service().get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    # 配置可读条目 health unknown / disabled skipped
    assert items[("mcp", "srv-on")].health.state == "unknown"
    assert items[("mcp", "srv-off")].health.state == "skipped"
    # 坏 skill 自 PR-1 即 integrity error + config_unreadable
    broken = items[("skill", "broken-dir")]
    assert (broken.health.kind, broken.health.state) == ("integrity", "error")
    assert broken.health.error_code == "parse_error"
    assert broken.health.relative_file == "meta.json"
    assert broken.config.reason_code == "config_unreadable"
    assert broken.details == {"runtime_type": "unknown", "source_type": "unknown", "bundle_file_count": None}
    # liveness stub：mcp/a2a unknown、skill not_applicable
    assert items[("mcp", "srv-on")].liveness.state == "unknown"
    assert items[("a2a", "a2a-1111-2222")].liveness.state == "unknown"
    assert items[("skill", "skill-good")].liveness.state == "not_applicable"
    # stats reason：A2A unsupported、其余 disabled（Admin）
    assert items[("a2a", "a2a-1111-2222")].stats.unavailable_reason == "unsupported"
    assert items[("mcp", "srv-on")].stats.unavailable_reason == "disabled"
    # 顶层双 flag false
    assert snap.probe_enabled is False and snap.stats_enabled is False
    # A2A 身份：无 agent_card → 前缀名 + description=None（禁 base_url fallback）
    a2a = items[("a2a", "a2a-1111-2222")]
    assert a2a.name == "A2A a2a-1111"
    assert a2a.description is None
    assert "http://remote:9000" not in str(a2a.description)


@pytest.mark.anyio
async def test_non_admin_minimal_projection():
    snap = await _service().get_extensions(user_id="u1", is_admin=False)
    items = {(i.kind, i.id): i for i in snap.items}
    broken = items[("skill", "broken-dir")]
    assert broken.health.error_code is None and broken.health.relative_file is None
    assert broken.details == {"runtime_type": "unknown"}
    for it in snap.items:
        assert it.stats.available is False and it.stats.unavailable_reason == "admin_only"
        assert it.health.latency_ms is None and it.health.error_message is None
        assert it.health.next_probe_at is None
        if it.kind in ("mcp", "a2a"):
            assert (it.liveness.state, it.liveness.active_run_count) == ("unknown", 0)
    a2a = items[("a2a", "a2a-1111-2222")]
    assert "base_url" not in a2a.details            # 键不存在，任何形态不下发
    assert set(items[("mcp", "srv-on")].details.keys()) == {"transport"}


@pytest.mark.anyio
async def test_reason_code_six_value_matrix():
    """R11#6：六值各至少一例。"""
    from app.domain.models.user_tool_enablement import ToolType  # 实际枚举路径 grep 确认
    def enab(tool_type, tool_id, enabled):
        e = MagicMock(); e.tool_type = tool_type; e.tool_id = tool_id; e.enabled = enabled
        return e
    svc = _service(enablements=[
        enab(ToolType.MCP, "srv-on", False),        # true/false → disabled_user
        enab(ToolType.MCP, "srv-off", False),       # false/false → disabled_both
    ])
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("mcp", "srv-on")].config.reason_code == "disabled_user"
    # F10 锁定：user 级 disable 不影响共享 health——srv-on global 仍 enabled，
    # health 只看 global（probe stub → unknown，绝不因 effective_enabled 落 skipped）
    assert items[("mcp", "srv-on")].health.state == "unknown"
    assert items[("mcp", "srv-off")].config.reason_code == "disabled_both"
    assert items[("skill", "skill-good")].config.reason_code == "enabled"       # true/true（无记录默认 True）
    assert items[("skill", "broken-dir")].config.reason_code == "config_unreadable"
    # disabled_global：srv-off 无 user 记录时
    snap2 = await _service().get_extensions(user_id="u1", is_admin=True)
    items2 = {(i.kind, i.id): i for i in snap2.items}
    assert items2[("mcp", "srv-off")].config.reason_code == "disabled_global"


@pytest.mark.anyio
async def test_user_enablement_db_failure_degrades():
    """R5#2：user 级查询降级 → enabled_user=None + user_enablement_unknown + 200。"""
    svc = _service()
    svc._enablement_service.list_user_enablements = AsyncMock(side_effect=RuntimeError("db down"))
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    on = next(i for i in snap.items if i.id == "srv-on")
    assert on.config.enabled_user is None
    assert on.config.reason_code == "user_enablement_unknown"
    assert on.config.effective_enabled is True      # 按 global 回退


@pytest.mark.anyio
async def test_skill_diagnostics_failure_degrades():
    svc = _service()
    svc._skill_repository.list_with_diagnostics = AsyncMock(side_effect=RuntimeError("io"))
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    kinds = {i.kind for i in snap.items}
    assert "mcp" in kinds and "a2a" in kinds and "skill" not in kinds


@pytest.mark.anyio
async def test_snapshot_at_within_request_window():
    """R11#4：snapshot_at ISO UTC + 落在请求前后区间。"""
    before = datetime.now(timezone.utc)
    snap = await _service().get_extensions(user_id="u1", is_admin=True)
    after = datetime.now(timezone.utc)
    assert before <= snap.snapshot_at <= after


def _skill_diags_disabled_good_plus_broken() -> list[SkillDiagnostic]:
    """globally-disabled 好 skill + 坏 skill（config 也读作 disabled）。"""
    good = MagicMock()
    good.id = "skill-off"; good.name = "Disabled Skill"; good.description = "d"
    good.enabled = False; good.runtime_type = MagicMock(value="native")
    good.source_type = MagicMock(value="local")
    good.manifest = {"bundle_file_count": 1}
    return [
        SkillDiagnostic(skill_key="skill-off", ok=True, skill=good),
        SkillDiagnostic(skill_key="broken-dir", ok=False,
                        error_code="parse_error", relative_file="meta.json"),
    ]


@pytest.mark.anyio
async def test_disabled_good_skill_health_skipped_broken_stays_error():
    """Fix 1：global-disabled 好 skill → integrity/skipped；坏 skill 恒 integrity/error。"""
    svc = _service(diags=_skill_diags_disabled_good_plus_broken())
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    good = items[("skill", "skill-off")]
    assert good.health.kind == "integrity"
    assert good.health.state == "skipped"
    broken = items[("skill", "broken-dir")]
    assert broken.health.kind == "integrity"
    assert broken.health.state == "error"


@pytest.mark.anyio
async def test_stats_reason_disabled_when_flag_off_despite_reader_present():
    """Fix 2(a)：admin + mcp + reader 存在 + stats_enabled=False → disabled（非 redis_unavailable）。"""
    reader = MagicMock()
    reader.read_many = AsyncMock()
    svc = RuntimeExtensionService(
        config_provider=_app_config,
        skill_repository=_service()._skill_repository,
        enablement_service=_service()._enablement_service,
        stats_reader=reader,
        stats_enabled_provider=lambda: False,
    )
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("mcp", "srv-on")].stats.unavailable_reason == "disabled"
    reader.read_many.assert_not_awaited()  # flag off ⇒ reader 连读都不读（非读了被 broad-catch 吞）


@pytest.mark.anyio
async def test_stats_reason_redis_unavailable_when_flag_on_reader_present():
    """Fix 2(b)：admin + mcp + reader 存在 + stats_enabled=True → redis_unavailable（PR-1 无真实数据）。"""
    svc = RuntimeExtensionService(
        config_provider=_app_config,
        skill_repository=_service()._skill_repository,
        enablement_service=_service()._enablement_service,
        stats_reader=MagicMock(),
        stats_enabled_provider=lambda: True,
    )
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("mcp", "srv-on")].stats.unavailable_reason == "redis_unavailable"


# ====================================================================== #
# Task 17：GET 接入真实 probe/liveness + 惰性 reconcile + INV-B9-4 零网络门
# ====================================================================== #


class _FakeProbeView:
    """结构化满足 ProbeView：snapshot() 返回注入记录；reconcile() 记录调用并回被剔除 key。

    刻意**不含** ``probe_*`` / ``initialize`` / ``httpx`` 之类网络方法——INV-B9-4
    断言 GET 只触达 snapshot()/reconcile()。
    """

    def __init__(
        self,
        records: dict[tuple[str, str], ProbeRecord],
        *,
        evicted: list[tuple[str, str]] | None = None,
    ) -> None:
        self._records = records
        self._evicted = evicted or []
        self.snapshot_calls = 0
        self.reconcile_calls: list[
            tuple[set[tuple[str, str]], dict[tuple[str, str], str]]
        ] = []

    def snapshot(self) -> dict[tuple[str, str], ProbeRecord]:
        self.snapshot_calls += 1
        return dict(self._records)

    def reconcile(self, live_keys, fingerprints):
        self.reconcile_calls.append((set(live_keys), dict(fingerprints)))
        return list(self._evicted)


class _FakeLivenessView:
    def __init__(self, snapshot: LivenessSnapshot) -> None:
        self._snapshot = snapshot
        self.snapshot_calls = 0

    def snapshot(self) -> LivenessSnapshot:
        self.snapshot_calls += 1
        return self._snapshot


def _probe_svc(
    *,
    records=None,
    liveness=None,
    stats_reader=None,
    probe_enabled=True,
    stats_enabled=False,
    clock=None,
    evicted=None,
    enablements=None,
    diags=None,
) -> RuntimeExtensionService:
    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(
        return_value=diags if diags is not None else _skill_diags()
    )
    enablement = MagicMock()
    enablement.list_user_enablements = AsyncMock(return_value=enablements or [])
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return RuntimeExtensionService(
        config_provider=_app_config,
        skill_repository=skill_repo,
        enablement_service=enablement,
        probe_view=_FakeProbeView(records or {}, evicted=evicted),
        liveness_view=_FakeLivenessView(liveness) if liveness is not None else None,
        stats_reader=stats_reader,
        probe_enabled_provider=lambda: probe_enabled,
        stats_enabled_provider=lambda: stats_enabled,
        **kwargs,
    )


@pytest.mark.anyio
async def test_health_from_probe_record():
    """fake probe_view 带 reachable record → health 全字段透传（Admin）。"""
    now = datetime.now(timezone.utc)
    rec = ProbeRecord(
        state="reachable",
        last_checked_at=now,
        latency_ms=42,
        consecutive_failures=0,
        next_probe_at=now + timedelta(seconds=60),
        tool_count=3,
    )
    svc = _probe_svc(records={("mcp", "srv-on"): rec}, clock=lambda: now)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    h = items[("mcp", "srv-on")].health
    assert h.kind == "probe"
    assert h.state == "reachable"
    assert h.last_checked_at == now
    assert h.latency_ms == 42
    assert h.consecutive_failures == 0
    assert h.next_probe_at == now + timedelta(seconds=60)
    assert h.stale is False
    # 无记录条目仍 unknown
    assert items[("mcp", "srv-off")].health.state == "skipped"  # global disabled


@pytest.mark.anyio
async def test_stale_boundary_matrix():
    """R11#2：last_checked_at=None→False；=now-120s→False；=now-121s→True（fake clock）。"""
    now = datetime.now(timezone.utc)

    async def _stale(delta_seconds):
        last = None if delta_seconds is None else now - timedelta(seconds=delta_seconds)
        rec = ProbeRecord(state="reachable", last_checked_at=last, latency_ms=1)
        svc = _probe_svc(records={("mcp", "srv-on"): rec}, clock=lambda: now)
        snap = await svc.get_extensions(user_id="u1", is_admin=True)
        return {(i.kind, i.id): i for i in snap.items}[("mcp", "srv-on")].health.stale

    assert await _stale(None) is False        # 无时间戳
    assert await _stale(120) is False         # 恰好 120s → 未过期（> 边界）
    assert await _stale(121) is True          # 121s → 过期


@pytest.mark.anyio
async def test_flag_off_projection_clears_timestamps():
    """R12#3/R18#1：probe_enabled_provider=False + 有记录 → unknown/None/False（内部 record 不清）。"""
    now = datetime.now(timezone.utc)
    rec = ProbeRecord(
        state="reachable", last_checked_at=now - timedelta(seconds=200),
        latency_ms=42, tool_count=3,
    )
    svc = _probe_svc(
        records={("mcp", "srv-on"): rec}, clock=lambda: now, probe_enabled=False
    )
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    h = items[("mcp", "srv-on")].health
    assert h.state == "unknown"
    assert h.last_checked_at is None
    assert h.stale is False
    assert snap.probe_enabled is False
    # mcp details.tool_count 投影清空为 None
    assert items[("mcp", "srv-on")].details.get("tool_count") is None


@pytest.mark.anyio
async def test_liveness_in_use_and_idle():
    """active[(kind,id)] 非空 → (in_use, len)；空 → (idle, 0)。"""
    liveness = LivenessSnapshot(
        active={("mcp", "srv-on"): frozenset({"run-a", "run-b"})}, degraded=False
    )
    svc = _probe_svc(liveness=liveness)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    on = items[("mcp", "srv-on")].liveness
    assert (on.state, on.active_run_count) == ("in_use", 2)
    a2a = items[("a2a", "a2a-1111-2222")].liveness
    assert (a2a.state, a2a.active_run_count) == ("idle", 0)
    # skill 恒 not_applicable
    assert items[("skill", "skill-good")].liveness.state == "not_applicable"


@pytest.mark.anyio
async def test_liveness_degraded_forces_unknown():
    """R12#2：liveness_view.snapshot().degraded → mcp/a2a 全 unknown。"""
    liveness = LivenessSnapshot(
        active={("mcp", "srv-on"): frozenset({"run-a"})}, degraded=True
    )
    svc = _probe_svc(liveness=liveness)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("mcp", "srv-on")].liveness.state == "unknown"
    assert items[("a2a", "a2a-1111-2222")].liveness.state == "unknown"
    # skill 不受 degraded 影响
    assert items[("skill", "skill-good")].liveness.state == "not_applicable"


@pytest.mark.anyio
async def test_reconcile_invoked_and_evicted_keys_forwarded_to_stats_delete():
    """GET 组装调 probe_view.reconcile(live_keys, fingerprints)；被剔除 keys → stats_reader.delete_key。"""
    reader = MagicMock()
    reader.read_many = AsyncMock(return_value={})
    reader.delete_key = MagicMock()
    svc = _probe_svc(
        records={},
        stats_reader=reader,
        stats_enabled=True,
        evicted=[("mcp", "gone-srv"), ("a2a", "gone-agent")],
    )
    await svc.get_extensions(user_id="u1", is_admin=True)
    fake_probe = svc._probe_view
    assert len(fake_probe.reconcile_calls) == 1
    live_keys, fingerprints = fake_probe.reconcile_calls[0]
    # live_keys 从 config 清单构造（mcp + a2a；skill 不进 probe reconcile）
    assert ("mcp", "srv-on") in live_keys
    assert ("mcp", "srv-off") in live_keys
    assert ("a2a", "a2a-1111-2222") in live_keys
    assert all(k[0] != "skill" for k in live_keys)
    # 每个 live key 都带 fingerprint
    assert set(fingerprints.keys()) == live_keys
    # 被剔除 key 转发 stats delete
    reader.delete_key.assert_any_call("mcp", "gone-srv")
    reader.delete_key.assert_any_call("a2a", "gone-agent")
    assert reader.delete_key.call_count == 2


@pytest.mark.anyio
async def test_reconcile_evicted_ignored_when_stats_reader_none():
    """stats_reader None → 被剔除 key 静默忽略（不抛）。"""
    svc = _probe_svc(records={}, evicted=[("mcp", "gone-srv")])
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    assert snap is not None  # 无异常


@pytest.mark.anyio
async def test_reconcile_delete_key_failure_log_sanitizes_ext_id(caplog):
    """Fix 5：delete_key 抛错的 reconcile-failure 日志——ext_id 攻击者可控（config id），
    含换行的坏 ext_id 不以原文进 record.message（过 _safe_log_id）。
    """
    import logging as _logging

    bad_id = "gone\nEVIL"
    reader = MagicMock()
    reader.read_many = AsyncMock(return_value={})
    reader.delete_key = MagicMock(side_effect=RuntimeError("del 失败"))
    svc = _probe_svc(
        records={},
        stats_reader=reader,
        stats_enabled=True,
        evicted=[("mcp", bad_id)],
    )
    with caplog.at_level(
        _logging.WARNING, logger="app.application.services.runtime_extension_service"
    ):
        await svc.get_extensions(user_id="u1", is_admin=True)

    delete_warns = [
        r for r in caplog.records if "stats delete_key failed" in r.message
    ]
    assert len(delete_warns) == 1
    msg = delete_warns[0].message
    assert bad_id not in msg          # 原文换行不出现
    assert "\n" not in msg            # 无裸换行
    assert "goneEVIL" in msg          # 脱敏后（换行剥离）出现


@pytest.mark.anyio
async def test_a2a_name_from_probe_record_display_name():
    """R1#5：record.display_name → a2a 条目 name；无 record → fallback；mcp tool_count → Admin details。"""
    rec_a2a = ProbeRecord(state="reachable", display_name="Remote Helper")
    rec_mcp = ProbeRecord(state="reachable", tool_count=3)
    svc = _probe_svc(
        records={("a2a", "a2a-1111-2222"): rec_a2a, ("mcp", "srv-on"): rec_mcp}
    )
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("a2a", "a2a-1111-2222")].name == "Remote Helper"
    assert items[("mcp", "srv-on")].details["tool_count"] == 3

    # 无 record → fallback 前缀名
    svc2 = _probe_svc(records={})
    snap2 = await svc2.get_extensions(user_id="u1", is_admin=True)
    items2 = {(i.kind, i.id): i for i in snap2.items}
    assert items2[("a2a", "a2a-1111-2222")].name == "A2A a2a-1111"


@pytest.mark.anyio
async def test_get_makes_zero_network_calls():
    """INV-B9-4：GET 只触达 snapshot()/reconcile()——fake views 无 probe_* 方法被调。"""
    rec = ProbeRecord(state="reachable", last_checked_at=datetime.now(timezone.utc))
    liveness = LivenessSnapshot(active={}, degraded=False)
    svc = _probe_svc(records={("mcp", "srv-on"): rec}, liveness=liveness)
    await svc.get_extensions(user_id="u1", is_admin=True)
    fake_probe = svc._probe_view
    fake_liveness = svc._liveness_view
    # 只调只读快照 + 惰性 reconcile；无任何探测入口
    assert fake_probe.snapshot_calls >= 1
    assert len(fake_probe.reconcile_calls) == 1
    assert fake_liveness.snapshot_calls >= 1
    assert not hasattr(fake_probe, "probe_one_manual") or not callable(
        getattr(fake_probe, "probe_one_manual", None)
    )


# ---------------------------------------------------------------------- #
# Task 20 (PR-3): GET stats 接入真实 read_many 数据
# ---------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_runtime_service_stats_happy_path():
    """Admin + flag on + read_many 返回该 key 的 ExtensionStatsData
    → available=True + 计数/时间戳逐字段透传。"""
    ts = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)
    data = ExtensionStatsData(
        call_count=7, success_count=5, failure_count=2,
        last_active_at=ts, last_success_at=ts, last_failure_at=ts,
    )
    reader = MagicMock()
    reader.read_many = AsyncMock(return_value={("mcp", "srv-on"): data})
    svc = _probe_svc(stats_reader=reader, stats_enabled=True)

    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    items = {(i.kind, i.id): i for i in snap.items}
    st = items[("mcp", "srv-on")].stats
    assert st.available is True
    assert st.unavailable_reason is None
    assert (st.call_count, st.success_count, st.failure_count) == (7, 5, 2)
    assert st.last_active_at == ts
    assert st.last_success_at == ts
    assert st.last_failure_at == ts
    # 无数据的 mcp（srv-off 是 disabled，不在 map 里）→ available=False + reason
    off = items[("mcp", "srv-off")].stats
    assert off.available is False and off.unavailable_reason is not None


@pytest.mark.anyio
async def test_runtime_service_stats_reason_matrix():
    """R8#7 组合矩阵：available=false 必带 reason。"""
    # 非Admin → admin_only（压过一切）
    snap = await _probe_svc(stats_enabled=True).get_extensions(
        user_id="u1", is_admin=False
    )
    for it in snap.items:
        assert it.stats.available is False
        assert it.stats.unavailable_reason == "admin_only"

    # Admin + A2A → unsupported（即使 flag on + reader 有数据）
    reader = MagicMock()
    reader.read_many = AsyncMock(return_value={
        ("a2a", "a2a-1111-2222"): ExtensionStatsData(call_count=9),
    })
    snap = await _probe_svc(stats_reader=reader, stats_enabled=True).get_extensions(
        user_id="u1", is_admin=True
    )
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("a2a", "a2a-1111-2222")].stats.available is False
    assert items[("a2a", "a2a-1111-2222")].stats.unavailable_reason == "unsupported"

    # Admin + mcp + flag off → disabled（reader 不被读）
    reader2 = MagicMock()
    reader2.read_many = AsyncMock()
    snap = await _probe_svc(stats_reader=reader2, stats_enabled=False).get_extensions(
        user_id="u1", is_admin=True
    )
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("mcp", "srv-on")].stats.unavailable_reason == "disabled"
    reader2.read_many.assert_not_awaited()

    # Admin + flag on + read_many 抛 → redis_unavailable（broad-catch 降级）
    reader3 = MagicMock()
    reader3.read_many = AsyncMock(side_effect=RuntimeError("redis down"))
    snap = await _probe_svc(stats_reader=reader3, stats_enabled=True).get_extensions(
        user_id="u1", is_admin=True
    )
    items = {(i.kind, i.id): i for i in snap.items}
    assert items[("mcp", "srv-on")].stats.available is False
    assert items[("mcp", "srv-on")].stats.unavailable_reason == "redis_unavailable"
