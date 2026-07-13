"""D1a Task 20 — ExtensionGovernanceService 单测（fake ports；§5.2/§9.2）。

覆盖：mcp refresh 的 config→reset→surface 顺序唯一化 + reset-fail 中止、surface
conflict→409、skill 本地重算 + I/O 缺失→probe_failed、状态前置（deleted→404 /
disabled·quarantined 照常观测）、approve_pins all-active-only + items 逐项细分、
batch refresh 四 outcome、summary passthrough、audit 复合游标翻页。

fake ports 用一个组合对象承载 read/write/admission/prober 四面（duck-typed），
事件序列由 ``events`` list 记录以断言执行顺序（§5.2 R32#2/R33#2）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.application.errors.exceptions import NotFoundError
from app.application.services.extension_governance_service import (
    ExtensionGovernanceService,
    ItemOutcome,
    RefreshResult,
    _reduce_refresh_outcome,
)
from app.domain.external.extension_admission import (
    AdmissionDecision,
    AuditLogEntry,
    AuditPage,
    GovernanceCounters,
)
from app.domain.models.app_config import (
    A2AServerConfig,
    MCPServerConfig,
    MCPTransport,
)
from app.domain.models.extension_governance import RevisionConflictError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------- fakes -------


def _row(kind, ext_id, *, status="active", row_revision=1, version=None):
    """轻量 row 快照（service 仅读 kind/ext_id/status/row_revision/version）。"""
    return SimpleNamespace(
        kind=kind, ext_id=ext_id, status=status,
        row_revision=row_revision, version=version)


def _decision(*, observation_outcome="persisted", row_revision=1,
              config_drift_detected=False):
    return AdmissionDecision(
        admitted=True, reason="ok", row_revision=row_revision,
        observation_outcome=observation_outcome,
        config_drift_detected=config_drift_detected)


class Fakes:
    """组合 read/write/admission/prober fake（同一对象承载四面）。"""

    def __init__(self):
        self.events: list = []
        # read port state
        self.rows: dict[tuple[str, str], object] = {}
        self.counters: GovernanceCounters | None = None
        self.audit_entries: list[AuditLogEntry] = []
        # write port config
        self.reset_result = True
        self.approve_pin_results: dict[tuple[str, str], str] = {}
        # admission config
        self.mode = "enforce"
        self.check_many_result: dict[str, AdmissionDecision] = {}
        self.verify_by_ext: dict[str, AdmissionDecision] = {}
        self.verify_result: AdmissionDecision | None = None
        # prober config
        self.probe_by_ext: dict[str, object] = {}
        self.probe_result: object | None = None
        # app config + skills
        self.app_config = SimpleNamespace(
            mcp_config=SimpleNamespace(mcpServers={}),
            a2a_config=SimpleNamespace(a2a_servers=[]))
        self.skill_dirs: dict[str, object] = {}

    # ---- read port ----
    async def get_row(self, kind, ext_id):
        return self.rows.get((kind, ext_id))

    async def list_live_rows(self):
        return list(self.rows.values())

    async def governance_counters(self):
        return self.counters

    async def list_audit(self, *, kind=None, ext_id=None, event=None,
                         cursor=None, limit=50):
        self.events.append(("list_audit", event, cursor, limit))
        rows = sorted(self.audit_entries, key=lambda e: (e.created_at, str(e.id)))
        if kind is not None:
            rows = [e for e in rows if e.kind == kind]
        if ext_id is not None:
            rows = [e for e in rows if e.ext_id == ext_id]
        if event is not None:
            rows = [e for e in rows if e.event == event]
        if cursor is not None:
            c_created, c_id = cursor.rsplit("|", 1)
            key = (datetime.fromisoformat(c_created), c_id)
            rows = [e for e in rows if (e.created_at, str(e.id)) > key]
        has_more = len(rows) > limit
        page = rows[:limit]
        nxt = (f"{page[-1].created_at.isoformat()}|{page[-1].id}"
               if has_more and page else None)
        return AuditPage(entries=page, next_cursor=nxt)

    # ---- write port ----
    async def reset_pins_after_config_drift(self, kind, ext_id, *, row_revision):
        self.events.append(("reset", kind, ext_id, row_revision))
        return self.reset_result

    async def approve_pin(self, kind, ext_id, *, expected_row_revision, actor_user_id):
        self.events.append(
            ("approve_pin", kind, ext_id, expected_row_revision, actor_user_id))
        return self.approve_pin_results.get((kind, ext_id), "pinned")

    async def quarantine(self, kind, ext_id, *, expected_row_revision,
                         actor_user_id, note=None):
        self.events.append(
            ("quarantine", kind, ext_id, expected_row_revision, actor_user_id, note))
        return 99

    async def reapprove(self, kind, ext_id, *, expected_row_revision, actor_user_id):
        self.events.append(("reapprove", kind, ext_id, expected_row_revision, actor_user_id))
        return 98

    async def set_governance_enabled(self, kind, ext_id, *, enabled,
                                     expected_row_revision, actor_user_id):
        self.events.append(
            ("set_enabled", kind, ext_id, enabled, expected_row_revision, actor_user_id))
        return 97

    # ---- admission port ----
    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        self.events.append(("check_many", kind, list(ext_ids)))
        return self.check_many_result

    async def verify_observation(self, kind, ext_id, obs):
        self.events.append(("verify", kind, ext_id, obs.category))
        if ext_id in self.verify_by_ext:
            return self.verify_by_ext[ext_id]
        return self.verify_result

    # ---- prober ----
    async def probe_mcp(self, ext_id, config):
        self.events.append(("probe_mcp", ext_id))
        return self.probe_by_ext.get(ext_id, self.probe_result)

    async def probe_a2a(self, config):
        self.events.append(("probe_a2a", config.id))
        return self.probe_by_ext.get(config.id, self.probe_result)


class _SkillRepo:
    def __init__(self, dirs):
        self._dirs = dirs

    def get_skill_dir(self, skill_id):
        return self._dirs.get(skill_id)


def _service(fakes, *, plugins_root=None):
    return ExtensionGovernanceService(
        read_port=fakes,
        write_port=fakes,
        admission_port=fakes,
        prober=fakes,
        app_config_provider=lambda: fakes.app_config,
        skill_repository=_SkillRepo(fakes.skill_dirs),
        plugins_root=plugins_root,
    )


def _ok_probe(payload):
    return SimpleNamespace(ok=True, surface_payload=payload)


def _mcp_entry():
    return MCPServerConfig(transport=MCPTransport.STREAMABLE_HTTP,
                           url="https://srv.test/mcp")


# ---------------------------------------------------- 1. mcp drift order -----


async def test_refresh_mcp_drift_reset_order():
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv", row_revision=4)
    fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
    fakes.probe_result = _ok_probe([{"name": "t", "description": "d", "input_schema": {}}])
    fakes.check_many_result = {
        "srv": _decision(observation_outcome="persisted", row_revision=5,
                         config_drift_detected=True)}
    fakes.verify_result = _decision(observation_outcome="persisted", row_revision=6)

    result = await _service(fakes).refresh_observation("mcp", "srv")

    assert result.outcome == "refreshed"
    assert result.row_revision == 6
    kinds = [e[0] for e in fakes.events]
    # 顺序唯一化：config 观测(check_many) → reset → surface 观测(verify)
    assert kinds.index("check_many") < kinds.index("reset") < kinds.index("verify")
    # probe 在 config 观测之前
    assert kinds.index("probe_mcp") < kinds.index("check_many")


async def test_refresh_mcp_reset_fail_aborts_before_surface():
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv", row_revision=4)
    fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
    fakes.probe_result = _ok_probe([{"name": "t", "description": "d", "input_schema": {}}])
    fakes.check_many_result = {
        "srv": _decision(observation_outcome="persisted", row_revision=5,
                         config_drift_detected=True)}
    fakes.reset_result = False  # CAS 丢失

    with pytest.raises(RevisionConflictError):
        await _service(fakes).refresh_observation("mcp", "srv")

    # surface 观测零调用（reset 失败即中止）
    assert not any(e[0] == "verify" for e in fakes.events)


# ------------------------------------------------- 2. surface conflict 409 ---


async def test_refresh_surface_gate_conflict_maps_409():
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv")
    fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
    fakes.probe_result = _ok_probe([{"name": "t", "description": "d", "input_schema": {}}])
    fakes.check_many_result = {"srv": _decision(observation_outcome="persisted")}
    fakes.verify_result = _decision(observation_outcome="conflict")

    with pytest.raises(RevisionConflictError):
        await _service(fakes).refresh_observation("mcp", "srv")


async def test_refresh_mcp_config_conflict_maps_409():
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv")
    fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
    fakes.probe_result = _ok_probe([{"name": "t", "description": "d", "input_schema": {}}])
    fakes.check_many_result = {"srv": _decision(observation_outcome="conflict")}

    with pytest.raises(RevisionConflictError):
        await _service(fakes).refresh_observation("mcp", "srv")
    # config conflict → 零 surface 观测
    assert not any(e[0] == "verify" for e in fakes.events)


async def test_refresh_mcp_double_observation_matrix():
    """双观测归约：persisted×unchanged→refreshed / 含 none→409。"""
    for cfg_oc, surf_oc in (("persisted", "unchanged"), ("unchanged", "persisted")):
        fakes = Fakes()
        fakes.rows[("mcp", "srv")] = _row("mcp", "srv")
        fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
        fakes.probe_result = _ok_probe([{"name": "t", "description": "d", "input_schema": {}}])
        fakes.check_many_result = {"srv": _decision(observation_outcome=cfg_oc, row_revision=5)}
        fakes.verify_result = _decision(observation_outcome=surf_oc, row_revision=6)
        result = await _service(fakes).refresh_observation("mcp", "srv")
        assert result.outcome == "refreshed"

    # 含 none（非 conflict 但非 persisted/unchanged）→ 保守 409
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv")
    fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
    fakes.probe_result = _ok_probe([{"name": "t", "description": "d", "input_schema": {}}])
    fakes.check_many_result = {"srv": _decision(observation_outcome="persisted")}
    fakes.verify_result = _decision(observation_outcome="none")
    with pytest.raises(RevisionConflictError):
        await _service(fakes).refresh_observation("mcp", "srv")


async def test_refresh_mcp_missing_entry_probe_failed():
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv", row_revision=3)
    # config 无该 server → probe_failed（不 500）
    result = await _service(fakes).refresh_observation("mcp", "srv")
    assert result.outcome == "probe_failed"
    assert result.row_revision == 3


async def test_refresh_mcp_probe_failure_probe_failed():
    fakes = Fakes()
    fakes.rows[("mcp", "srv")] = _row("mcp", "srv", row_revision=3)
    fakes.app_config.mcp_config.mcpServers = {"srv": _mcp_entry()}
    fakes.probe_result = SimpleNamespace(ok=False, surface_payload=None)
    result = await _service(fakes).refresh_observation("mcp", "srv")
    assert result.outcome == "probe_failed"
    assert result.row_revision == 3


# ----------------------------------------- 3. skill local recompute ----------


def _make_skill_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_text('{"name": "x"}', encoding="utf-8")
    return d


async def test_refresh_skill_local_recompute(tmp_path):
    fakes = Fakes()
    fakes.rows[("skill", "s1")] = _row("skill", "s1")
    fakes.skill_dirs["s1"] = _make_skill_dir(tmp_path, "s1")
    fakes.verify_result = _decision(observation_outcome="persisted", row_revision=7)

    result = await _service(fakes).refresh_observation("skill", "s1")

    assert result.outcome == "refreshed"
    assert result.row_revision == 7
    assert any(e[0] == "verify" and e[3] == "artifact" for e in fakes.events)


async def test_refresh_skill_missing_dir_probe_failed(tmp_path):
    fakes = Fakes()
    fakes.rows[("skill", "s1")] = _row("skill", "s1", row_revision=2)
    fakes.skill_dirs["s1"] = tmp_path / "does_not_exist"

    result = await _service(fakes).refresh_observation("skill", "s1")

    assert result.outcome == "probe_failed"
    assert result.row_revision == 2
    # I/O 缺失绝不触碰观测
    assert not any(e[0] == "verify" for e in fakes.events)


async def test_refresh_plugin_local_recompute(tmp_path):
    fakes = Fakes()
    fakes.rows[("plugin", "p1")] = _row("plugin", "p1", version="1.0")
    root = tmp_path / "plugins"
    (root / "p1" / "1.0").mkdir(parents=True)
    (root / "p1" / "1.0" / "manifest.json").write_text("{}", encoding="utf-8")
    fakes.verify_result = _decision(observation_outcome="unchanged", row_revision=4)

    result = await _service(fakes, plugins_root=root).refresh_observation("plugin", "p1")
    assert result.outcome == "refreshed"
    assert result.row_revision == 4


# ------------------------------------------- 4. state preconditions ----------


async def test_refresh_deleted_or_absent_raises_notfound():
    fakes = Fakes()  # 无行
    with pytest.raises(NotFoundError):
        await _service(fakes).refresh_observation("skill", "gone")


async def test_refresh_disabled_row_still_observes(tmp_path):
    fakes = Fakes()
    fakes.rows[("skill", "s1")] = _row("skill", "s1", status="disabled")
    fakes.skill_dirs["s1"] = _make_skill_dir(tmp_path, "s1")
    fakes.verify_result = _decision(observation_outcome="persisted", row_revision=9)
    # 服务不做状态豁免——disabled 照常走观测路
    result = await _service(fakes).refresh_observation("skill", "s1")
    assert result.outcome == "refreshed"


# ------------------------------------------- 5. approve_pins -----------------


async def test_approve_pins_all_scans_active_only():
    fakes = Fakes()
    fakes.rows[("mcp", "a")] = _row("mcp", "a", status="active", row_revision=1)
    fakes.rows[("a2a", "b")] = _row("a2a", "b", status="quarantined", row_revision=1)
    fakes.rows[("skill", "c")] = _row("skill", "c", status="disabled", row_revision=1)

    outcomes = await _service(fakes).approve_pins(all=True, actor_id="admin")

    approve_calls = [e for e in fakes.events if e[0] == "approve_pin"]
    assert len(approve_calls) == 1
    assert approve_calls[0][1:] == ("mcp", "a", 1, "admin")
    assert outcomes == [ItemOutcome(kind="mcp", ext_id="a", outcome="pinned")]


async def test_approve_pins_items_mixed_states():
    fakes = Fakes()
    fakes.rows[("mcp", "act")] = _row("mcp", "act", status="active", row_revision=3)
    fakes.rows[("skill", "dis")] = _row("skill", "dis", status="disabled", row_revision=3)
    # disabled 行由 write.approve_pin 内部前置回报 skipped_invalid_state
    fakes.approve_pin_results[("skill", "dis")] = "skipped_invalid_state"

    items = [
        SimpleNamespace(kind="mcp", ext_id="act", expected_row_revision=None),
        SimpleNamespace(kind="skill", ext_id="dis", expected_row_revision=None),
        SimpleNamespace(kind="mcp", ext_id="unknown", expected_row_revision=5),
    ]
    outcomes = await _service(fakes).approve_pins(items=items, actor_id="admin")

    assert outcomes == [
        ItemOutcome(kind="mcp", ext_id="act", outcome="pinned"),
        ItemOutcome(kind="skill", ext_id="dis", outcome="skipped_invalid_state"),
        ItemOutcome(kind="mcp", ext_id="unknown", outcome="skipped_invalid_state"),
    ]
    # row_revision 可空 → fallback 取 fetched row.row_revision(=3)
    act_call = next(e for e in fakes.events if e[0] == "approve_pin" and e[2] == "act")
    assert act_call[3] == 3


async def test_approve_pins_items_conflict_passthrough():
    fakes = Fakes()
    fakes.rows[("mcp", "act")] = _row("mcp", "act", status="active", row_revision=2)
    fakes.approve_pin_results[("mcp", "act")] = "conflict"
    items = [SimpleNamespace(kind="mcp", ext_id="act", expected_row_revision=2)]
    outcomes = await _service(fakes).approve_pins(items=items, actor_id="admin")
    assert outcomes == [ItemOutcome(kind="mcp", ext_id="act", outcome="conflict")]
    # 显式 expected_row_revision 透传（非 fallback）
    call = next(e for e in fakes.events if e[0] == "approve_pin")
    assert call[3] == 2


# ------------------------------------------- 6. batch refresh outcomes -------


async def test_refresh_batch_outcomes(tmp_path):
    fakes = Fakes()
    fakes.rows[("skill", "ok")] = _row("skill", "ok")
    fakes.rows[("skill", "gone")] = _row("skill", "gone")
    fakes.rows[("skill", "conf")] = _row("skill", "conf")
    fakes.skill_dirs["ok"] = _make_skill_dir(tmp_path, "ok")
    fakes.skill_dirs["conf"] = _make_skill_dir(tmp_path, "conf")
    fakes.skill_dirs["gone"] = tmp_path / "nope"
    fakes.verify_by_ext["ok"] = _decision(observation_outcome="persisted", row_revision=1)
    fakes.verify_by_ext["conf"] = _decision(observation_outcome="conflict")

    items = [
        SimpleNamespace(kind="skill", ext_id="ok"),
        SimpleNamespace(kind="skill", ext_id="gone"),
        SimpleNamespace(kind="skill", ext_id="conf"),
        SimpleNamespace(kind="skill", ext_id="missing"),  # 无行 → skipped
    ]
    outcomes = await _service(fakes).refresh_observations_batch(items=items)

    by_ext = {o.ext_id: o.outcome for o in outcomes}
    assert by_ext == {
        "ok": "refreshed",
        "gone": "probe_failed",
        "conf": "conflict",
        "missing": "skipped_invalid_state",
    }


async def test_refresh_batch_all_scans_live_rows(tmp_path):
    fakes = Fakes()
    fakes.rows[("skill", "ok")] = _row("skill", "ok")
    fakes.skill_dirs["ok"] = _make_skill_dir(tmp_path, "ok")
    fakes.verify_result = _decision(observation_outcome="persisted", row_revision=1)
    outcomes = await _service(fakes).refresh_observations_batch(all=True)
    assert [o.ext_id for o in outcomes] == ["ok"]
    assert outcomes[0].outcome == "refreshed"


# ------------------------------------------- 7. summary ----------------------


async def test_summary_counters_passthrough():
    fakes = Fakes()
    fakes.mode = "shadow"
    fakes.counters = GovernanceCounters(
        unpinned_count=1, missing_observation_count=2, quarantined_count=3)
    result = await _service(fakes).summary()
    assert result == {
        "mode": "shadow",
        "unpinned_count": 1,
        "missing_observation_count": 2,
        "quarantined_count": 3,
    }


# ------------------------------------------- 8. audit pagination -------------


async def test_audit_cursor_pagination():
    fakes = Fakes()
    base = datetime(2026, 7, 11, tzinfo=timezone.utc)
    for i in range(7):
        fakes.audit_entries.append(AuditLogEntry(
            id=uuid.UUID(int=i), kind="mcp", ext_id="srv",
            actor_user_id="admin", event="quarantined",
            before=None, after=None, details=None, correlation_id=None,
            created_at=base + timedelta(seconds=i)))
    svc = _service(fakes)

    seen: list = []
    cursor = None
    pages = 0
    while True:
        page = await svc.list_audit(event="quarantined", cursor=cursor, limit=3)
        pages += 1
        seen.extend(e.id for e in page.entries)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
        assert pages < 10  # 防死循环

    assert pages == 3  # 7 条 / limit 3 → 3 页
    assert len(seen) == 7
    assert len(set(seen)) == 7  # 无重叠稳定翻页


# --------------------------------- admin action passthroughs -----------------


async def test_admin_actions_passthrough():
    fakes = Fakes()
    svc = _service(fakes)
    assert await svc.quarantine("mcp", "a", expected_row_revision=1,
                                actor_id="admin", note="x") == 99
    assert await svc.reapprove("mcp", "a", expected_row_revision=1, actor_id="admin") == 98
    assert await svc.set_enabled("mcp", "a", enabled=False,
                                 expected_row_revision=1, actor_id="admin") == 97
    events = {e[0] for e in fakes.events}
    assert {"quarantine", "reapprove", "set_enabled"} <= events


# --------------------------------- pure reduce fn ----------------------------


def test_reduce_refresh_outcome_pure():
    assert _reduce_refresh_outcome([_decision(observation_outcome="persisted")]) == "refreshed"
    assert _reduce_refresh_outcome([
        _decision(observation_outcome="persisted"),
        _decision(observation_outcome="unchanged"),
    ]) == "refreshed"
    with pytest.raises(RevisionConflictError):
        _reduce_refresh_outcome([_decision(observation_outcome="conflict")])
    with pytest.raises(RevisionConflictError):
        _reduce_refresh_outcome([_decision(observation_outcome="none")])
