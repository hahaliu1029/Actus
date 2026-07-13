"""D1a §6.1 治理写/读 port — DbExtensionRegistryWritePort/ReadPort 集成测试（需 PostgreSQL）。

autouse `_migrate`（tests/integration/conftest.py）跑 alembic upgrade head →
建 extensions / plugin_memberships / extension_install_operations / extension_audit_log 表。

port 自管事务（每方法 `session.begin()` 提交），因此本文件**不用** auto-rollback 的
`db_session` fixture——setup 行显式 COMMIT，断言用独立 session 重读，用例间靠**唯一
ext_id** 隔离（governance_counters 全表聚合的用例用 before/after 增量断言绕开累积污染）。

**集成未本地跑，CI 验证。**

Run:
    cd api && uv run pytest tests/integration/governance/test_db_extension_registry.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.external.extension_admission import InstallContext, UninstallContext
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    GovernanceScanSummary,
    InvalidStateTransitionError,
    ManagedByPluginError,
    MissingObservationError,
    OperationPendingError,
    RevisionConflictError,
)
from app.infrastructure.external.governance import db_extension_registry as reg_mod
from app.infrastructure.external.governance.db_extension_registry import (
    DbExtensionRegistryReadPort,
    DbExtensionRegistryWritePort,
)
from app.infrastructure.models.extension_governance import (
    ExtensionAuditLogModel,
    ExtensionInstallOperationModel,
    ExtensionModel,
    PluginMembershipModel,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def factory(async_engine) -> async_sessionmaker[AsyncSession]:
    """自建 async_sessionmaker——port + setup/断言共用同一 engine。"""
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def _eid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _scan(verdict: str = "safe") -> GovernanceScanSummary:
    return GovernanceScanSummary(verdict=verdict, finding_count=0, findings=[])


def _install_ctx(**overrides: Any) -> InstallContext:
    base: dict[str, Any] = dict(
        actor_user_id="admin-1",
        correlation_id=None,
        source_type="config",
        source_ref=None,
        version=None,
        trust_origin="user_installed",
        artifact_hash=None,
        surface_hash=None,
        config_fingerprint=None,
        hash_schema_version=HASH_SCHEMA_VERSION,
        scan=None,
    )
    base.update(overrides)
    return InstallContext(**base)


async def _insert_ext(
    factory: async_sessionmaker[AsyncSession], *, kind: str, ext_id: str, **cols: Any,
) -> ExtensionModel:
    row = ExtensionModel(
        kind=kind,
        ext_id=ext_id,
        status=cols.pop("status", "active"),
        trust_origin=cols.pop("trust_origin", "user_installed"),
        source_type=cols.pop("source_type", "config"),
        hash_schema_version=cols.pop("hash_schema_version", HASH_SCHEMA_VERSION),
        **cols,
    )
    async with factory() as session:
        async with session.begin():
            session.add(row)
    return row


async def _set_cols(
    factory: async_sessionmaker[AsyncSession], *, kind: str, ext_id: str, **cols: Any,
) -> None:
    async with factory() as session:
        async with session.begin():
            await session.execute(
                update(ExtensionModel)
                .where(ExtensionModel.kind == kind, ExtensionModel.ext_id == ext_id,
                       ExtensionModel.deleted_at.is_(None))
                .values(**cols))


async def _insert_membership(
    factory: async_sessionmaker[AsyncSession], *,
    plugin_id: uuid.UUID, child_extension_id: uuid.UUID, declared_component_id: str = "c1",
) -> None:
    async with factory() as session:
        async with session.begin():
            session.add(PluginMembershipModel(
                plugin_id=plugin_id, child_extension_id=child_extension_id,
                declared_component_id=declared_component_id))


async def _insert_operation(
    factory: async_sessionmaker[AsyncSession], *,
    plugin_extension_id: uuid.UUID, state: str, operation_type: str = "plugin_install",
) -> None:
    async with factory() as session:
        async with session.begin():
            session.add(ExtensionInstallOperationModel(
                operation_type=operation_type, plugin_extension_id=plugin_extension_id,
                initiated_by="admin-1", state=state, steps=[]))


async def _fetch(
    factory: async_sessionmaker[AsyncSession], *, kind: str, ext_id: str,
) -> ExtensionModel:
    async with factory() as session:
        result = await session.execute(
            select(ExtensionModel).where(
                ExtensionModel.kind == kind, ExtensionModel.ext_id == ext_id))
        return result.scalar_one()


async def _audit_count(
    factory: async_sessionmaker[AsyncSession], *, ext_id: str, event: str,
) -> int:
    async with factory() as session:
        result = await session.execute(
            select(func.count()).select_from(ExtensionAuditLogModel).where(
                ExtensionAuditLogModel.ext_id == ext_id,
                ExtensionAuditLogModel.event == event))
        return int(result.scalar_one())


async def _audit_rows(
    factory: async_sessionmaker[AsyncSession], *, ext_id: str,
) -> list[ExtensionAuditLogModel]:
    async with factory() as session:
        result = await session.execute(
            select(ExtensionAuditLogModel).where(ExtensionAuditLogModel.ext_id == ext_id))
        return list(result.scalars().all())


async def _audit_details(
    factory: async_sessionmaker[AsyncSession], *, ext_id: str, event: str,
) -> dict[str, Any] | None:
    for row in await _audit_rows(factory, ext_id=ext_id):
        if row.event == event:
            return row.details
    return None


# 1 ---------------------------------------------------------------------------
async def test_record_install_insert_then_update_reinstall(factory):
    """R46#6/R48#1 断言③：首装 insert（pins+3 audit）；预置 observed 后重装 →
    pin 重建、observed 四列 NULL、status 不变、revision 已 bump（旧 revision approve→conflict）。"""
    ext_id = _eid("skill-reinstall")
    write = DbExtensionRegistryWritePort(factory)

    # 首装：insert
    await write.record_install("skill", ext_id, _install_ctx(artifact_hash="a1", scan=_scan()))
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.artifact_hash == "a1" and row.status == "active"
    assert row.pinned_at is not None and row.pinned_by == "admin-1"
    assert row.row_revision == 0
    assert await _audit_count(factory, ext_id=ext_id, event="installed") == 1
    assert await _audit_count(factory, ext_id=ext_id, event="pin_established") == 1
    assert await _audit_count(factory, ext_id=ext_id, event="scan_recorded") == 1

    # 预置 observed 三列非 NULL（模拟 admission 观测后）
    await _set_cols(
        factory, kind="skill", ext_id=ext_id,
        observed_surface_hash="OS", observed_artifact_hash="OA",
        observed_config_fingerprint="OC", observed_hash_schema_version=HASH_SCHEMA_VERSION)

    # 重装（update-only，standalone 无 membership）
    await write.record_install("skill", ext_id, _install_ctx(artifact_hash="a2", scan=_scan()))
    row2 = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row2.artifact_hash == "a2"                     # pin 重建
    assert row2.observed_surface_hash is None             # R46#6：observed 四列清空
    assert row2.observed_artifact_hash is None
    assert row2.observed_config_fingerprint is None
    assert row2.observed_hash_schema_version is None
    assert row2.status == "active"                        # R44#3：status 不变
    assert row2.row_revision == 1                         # R48#1：CAS bump
    assert await _audit_count(factory, ext_id=ext_id, event="installed") == 2

    # 旧 revision approve → conflict（证明 revision 已 bump）
    await _set_cols(
        factory, kind="skill", ext_id=ext_id,
        observed_artifact_hash="a2", observed_hash_schema_version=HASH_SCHEMA_VERSION)
    outcome = await write.approve_pin(
        "skill", ext_id, expected_row_revision=0, actor_user_id="admin-1")
    assert outcome == "conflict"


# 2 ---------------------------------------------------------------------------
async def test_record_install_membership_guard(factory):
    """R32#4：成员行 + standalone(correlation None) → ManagedByPluginError；
    saga(correlation 非 None) → 放行 update。"""
    write = DbExtensionRegistryWritePort(factory)
    parent = await _insert_ext(factory, kind="plugin", ext_id=_eid("plugin-parent"), artifact_hash="pa")
    child_ext = _eid("mcp-child")
    child = await _insert_ext(
        factory, kind="mcp", ext_id=child_ext, surface_hash="s0", config_fingerprint="c0")
    await _insert_membership(factory, plugin_id=parent.id, child_extension_id=child.id)

    # standalone 命中成员行 → 拒绝并回滚
    with pytest.raises(ManagedByPluginError):
        await write.record_install("mcp", child_ext, _install_ctx(
            correlation_id=None, surface_hash="s1", config_fingerprint="c1"))
    rolled = await _fetch(factory, kind="mcp", ext_id=child_ext)
    assert rolled.surface_hash == "s0" and rolled.row_revision == 0

    # saga context → 放行 update
    await write.record_install("mcp", child_ext, _install_ctx(
        correlation_id=uuid.uuid4(), surface_hash="s1", config_fingerprint="c1"))
    updated = await _fetch(factory, kind="mcp", ext_id=child_ext)
    assert updated.surface_hash == "s1" and updated.row_revision == 1


# 3 ---------------------------------------------------------------------------
async def test_record_install_cas_retry(factory, monkeypatch):
    """R49#3 断言③：陈旧快照 CAS 落败前两次 → 第三次成功；全部耗尽 → RevisionConflictError。"""
    write = DbExtensionRegistryWritePort(factory)
    real_load = reg_mod._load_live

    # (a) 前两次陈旧 → 第三次成功
    ext_a = _eid("skill-casretry")
    await _insert_ext(factory, kind="skill", ext_id=ext_a, artifact_hash="h0", row_revision=5)
    counter = {"n": 0}

    async def _stale_twice(session, kind, ext_id):
        counter["n"] += 1
        row = await real_load(session, kind, ext_id)
        if row is None:
            return None
        if counter["n"] <= 2:
            return SimpleNamespace(id=row.id, row_revision=row.row_revision - 1)
        return row

    monkeypatch.setattr(reg_mod, "_load_live", _stale_twice)
    await write.record_install("skill", ext_a, _install_ctx(artifact_hash="h1"))
    row_a = await _fetch(factory, kind="skill", ext_id=ext_a)
    assert counter["n"] == 3
    assert row_a.artifact_hash == "h1" and row_a.row_revision == 6
    assert await _audit_count(factory, ext_id=ext_a, event="installed") == 1

    # (b) 永远陈旧 → 耗尽 3 次 → RevisionConflictError，行零写零 audit
    ext_b = _eid("skill-casexhaust")
    await _insert_ext(factory, kind="skill", ext_id=ext_b, artifact_hash="h0", row_revision=5)

    async def _always_stale(session, kind, ext_id):
        row = await real_load(session, kind, ext_id)
        if row is None:
            return None
        return SimpleNamespace(id=row.id, row_revision=row.row_revision - 1)

    monkeypatch.setattr(reg_mod, "_load_live", _always_stale)
    with pytest.raises(RevisionConflictError):
        await write.record_install("skill", ext_b, _install_ctx(artifact_hash="h1"))
    row_b = await _fetch(factory, kind="skill", ext_id=ext_b)
    assert row_b.artifact_hash == "h0" and row_b.row_revision == 5
    assert await _audit_count(factory, ext_id=ext_b, event="installed") == 0


# 4 ---------------------------------------------------------------------------
async def test_record_delete_idempotent(factory):
    """软删 + uninstalled audit 双字段；重复零 audit；+R1#17 纵深 membership 守卫。"""
    write = DbExtensionRegistryWritePort(factory)

    # 软删 + audit 双字段
    ext_id = _eid("skill-del")
    await _insert_ext(factory, kind="skill", ext_id=ext_id, artifact_hash="a", row_revision=2)
    corr = uuid.uuid4()
    await write.record_delete(
        "skill", ext_id, uninstall_context=UninstallContext(correlation_id=corr, actor_user_id="admin-1"))
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.deleted_at is not None and row.row_revision == 3
    audits = [r for r in await _audit_rows(factory, ext_id=ext_id) if r.event == "uninstalled"]
    assert len(audits) == 1
    assert audits[0].correlation_id == corr and audits[0].actor_user_id == "admin-1"

    # 幂等：重复调用零新 audit
    await write.record_delete(
        "skill", ext_id, uninstall_context=UninstallContext(correlation_id=None, actor_user_id="admin-1"))
    assert await _audit_count(factory, ext_id=ext_id, event="uninstalled") == 1

    # R1#17 纵深：standalone 命中带 membership 行 → 拒绝、回滚零写；saga → 放行
    parent = await _insert_ext(factory, kind="plugin", ext_id=_eid("plugin-p"), artifact_hash="pa")
    mem_ext = _eid("mcp-member")
    child = await _insert_ext(factory, kind="mcp", ext_id=mem_ext, surface_hash="s", config_fingerprint="c")
    await _insert_membership(factory, plugin_id=parent.id, child_extension_id=child.id)
    with pytest.raises(ManagedByPluginError):
        await write.record_delete(
            "mcp", mem_ext, uninstall_context=UninstallContext(correlation_id=None, actor_user_id="admin-1"))
    still = await _fetch(factory, kind="mcp", ext_id=mem_ext)
    assert still.deleted_at is None
    assert await _audit_count(factory, ext_id=mem_ext, event="uninstalled") == 0
    await write.record_delete(
        "mcp", mem_ext, uninstall_context=UninstallContext(correlation_id=uuid.uuid4(), actor_user_id="admin-1"))
    deleted = await _fetch(factory, kind="mcp", ext_id=mem_ext)
    assert deleted.deleted_at is not None


# 5 ---------------------------------------------------------------------------
async def test_admin_transitions_per_state_table(factory):
    """§2 表逐行：quarantine/disable/enable/reapprove 前置条件 + 软删 404 + CAS 失配。"""
    write = DbExtensionRegistryWritePort(factory)

    # quarantine(active) → ok
    active_ext = _eid("skill-active")
    await _insert_ext(factory, kind="skill", ext_id=active_ext, artifact_hash="a", row_revision=0)
    new_rev = await write.quarantine(
        "skill", active_ext, expected_row_revision=0, actor_user_id="admin-1")
    row = await _fetch(factory, kind="skill", ext_id=active_ext)
    assert new_rev == 1 and row.status == "quarantined" and row.quarantine_reason == "admin_manual"
    assert await _audit_count(factory, ext_id=active_ext, event="quarantined") == 1

    # quarantine(quarantined) → InvalidStateTransition
    with pytest.raises(InvalidStateTransitionError):
        await write.quarantine("skill", active_ext, expected_row_revision=1, actor_user_id="admin-1")
    # disable(quarantined) → InvalidStateTransition（不掩埋隔离证据）
    with pytest.raises(InvalidStateTransitionError):
        await write.set_governance_enabled(
            "skill", active_ext, enabled=False, expected_row_revision=1, actor_user_id="admin-1")

    # enable(disabled) → ok
    disabled_ext = _eid("skill-disabled")
    await _insert_ext(
        factory, kind="skill", ext_id=disabled_ext, status="disabled", artifact_hash="a", row_revision=0)
    erev = await write.set_governance_enabled(
        "skill", disabled_ext, enabled=True, expected_row_revision=0, actor_user_id="admin-1")
    drow = await _fetch(factory, kind="skill", ext_id=disabled_ext)
    assert erev == 1 and drow.status == "active"
    assert await _audit_count(factory, ext_id=disabled_ext, event="enabled") == 1

    # reapprove(active) → InvalidStateTransition
    with pytest.raises(InvalidStateTransitionError):
        await write.reapprove(
            "skill", disabled_ext, expected_row_revision=1, actor_user_id="admin-1")

    # 软删行 → 全部行政动作 NotFoundError
    gone_ext = _eid("skill-gone")
    await _insert_ext(factory, kind="skill", ext_id=gone_ext, artifact_hash="a")
    await _set_cols(factory, kind="skill", ext_id=gone_ext, deleted_at=datetime.now(timezone.utc))
    from app.application.errors.exceptions import NotFoundError
    for action in (
        lambda: write.quarantine("skill", gone_ext, expected_row_revision=0, actor_user_id="a"),
        lambda: write.reapprove("skill", gone_ext, expected_row_revision=0, actor_user_id="a"),
        lambda: write.set_governance_enabled(
            "skill", gone_ext, enabled=True, expected_row_revision=0, actor_user_id="a"),
    ):
        with pytest.raises(NotFoundError):
            await action()

    # CAS 失配 → RevisionConflictError
    cas_ext = _eid("skill-cas")
    await _insert_ext(factory, kind="skill", ext_id=cas_ext, artifact_hash="a", row_revision=0)
    with pytest.raises(RevisionConflictError):
        await write.quarantine("skill", cas_ext, expected_row_revision=999, actor_user_id="admin-1")


# 6 ---------------------------------------------------------------------------
async def test_reapprove_full_atomicity(factory):
    """R43#3 断言①：缺 observed_config / observed 版本过期 → MissingObservationError；
    齐且版本=当前 → observed 双列转正 pins + status=active + reapproved+pin_established 双 audit。"""
    write = DbExtensionRegistryWritePort(factory)

    # 缺 observed_config_fingerprint（mcp 必需 surface+config_fingerprint）
    miss_ext = _eid("mcp-missobs")
    await _insert_ext(
        factory, kind="mcp", ext_id=miss_ext, status="quarantined", quarantine_reason="admin_manual",
        observed_surface_hash="S", observed_hash_schema_version=HASH_SCHEMA_VERSION, row_revision=1)
    with pytest.raises(MissingObservationError):
        await write.reapprove("mcp", miss_ext, expected_row_revision=1, actor_user_id="admin-1")

    # observed 齐但版本=0（过期）
    stale_ext = _eid("mcp-staleobs")
    await _insert_ext(
        factory, kind="mcp", ext_id=stale_ext, status="quarantined", quarantine_reason="admin_manual",
        observed_surface_hash="S", observed_config_fingerprint="C",
        observed_hash_schema_version=0, row_revision=1)
    with pytest.raises(MissingObservationError):
        await write.reapprove("mcp", stale_ext, expected_row_revision=1, actor_user_id="admin-1")

    # 齐且版本=当前 → 转正
    ok_ext = _eid("mcp-reapprove")
    await _insert_ext(
        factory, kind="mcp", ext_id=ok_ext, status="quarantined", quarantine_reason="admin_manual",
        observed_surface_hash="S", observed_config_fingerprint="C",
        observed_hash_schema_version=HASH_SCHEMA_VERSION, row_revision=2)
    new_rev = await write.reapprove("mcp", ok_ext, expected_row_revision=2, actor_user_id="admin-1")
    row = await _fetch(factory, kind="mcp", ext_id=ok_ext)
    assert new_rev == 3
    assert row.status == "active" and row.quarantine_reason is None
    assert row.surface_hash == "S" and row.config_fingerprint == "C"   # observed → pin 转正
    assert row.pinned_at is not None and row.pinned_by == "admin-1"
    assert await _audit_count(factory, ext_id=ok_ext, event="reapproved") == 1
    assert await _audit_count(factory, ext_id=ok_ext, event="pin_established") == 1


# 7 ---------------------------------------------------------------------------
async def test_reapprove_plugin_operation_pending(factory):
    """R46#2/R47#1：plugin 行存非终态 operation → reapprove/enable 均 OperationPendingError，行不变。"""
    write = DbExtensionRegistryWritePort(factory)

    # reapprove：quarantined plugin + in_progress operation
    reapp_ext = _eid("plugin-reapprove")
    plugin_a = await _insert_ext(
        factory, kind="plugin", ext_id=reapp_ext, status="quarantined", quarantine_reason="admin_manual",
        observed_artifact_hash="PA", observed_hash_schema_version=HASH_SCHEMA_VERSION, row_revision=1)
    await _insert_operation(factory, plugin_extension_id=plugin_a.id, state="in_progress")
    with pytest.raises(OperationPendingError):
        await write.reapprove("plugin", reapp_ext, expected_row_revision=1, actor_user_id="admin-1")
    row_a = await _fetch(factory, kind="plugin", ext_id=reapp_ext)
    assert row_a.status == "quarantined" and row_a.row_revision == 1

    # enable：disabled plugin + failed operation
    enable_ext = _eid("plugin-enable")
    plugin_b = await _insert_ext(
        factory, kind="plugin", ext_id=enable_ext, status="disabled",
        artifact_hash="PB", row_revision=1)
    await _insert_operation(
        factory, plugin_extension_id=plugin_b.id, state="failed", operation_type="plugin_uninstall")
    with pytest.raises(OperationPendingError):
        await write.set_governance_enabled(
            "plugin", enable_ext, enabled=True, expected_row_revision=1, actor_user_id="admin-1")
    row_b = await _fetch(factory, kind="plugin", ext_id=enable_ext)
    assert row_b.status == "disabled" and row_b.row_revision == 1


# 8 ---------------------------------------------------------------------------
async def test_approve_pin_outcomes(factory):
    """active 缺观测→skipped_no_observation；disabled→skipped_invalid_state；
    版本过期→skipped_no_observation；CAS 失配→conflict；正常→pinned+audit+版本升当前。"""
    write = DbExtensionRegistryWritePort(factory)

    # active 缺观测
    miss_ext = _eid("skill-nopin")
    await _insert_ext(factory, kind="skill", ext_id=miss_ext, artifact_hash="a", row_revision=0)
    assert await write.approve_pin(
        "skill", miss_ext, expected_row_revision=0, actor_user_id="a") == "skipped_no_observation"

    # disabled
    dis_ext = _eid("skill-dispin")
    await _insert_ext(
        factory, kind="skill", ext_id=dis_ext, status="disabled",
        observed_artifact_hash="a", observed_hash_schema_version=HASH_SCHEMA_VERSION, row_revision=0)
    assert await write.approve_pin(
        "skill", dis_ext, expected_row_revision=0, actor_user_id="a") == "skipped_invalid_state"

    # 观测版本过期
    stale_ext = _eid("skill-stalepin")
    await _insert_ext(
        factory, kind="skill", ext_id=stale_ext,
        observed_artifact_hash="a", observed_hash_schema_version=0, row_revision=0)
    assert await write.approve_pin(
        "skill", stale_ext, expected_row_revision=0, actor_user_id="a") == "skipped_no_observation"

    # CAS 失配
    conf_ext = _eid("skill-confpin")
    await _insert_ext(
        factory, kind="skill", ext_id=conf_ext,
        observed_artifact_hash="a", observed_hash_schema_version=HASH_SCHEMA_VERSION, row_revision=3)
    assert await write.approve_pin(
        "skill", conf_ext, expected_row_revision=99, actor_user_id="a") == "conflict"

    # 正常 → pinned + 版本升当前
    ok_ext = _eid("skill-okpin")
    await _insert_ext(
        factory, kind="skill", ext_id=ok_ext, hash_schema_version=0,
        observed_artifact_hash="a", observed_hash_schema_version=HASH_SCHEMA_VERSION, row_revision=7)
    assert await write.approve_pin(
        "skill", ok_ext, expected_row_revision=7, actor_user_id="admin-1") == "pinned"
    row = await _fetch(factory, kind="skill", ext_id=ok_ext)
    assert row.artifact_hash == "a"                       # observed → pin
    assert row.hash_schema_version == HASH_SCHEMA_VERSION  # 版本升当前
    assert row.pinned_at is not None and row.row_revision == 8
    assert await _audit_count(factory, ext_id=ok_ext, event="pin_established") == 1


# 9 ---------------------------------------------------------------------------
async def test_reset_pins_after_config_drift(factory):
    """R17#4：清 surface/config 两 pin + observed_surface 连带清 + config_changed audit；
    错误 revision → False 零写。"""
    write = DbExtensionRegistryWritePort(factory)

    ok_ext = _eid("mcp-drift")
    await _insert_ext(
        factory, kind="mcp", ext_id=ok_ext, surface_hash="S", config_fingerprint="C",
        observed_surface_hash="OS", observed_config_fingerprint="OC", row_revision=4)
    assert await write.reset_pins_after_config_drift("mcp", ok_ext, row_revision=4) is True
    row = await _fetch(factory, kind="mcp", ext_id=ok_ext)
    assert row.surface_hash is None and row.config_fingerprint is None
    assert row.observed_surface_hash is None              # R17#4：连带清 observed_surface
    assert row.observed_config_fingerprint == "OC"        # observed_config 不清
    assert row.row_revision == 5
    assert await _audit_count(factory, ext_id=ok_ext, event="config_changed") == 1

    # 错误 revision → False 零写
    bad_ext = _eid("mcp-driftbad")
    await _insert_ext(
        factory, kind="mcp", ext_id=bad_ext, surface_hash="S", config_fingerprint="C", row_revision=4)
    assert await write.reset_pins_after_config_drift("mcp", bad_ext, row_revision=99) is False
    bad = await _fetch(factory, kind="mcp", ext_id=bad_ext)
    assert bad.surface_hash == "S" and bad.row_revision == 4
    assert await _audit_count(factory, ext_id=bad_ext, event="config_changed") == 0


# 10 --------------------------------------------------------------------------
async def test_reconciled_family_idempotency(factory):
    """record_reconciled_seen 幂等；mark_source_missing/restored 幂等对；
    record_reconciled_missing standalone=软删/disposition=soft_deleted / plugin=保持/retained_plugin。"""
    write = DbExtensionRegistryWritePort(factory)

    # reconciled_seen 幂等
    seen_ext = _eid("mcp-seen")
    for _ in range(2):
        await write.record_reconciled_seen(
            "mcp", seen_ext, source_type="config", source_ref=None, version=None,
            trust_origin="user_installed")
    seen_row = await _fetch(factory, kind="mcp", ext_id=seen_ext)
    assert seen_row.status == "active" and seen_row.hash_schema_version == HASH_SCHEMA_VERSION
    assert await _audit_count(factory, ext_id=seen_ext, event="reconciled_seen") == 1

    # mark_source_missing / restored 幂等对
    src_ext = _eid("mcp-src")
    await _insert_ext(factory, kind="mcp", ext_id=src_ext, surface_hash="s")
    for _ in range(2):
        await write.mark_source_missing("mcp", src_ext)
    miss_row = await _fetch(factory, kind="mcp", ext_id=src_ext)
    assert miss_row.source_missing_at is not None
    assert await _audit_count(factory, ext_id=src_ext, event="source_missing") == 1
    for _ in range(2):
        await write.mark_source_restored("mcp", src_ext)
    rest_row = await _fetch(factory, kind="mcp", ext_id=src_ext)
    assert rest_row.source_missing_at is None
    assert await _audit_count(factory, ext_id=src_ext, event="source_restored") == 1

    # reconciled_missing standalone → 软删 + disposition=soft_deleted
    # F4：镜像真实 two-strike 前置——record_reconciled_missing 仅在 source_missing_at
    # 已置位（首轮 mark_source_missing 宽限后）才被 reconciler 调用（_reconcile_sources）。
    # R2b#4：source_missing_at 必须**早于**宽限期（此处 1h ago，远超 15min）才软删——
    # 时间宽限门见下方 test_reconciled_missing_time_grace。
    std_ext = _eid("mcp-recmiss")
    await _insert_ext(factory, kind="mcp", ext_id=std_ext, surface_hash="s", row_revision=1,
                      source_missing_at=datetime.now(timezone.utc) - timedelta(hours=1))
    await write.record_reconciled_missing("mcp", std_ext)
    std_row = await _fetch(factory, kind="mcp", ext_id=std_ext)
    assert std_row.deleted_at is not None and std_row.row_revision == 2
    assert (await _audit_details(factory, ext_id=std_ext, event="reconciled_missing")
            == {"disposition": "soft_deleted"})

    # reconciled_missing plugin → 保持行 + disposition=retained_plugin
    plug_ext = _eid("plugin-recmiss")
    await _insert_ext(factory, kind="plugin", ext_id=plug_ext, artifact_hash="pa", row_revision=1)
    await write.record_reconciled_missing("plugin", plug_ext)
    plug_row = await _fetch(factory, kind="plugin", ext_id=plug_ext)
    assert plug_row.deleted_at is None and plug_row.row_revision == 1   # 父行保持
    assert (await _audit_details(factory, ext_id=plug_ext, event="reconciled_missing")
            == {"disposition": "retained_plugin"})


# 10b -------------------------------------------------------------------------
async def test_reinstall_clears_source_missing_then_reconcile_noop(factory):
    """F4 数据损坏竞态：source_missing_at 置位的行被并发重装（record_install 清标记）后，
    一条陈旧 record_reconciled_missing 必须 no-op——不软删已恢复的 live 行、不写 audit。
    fix(a)=record_install update 分支清 source_missing_at；fix(b)=reconcile 软删加
    `source_missing_at IS NOT NULL` WHERE 守卫 + rowcount==0 跳 audit。"""
    write = DbExtensionRegistryWritePort(factory)
    ext_id = _eid("mcp-f4race")
    # 首装建行 + 置 source_missing_at（模拟首轮宽限后的源缺失标记，row_revision=1）
    await _insert_ext(factory, kind="mcp", ext_id=ext_id, surface_hash="s",
                      config_fingerprint="cfg", row_revision=1,
                      source_missing_at=datetime.now(timezone.utc))
    # 并发重装（update-only 分支）→ fix(a)：清 source_missing_at + CAS bump→2
    await write.record_install("mcp", ext_id, _install_ctx(
        surface_hash="s2", config_fingerprint="cfg2"))
    reinstalled = await _fetch(factory, kind="mcp", ext_id=ext_id)
    assert reinstalled.source_missing_at is None            # fix(a)：missing 标记已清
    assert reinstalled.deleted_at is None
    rev_after_install = reinstalled.row_revision
    # 陈旧 reconcile-missing 落地 → fix(b) WHERE 守卫 miss → no-op（不删、不 bump、不写 audit）
    await write.record_reconciled_missing("mcp", ext_id)
    final = await _fetch(factory, kind="mcp", ext_id=ext_id)
    assert final.deleted_at is None                         # live 行保留（未被陈旧 delete 覆盖）
    assert final.row_revision == rev_after_install          # 无 bump（UPDATE 未命中）
    assert await _audit_count(factory, ext_id=ext_id, event="reconciled_missing") == 0


# 11 --------------------------------------------------------------------------
async def test_governance_counters_semantics(factory):
    """R32#9/R43#3/#4：五行（active-unpinned/active-pin_stale/active-missing-obs/
    quarantined/disabled）→ 三计数增量（unpinned=2 含 stale；missing=1；quarantined=1；
    disabled 不进前两者）。全表聚合用 before/after 增量绕开累积污染。"""
    read = DbExtensionRegistryReadPort(factory)
    before = await read.governance_counters()

    # 1) active-unpinned：pin=NULL，observed 齐
    await _insert_ext(
        factory, kind="skill", ext_id=_eid("skill-unpinned"), artifact_hash=None,
        observed_artifact_hash="a", observed_hash_schema_version=HASH_SCHEMA_VERSION)
    # 2) active-pin_stale：pin 有但 pin 版本过期
    await _insert_ext(
        factory, kind="skill", ext_id=_eid("skill-pinstale"), artifact_hash="a",
        hash_schema_version=0,
        observed_artifact_hash="a", observed_hash_schema_version=HASH_SCHEMA_VERSION)
    # 3) active-missing-observation：pin 齐但 observed=NULL
    await _insert_ext(
        factory, kind="skill", ext_id=_eid("skill-missobs"), artifact_hash="a",
        observed_artifact_hash=None, observed_hash_schema_version=HASH_SCHEMA_VERSION)
    # 4) quarantined
    await _insert_ext(
        factory, kind="skill", ext_id=_eid("skill-quar"), status="quarantined",
        quarantine_reason="admin_manual", artifact_hash="a")
    # 5) disabled
    await _insert_ext(
        factory, kind="skill", ext_id=_eid("skill-dis"), status="disabled", artifact_hash="a")

    after = await read.governance_counters()
    assert after.unpinned_count - before.unpinned_count == 2          # unpinned + pin_stale
    assert after.missing_observation_count - before.missing_observation_count == 1
    assert after.quarantined_count - before.quarantined_count == 1


# 12 --------------------------------------------------------------------------
async def test_record_delete_cas_retry(factory, monkeypatch):
    """F4：软删带 row_revision CAS——陈旧快照落败前两次 → 第三次成功；全部耗尽 →
    RevisionConflictError（镜像 record_install 的读-CAS-重试(3)）。"""
    write = DbExtensionRegistryWritePort(factory)
    real_load = reg_mod._load_live

    # (a) 前两次陈旧 → 第三次成功
    ext_a = _eid("skill-delcasretry")
    await _insert_ext(factory, kind="skill", ext_id=ext_a, artifact_hash="h0", row_revision=5)
    counter = {"n": 0}

    async def _stale_twice(session, kind, ext_id):
        counter["n"] += 1
        row = await real_load(session, kind, ext_id)
        if row is None:
            return None
        if counter["n"] <= 2:
            return SimpleNamespace(id=row.id, row_revision=row.row_revision - 1)
        return row

    monkeypatch.setattr(reg_mod, "_load_live", _stale_twice)
    await write.record_delete(
        "skill", ext_a,
        uninstall_context=UninstallContext(correlation_id=uuid.uuid4(), actor_user_id="admin-1"))
    row_a = await _fetch(factory, kind="skill", ext_id=ext_a)
    assert counter["n"] == 3
    assert row_a.deleted_at is not None and row_a.row_revision == 6
    assert await _audit_count(factory, ext_id=ext_a, event="uninstalled") == 1

    # (b) 永远陈旧 → 耗尽 3 次 → RevisionConflictError，行零写零 audit
    ext_b = _eid("skill-delcasexhaust")
    await _insert_ext(factory, kind="skill", ext_id=ext_b, artifact_hash="h0", row_revision=5)

    async def _always_stale(session, kind, ext_id):
        row = await real_load(session, kind, ext_id)
        if row is None:
            return None
        return SimpleNamespace(id=row.id, row_revision=row.row_revision - 1)

    monkeypatch.setattr(reg_mod, "_load_live", _always_stale)
    with pytest.raises(RevisionConflictError):
        await write.record_delete(
            "skill", ext_b,
            uninstall_context=UninstallContext(correlation_id=uuid.uuid4(), actor_user_id="admin-1"))
    row_b = await _fetch(factory, kind="skill", ext_id=ext_b)
    assert row_b.deleted_at is None and row_b.row_revision == 5
    assert await _audit_count(factory, ext_id=ext_b, event="uninstalled") == 0


# 13 --------------------------------------------------------------------------
async def test_admin_audit_before_snapshot_pre_update(factory):
    """F5：active→quarantined 的 audit before.status 必须是 'active'（真实 ORM update()
    synchronize_session 会同步 in-memory row → before 必须在 UPDATE 前捕获）。"""
    write = DbExtensionRegistryWritePort(factory)
    ext_id = _eid("skill-auditbefore")
    await _insert_ext(factory, kind="skill", ext_id=ext_id, artifact_hash="a", row_revision=0)
    await write.quarantine("skill", ext_id, expected_row_revision=0, actor_user_id="admin-1")
    audit = next(r for r in await _audit_rows(factory, ext_id=ext_id) if r.event == "quarantined")
    assert audit.before == {"status": "active"}
    assert audit.after["status"] == "quarantined"


# 14 --------------------------------------------------------------------------
async def test_list_audit_valid_cursor_paginates_invalid_400(factory):
    """R2b#2：valid 复合游标照常翻页；畸形 ``?cursor=garbage`` → BadRequestError（400，
    非全局 500）。decode 早于 DB session，畸形串零 DB 触碰。"""
    from app.application.errors.exceptions import BadRequestError

    read = DbExtensionRegistryReadPort(factory)
    # 三条同 kind/ext_id 的 audit，created_at 递增（确定翻页序）
    ext_id = _eid("skill-audit-page")
    base = datetime(2026, 7, 11, tzinfo=timezone.utc)
    async with factory() as session:
        async with session.begin():
            for i in range(3):
                session.add(ExtensionAuditLogModel(
                    kind="skill", ext_id=ext_id, event="installed",
                    created_at=base + timedelta(seconds=i)))

    # 第一页 limit=2 → 2 条 + next_cursor 非空
    page1 = await read.list_audit(ext_id=ext_id, limit=2)
    assert len(page1.entries) == 2
    assert page1.next_cursor is not None
    # valid 游标翻第二页 → 剩 1 条、末页 next_cursor=None
    page2 = await read.list_audit(ext_id=ext_id, cursor=page1.next_cursor, limit=2)
    assert len(page2.entries) == 1
    assert page2.next_cursor is None
    # 两页 id 不重叠（真翻页而非重取）
    ids1 = {e.id for e in page1.entries}
    assert page2.entries[0].id not in ids1

    # 畸形游标 → BadRequestError（400）
    with pytest.raises(BadRequestError):
        await read.list_audit(ext_id=ext_id, cursor="garbage")


# 15 --------------------------------------------------------------------------
async def test_reconciled_missing_time_grace(factory):
    """R2b#4：record_reconciled_missing 的时间宽限门——source_missing_at **在宽限期内**
    （just now）→ **不软删**（多 pod 两撑压缩防护）；**早于**宽限期（远超 15min）→ 软删。
    保留 #4-R1：source_missing_at NULL 行同样不删（isnot(None) 守卫）。"""
    write = DbExtensionRegistryWritePort(factory)

    # (a) 宽限期内（just now）→ 不软删、不 bump、零 audit
    recent = _eid("mcp-grace-recent")
    await _insert_ext(factory, kind="mcp", ext_id=recent, surface_hash="s", row_revision=1,
                      source_missing_at=datetime.now(timezone.utc))
    await write.record_reconciled_missing("mcp", recent)
    recent_row = await _fetch(factory, kind="mcp", ext_id=recent)
    assert recent_row.deleted_at is None                # 宽限内不删
    assert recent_row.row_revision == 1                 # 无 bump
    assert recent_row.source_missing_at is not None     # 标记保留（待下轮过宽限）
    assert await _audit_count(factory, ext_id=recent, event="reconciled_missing") == 0

    # (b) 早于宽限期（1h ago）→ 软删 + disposition=soft_deleted
    old = _eid("mcp-grace-old")
    await _insert_ext(factory, kind="mcp", ext_id=old, surface_hash="s", row_revision=1,
                      source_missing_at=datetime.now(timezone.utc) - timedelta(hours=1))
    await write.record_reconciled_missing("mcp", old)
    old_row = await _fetch(factory, kind="mcp", ext_id=old)
    assert old_row.deleted_at is not None               # 过宽限 → 软删
    assert old_row.row_revision == 2
    assert (await _audit_details(factory, ext_id=old, event="reconciled_missing")
            == {"disposition": "soft_deleted"})

    # (c) source_missing_at NULL（从未标记）→ 不删（#4-R1 isnot(None) 守卫，与宽限门正交）
    never = _eid("mcp-grace-never")
    await _insert_ext(factory, kind="mcp", ext_id=never, surface_hash="s", row_revision=1)
    await write.record_reconciled_missing("mcp", never)
    never_row = await _fetch(factory, kind="mcp", ext_id=never)
    assert never_row.deleted_at is None
    assert await _audit_count(factory, ext_id=never, event="reconciled_missing") == 0
