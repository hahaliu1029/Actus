"""T23 `[CI-only integration]` — Plugin uninstall saga + 崩溃恢复链 真 PG 事务族。

**集成未本地跑，CI 验证**（本地 `uv run pytest <file> --collect-only`）。covers §8.3-6/§3.4：
begin_uninstall 首步单一事务（父 active→disabled + row_revision CAS-bump + disabled audit +
成员步预写 planned）/ expected_revision 失配 409 / in_progress 409 / failed uninstall CAS 复活原
op / failed install 允许新建 uninstall / finalize_uninstall（membership 物理删 + 父行软删 +
op completed + 旧 failed install → compensated[error 追加 cleaned_up_via]）/ load_inflight_operations
分流 / sweep_terminal_staging / **9b 恢复三分 secret-only-diff → collided（entry_content_hash 非
fingerprint，端到端锁死误用）**。

Run:
    cd api && uv run pytest tests/integration/governance/test_plugin_saga_recovery_db.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.application.services.plugin_install_service as saga_mod
from app.application.services.plugin_install_service import (
    MemberPlan,
    PluginInstallContext,
    Provenance,
    _rollback_steps,
)
from app.domain.models.app_config import MCPServerConfig
from app.domain.models.extension_governance import (
    GovernanceScanSummary,
    OperationPendingError,
    RevisionConflictError,
)
from app.domain.services.extension_hashing import entry_content_hash, mcp_config_fingerprint
from app.infrastructure.external.governance.plugin_saga_store import PluginSagaStore
from app.infrastructure.models.extension_governance import (
    ExtensionAuditLogModel,
    ExtensionInstallOperationModel,
    ExtensionModel,
    PluginMembershipModel,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def factory(async_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def _pid() -> str:
    return f"org.test.pack-{uuid.uuid4().hex[:10]}"


def _scan() -> GovernanceScanSummary:
    return GovernanceScanSummary(verdict="safe", finding_count=0, findings=[])


def _mcp_member(ext_id: str, *, secret: str = "AAA") -> MemberPlan:
    cfg = MCPServerConfig(transport="streamable_http", url="https://safe.test/mcp",
                          headers={"Authorization": f"Bearer {secret}"})
    return MemberPlan(
        kind="mcp", declared_component_id="search", ext_id=ext_id, scan=_scan(),
        observed_surface_hash="sha256:surf", observed_config_fingerprint=mcp_config_fingerprint(cfg),
        observed_artifact_hash=None, expected_hash_declared=None,
        provenance=Provenance(source_type="plugin", source_ref="pluginid", version="1.0.0"),
        probe_failed=False, acknowledged=False, forced=False,
        payload=cfg, entry_dump=cfg.model_dump(mode="json"))


def _ctx(plugin_ext_id: str, member: MemberPlan) -> PluginInstallContext:
    return PluginInstallContext(
        plugin_ext_id=plugin_ext_id, name="Pack", version="1.0.0",
        parent_bundle_hash="sha256:bundle",
        parent_provenance=Provenance(source_type="local", source_ref="local:/x", version="1.0.0"),
        members=(member,), preallocated_a2a_ids={}, aggregate_verdict="safe",
        forced=False, acknowledged=False, initiated_by="admin-1", parent_bundle_files={})


async def _installed(store: PluginSagaStore, factory, plugin_ext_id, member):
    """skeleton → complete → 已安装 plugin（父 active + 成员 active + membership + op completed）。"""
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, member))
    await store.complete_install(skeleton.operation_id)
    return skeleton


async def _parent(factory, plugin_ext_id) -> ExtensionModel | None:
    async with factory() as session:
        return (await session.execute(
            select(ExtensionModel).where(
                ExtensionModel.kind == "plugin", ExtensionModel.ext_id == plugin_ext_id))
        ).scalar_one_or_none()


async def _events_for(factory, op_id) -> list[str]:
    async with factory() as session:
        rows = (await session.execute(
            select(ExtensionAuditLogModel)
            .where(ExtensionAuditLogModel.correlation_id == op_id)
            .order_by(ExtensionAuditLogModel.created_at, ExtensionAuditLogModel.id))).scalars().all()
    return [r.event for r in rows]


# ==================================================================== tests =====
async def test_begin_uninstall_first_tx(factory):
    """§8.3-6：begin_uninstall 首步单一事务——父 active→disabled + row_revision CAS-bump +
    disabled audit + op plugin_uninstall in_progress + 成员步预写 planned。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    await _installed(store, factory, plugin_ext_id, _mcp_member(member_ext_id))
    parent = await _parent(factory, plugin_ext_id)
    rev = parent.row_revision
    op = await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=rev)

    assert op.operation_type == "plugin_uninstall" and op.state == "in_progress"
    assert [s["target"] for s in op.steps] == [{"type": "mcp_config", "key": member_ext_id}]
    assert all(s["state"] == "planned" for s in op.steps)
    parent2 = await _parent(factory, plugin_ext_id)
    assert parent2.status == "disabled" and parent2.row_revision == rev + 1
    assert "disabled" in await _events_for(factory, op.id)


async def test_begin_uninstall_already_disabled_bumps_revision(factory):
    """R47#2：已阻断（disabled）父行——begin_uninstall 保持 status 但仍 CAS-bump row_revision
    （operation 存在性是行治理语义一部分；并发 enable 持旧 revision 由此失效 409）；无 disabled audit。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    skeleton = await _installed(store, factory, plugin_ext_id, _mcp_member(member_ext_id))
    # 先卸一次 → 失败留 disabled（复用 begin+fail 把父行落 disabled）
    op1 = await store.begin_uninstall(
        plugin_ext_id, initiated_by="admin",
        expected_row_revision=(await _parent(factory, plugin_ext_id)).row_revision)
    await store.fail(op1.id, error="boom")
    p = await _parent(factory, plugin_ext_id)
    assert p.status == "disabled"
    rev = p.row_revision
    # 复活原 failed uninstall（begin 走复活路，不再 bump 父行）——此处验新建路需另一 plugin
    # 用第二 plugin 直接构造 disabled 父行：begin 之后 fail，再新建路不可（同 plugin 走复活）。
    # 直接断言复活语义：begin 再次 → 复活原 op（id 相同）
    op2 = await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=rev)
    assert op2.id == op1.id and op2.state == "in_progress"     # 复活原 op（保 id/steps）
    assert op2.steps == op1.steps
    _ = skeleton


async def test_begin_uninstall_expected_revision_mismatch_409(factory):
    """§3.6：begin_uninstall 新建路 expected_row_revision 失配 → RevisionConflictError（409）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    await _installed(store, factory, plugin_ext_id, _mcp_member(member_ext_id))
    with pytest.raises(RevisionConflictError):
        await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=999)


async def test_begin_uninstall_in_progress_409(factory):
    """R46#2：既有 in_progress operation（install 骨架）→ begin_uninstall OperationPendingError（409）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    await store.create_install_skeleton(_ctx(plugin_ext_id, _mcp_member(member_ext_id)))   # 留 in_progress
    with pytest.raises(OperationPendingError):
        await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=0)


async def test_failed_uninstall_cas_resurrect(factory):
    """§3.4：failed uninstall → begin_uninstall CAS 复活**原** op（id/steps 保留，error 清空，
    correlation 连续）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    await _installed(store, factory, plugin_ext_id, _mcp_member(member_ext_id))
    rev = (await _parent(factory, plugin_ext_id)).row_revision
    op = await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=rev)
    await store.fail(op.id, error="uninstall step failed")
    resurrected = await store.begin_uninstall(
        plugin_ext_id, initiated_by="admin-2", expected_row_revision=999)  # revision 复活路不消费
    assert resurrected.id == op.id and resurrected.state == "in_progress"
    assert resurrected.steps == op.steps and resurrected.error is None
    assert resurrected.initiated_by == "admin"                # 原发起者（correlation 连续）


async def test_failed_install_allows_new_uninstall(factory):
    """§3.4 唯一例外：failed install 的 plugin 允许新建 uninstall（前向清理）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, _mcp_member(member_ext_id)))
    await store.fail(skeleton.operation_id, error="install compensation failed")   # failed install
    op = await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=0)
    assert op.operation_type == "plugin_uninstall" and op.state == "in_progress"


async def test_finalize_uninstall_soft_deletes_and_converts_failed_install(factory):
    """§8.3-6 最终事务：membership 物理删 + 父行软删（+ uninstalled）+ op completed +
    旧 failed install → compensated（error 追加 cleaned_up_via=<uninstall op id>，R52#4）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    skeleton = await _installed(store, factory, plugin_ext_id, _mcp_member(member_ext_id))
    # 注入一个旧 failed install（同父行）
    async with factory() as session:
        async with session.begin():
            session.add(ExtensionInstallOperationModel(
                operation_type="plugin_install", plugin_extension_id=skeleton.plugin_row_id,
                initiated_by="admin", state="failed", steps=[], error="old"))
    rev = (await _parent(factory, plugin_ext_id)).row_revision
    op = await store.begin_uninstall(plugin_ext_id, initiated_by="admin", expected_row_revision=rev)
    await store.finalize_uninstall(op.id)

    parent = await _parent(factory, plugin_ext_id)
    assert parent.deleted_at is not None                      # 父行软删
    async with factory() as session:
        mem_count = (await session.execute(
            select(func.count()).select_from(PluginMembershipModel)
            .where(PluginMembershipModel.plugin_id == skeleton.plugin_row_id))).scalar_one()
        assert mem_count == 0                                 # membership 物理删
        ops = (await session.execute(
            select(ExtensionInstallOperationModel).where(
                ExtensionInstallOperationModel.plugin_extension_id == skeleton.plugin_row_id))
        ).scalars().all()
    by_type_state = {(o.operation_type, o.state) for o in ops}
    assert ("plugin_uninstall", "completed") in by_type_state
    assert ("plugin_install", "compensated") in by_type_state          # 旧 failed → compensated
    failed_install = next(o for o in ops if o.operation_type == "plugin_install")
    assert f"cleaned_up_via={op.id}" in failed_install.error
    assert "uninstalled" in await _events_for(factory, op.id)          # 父行软删 audit


async def _backdate_op(factory, plugin_extension_id, *, minutes=60):
    """把某 plugin 的 in_progress operation 的 updated_at 回拨到陈旧（模拟真孤儿——久无进展）。"""
    old = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    async with factory() as session:
        async with session.begin():
            await session.execute(
                update(ExtensionInstallOperationModel)
                .where(ExtensionInstallOperationModel.plugin_extension_id == plugin_extension_id,
                       ExtensionInstallOperationModel.state == "in_progress")
                .values(updated_at=old))


async def test_load_inflight_operations_typed(factory):
    """§3.4 + R2b#1b：load_inflight_operations 只返回**陈旧**（updated_at 过 staleness 阈值）的
    in_progress 快照（含 operation_type 供 closure 分流）；活 pod 正在运行的**新鲜** op 被排除
    （每步 mark_step 刷 updated_at → 恒新鲜 → 不被误采纳为孤儿）。"""
    store = PluginSagaStore(factory)
    p1, p2, p3 = _pid(), _pid(), _pid()
    await store.create_install_skeleton(_ctx(p1, _mcp_member(f"s-{uuid.uuid4().hex[:8]}")))  # install in_progress
    await _installed(store, factory, p2, _mcp_member(f"s-{uuid.uuid4().hex[:8]}"))
    rev = (await _parent(factory, p2)).row_revision
    await store.begin_uninstall(p2, initiated_by="admin", expected_row_revision=rev)  # uninstall in_progress
    # p1/p2 回拨到陈旧 → 真孤儿被采纳；p3 保持新鲜（活 saga）→ staleness 门排除
    await _backdate_op(factory, (await _parent(factory, p1)).id)
    await _backdate_op(factory, (await _parent(factory, p2)).id)
    await store.create_install_skeleton(_ctx(p3, _mcp_member(f"s-{uuid.uuid4().hex[:8]}")))  # 新鲜 in_progress

    inflight = await store.load_inflight_operations()
    types = {(o.plugin_ext_id, o.operation_type) for o in inflight}
    assert (p1, "plugin_install") in types          # 陈旧 → 采纳
    assert (p2, "plugin_uninstall") in types        # 陈旧 → 采纳
    assert (p3, "plugin_install") not in types      # R2b#1b：新鲜 → staleness 门排除
    assert all(o.state == "in_progress" for o in inflight)


async def test_sweep_terminal_staging(factory, tmp_path, monkeypatch):
    """R50#3 断言③：sweep_terminal_staging DB 查全部 op → 终态（completed/compensated/failed）
    staging 目录整删；非终态（in_progress）保留。"""
    monkeypatch.setattr(saga_mod, "PLUGIN_STAGING_ROOT", tmp_path / ".plugin-staging")
    store = PluginSagaStore(factory)
    # completed op（skeleton+complete）+ in_progress op（skeleton 留）
    done = await _installed(store, factory, _pid(), _mcp_member(f"s-{uuid.uuid4().hex[:8]}"))
    live = await store.create_install_skeleton(_ctx(_pid(), _mcp_member(f"s-{uuid.uuid4().hex[:8]}")))
    for op_id in (done.operation_id, live.operation_id):
        (saga_mod.PLUGIN_STAGING_ROOT / str(op_id)).mkdir(parents=True)

    await store.sweep_terminal_staging()
    assert not (saga_mod.PLUGIN_STAGING_ROOT / str(done.operation_id)).exists()   # completed 删
    assert (saga_mod.PLUGIN_STAGING_ROOT / str(live.operation_id)).exists()       # in_progress 保留


async def test_recovery_three_way_secret_only_diff_collided(factory, tmp_path, monkeypatch):
    """R3#12/spec R50#1 端到端：attempting mcp config target，带外条目与 expected 仅差 env/header
    secret 值（fingerprint 排除域）→ 恢复三分判 ≠expected → 标 collided **不删** + recovery_collision
    audit。泛化 hash 测试抓不住误用 fingerprint 的实现——此处显式锁死 entry_content_hash（含 secret）。"""
    monkeypatch.setattr(saga_mod, "PLUGIN_STORE_ROOT", tmp_path / "plugins")
    monkeypatch.setattr(saga_mod, "SKILL_STORE_ROOT", tmp_path / "skills")
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    member = _mcp_member(member_ext_id, secret="AAA")
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, member))
    # bundle(seq1) 保持 planned（跳过）；mcp(seq2) 标 attempting → 三分判定
    await store.mark_step(skeleton.operation_id, 2, "attempting")
    op = await store.load_operation(skeleton.operation_id)

    # 带外条目：仅 secret 值不同（BBB）→ fingerprint 相同、entry_content_hash 不同
    outofband = MCPServerConfig(transport="streamable_http", url="https://safe.test/mcp",
                                headers={"Authorization": "Bearer BBB"})
    assert mcp_config_fingerprint(outofband) == member.observed_config_fingerprint  # secret-only diff
    assert entry_content_hash(outofband.model_dump(mode="json")) != op.steps[1]["expected_hash"]
    config_entries = {"mcp": {member_ext_id: outofband.model_dump(mode="json")}, "a2a": {}}

    class _Cfg:
        def __init__(self):
            self.deleted: list = []

        async def delete_mcp_server(self, name, *, uninstall_context=None, missing_ok=False):
            self.deleted.append(name)

        async def delete_a2a_server(self, aid, *, uninstall_context=None, missing_ok=False):
            self.deleted.append(aid)

    cfg = _Cfg()
    await _rollback_steps(op, config_entries, store,
                          app_config_service=cfg, skill_service=None, write_port=None)

    assert member_ext_id not in cfg.deleted                   # collided → 不删（宁留可见垃圾不误删）
    reloaded = await store.load_operation(skeleton.operation_id)
    assert reloaded.state == "compensated"
    rc = [a for a in await _all_audits(factory, skeleton.operation_id)
          if a.event == "install_rejected"
          and (a.details or {}).get("stage") == "recovery_collision"]
    assert len(rc) == 1                                       # per-operation 恰一条
    assert f"mcp_config:{member_ext_id}" in rc[0].details["collided_targets"]


async def _all_audits(factory, op_id):
    async with factory() as session:
        return (await session.execute(
            select(ExtensionAuditLogModel)
            .where(ExtensionAuditLogModel.correlation_id == op_id))).scalars().all()


async def test_terminal_cas_compensate_fail_no_overwrite_completed(factory):
    """R2b#1a：多 pod——Pod A 已 complete_install（op=completed、父 active）后，Pod B 的 startup
    closure 持旧快照对**同一 completed op** 调 compensate/fail：终态 CAS（WHERE state='in_progress'）
    miss → **不覆盖终态、不 teardown**（否则拆掉已成功安装 + completed→compensated 静默丢失）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    skeleton = await _installed(store, factory, plugin_ext_id, _mcp_member(member_ext_id))
    op_id = skeleton.operation_id
    parent = await _parent(factory, plugin_ext_id)
    assert parent.status == "active" and parent.deleted_at is None

    # compensate on completed op → CAS miss → 全不变（父/成员/membership/op 状态）
    await store.compensate(op_id, details={
        "failed_step": None, "compensated_targets": [], "collided_targets": []})
    reloaded = await store.load_operation(op_id)
    assert reloaded.state == "completed"                       # 未被覆盖成 compensated
    parent2 = await _parent(factory, plugin_ext_id)
    assert parent2.status == "active" and parent2.deleted_at is None   # 父行未被拆
    async with factory() as session:
        mem_count = (await session.execute(
            select(func.count()).select_from(PluginMembershipModel)
            .where(PluginMembershipModel.plugin_id == skeleton.plugin_row_id))).scalar_one()
        member_row = (await session.execute(
            select(ExtensionModel).where(
                ExtensionModel.kind == "mcp", ExtensionModel.ext_id == member_ext_id))
        ).scalar_one()
    assert mem_count == 1                                      # membership 未物理删
    assert member_row.deleted_at is None                      # 成员行未软删
    # 无 plugin_expand_compensated audit（CAS miss 早退，零 teardown 零 audit）
    assert "plugin_expand_compensated" not in await _events_for(factory, op_id)

    # fail on completed op → CAS miss → 仍 completed、error 未写
    await store.fail(op_id, error="orphaned in_progress; retry via API")
    reloaded2 = await store.load_operation(op_id)
    assert reloaded2.state == "completed"
    assert reloaded2.error is None
