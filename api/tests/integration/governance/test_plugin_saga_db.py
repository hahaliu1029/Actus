"""T22 `[CI-only integration]` — Plugin install saga store 真 PG 事务族 + §3.5 审计序列 +
INV-D1-8 性质断言。

**集成未本地跑，CI 验证**（本地 `uv run pytest <file> --collect-only`）。covers §8.3/§3.4：
骨架单事务 NOT NULL FK 插入序（父 disabled / 成员 active / membership / op in_progress / steps
全 planned / plugin_expand_started）→ mark_step 更新不追加 → complete_install 发布（父 active /
op completed / plugin_expand_completed）→ compensate 骨架回收（父/成员软删 + membership 物理删 /
op compensated / plugin_expand_compensated）→ fail 保持父 disabled → record_content_collision_audit
三 stage。INV-D1-8：operation ∈ {in_progress, failed} ⇒ 父行阻断（disabled）。

Run:
    cd api && uv run pytest tests/integration/governance/test_plugin_saga_db.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.services.plugin_install_service import (
    MemberPlan,
    PluginInstallContext,
    Provenance,
    _rollback_steps,
)
from app.domain.models.extension_governance import GovernanceScanSummary
from app.domain.services.extension_hashing import entry_content_hash
from app.infrastructure.external.governance.plugin_saga_store import (
    OperationSnapshot,
    PluginActivationConflictError,
    PluginSagaStore,
    build_saga_steps,
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
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def _pid() -> str:
    return f"org.test.pack-{uuid.uuid4().hex[:10]}"


def _scan() -> GovernanceScanSummary:
    return GovernanceScanSummary(verdict="safe", finding_count=0, findings=[])


def _mcp_member(ext_id: str) -> MemberPlan:
    entry_dump = {"transport": "streamable_http", "url": "https://safe.test/mcp"}
    return MemberPlan(
        kind="mcp", declared_component_id="search", ext_id=ext_id, scan=_scan(),
        observed_surface_hash="sha256:surf", observed_config_fingerprint="sha256:fp",
        observed_artifact_hash=None, expected_hash_declared=None,
        provenance=Provenance(source_type="plugin", source_ref="pluginid", version="1.0.0"),
        probe_failed=False, acknowledged=False, forced=False,
        payload=None, entry_dump=entry_dump)


def _ctx(plugin_ext_id: str, member_ext_id: str) -> PluginInstallContext:
    return PluginInstallContext(
        plugin_ext_id=plugin_ext_id, name="Pack", version="1.0.0",
        parent_bundle_hash="sha256:bundle",
        parent_provenance=Provenance(source_type="local", source_ref="local:/x", version="1.0.0"),
        members=(_mcp_member(member_ext_id),),
        preallocated_a2a_ids={}, aggregate_verdict="safe",
        forced=False, acknowledged=False, initiated_by="admin-1",
        parent_bundle_files={})


async def _events_for(factory, op_id: uuid.UUID) -> list[str]:
    async with factory() as session:
        rows = (await session.execute(
            select(ExtensionAuditLogModel)
            .where(ExtensionAuditLogModel.correlation_id == op_id)
            .order_by(ExtensionAuditLogModel.created_at, ExtensionAuditLogModel.id))).scalars().all()
    return [r.event for r in rows]


async def _parent(factory, plugin_ext_id: str) -> ExtensionModel | None:
    async with factory() as session:
        return (await session.execute(
            select(ExtensionModel).where(
                ExtensionModel.kind == "plugin", ExtensionModel.ext_id == plugin_ext_id))
        ).scalar_one_or_none()


# ==================================================================== tests =====
async def test_skeleton_first_transaction(factory):
    """§8.3-2：单事务——父 disabled / 成员 active / membership / op in_progress / steps 全 planned
    / plugin_expand_started。NOT NULL FK 插入序（无异常即证 R10#1）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"srv-{uuid.uuid4().hex[:8]}"
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, member_ext_id))

    parent = await _parent(factory, plugin_ext_id)
    assert parent is not None and parent.status == "disabled"
    async with factory() as session:
        member = (await session.execute(
            select(ExtensionModel).where(ExtensionModel.kind == "mcp",
                                         ExtensionModel.ext_id == member_ext_id))).scalar_one()
        assert member.status == "active"
        m_count = (await session.execute(
            select(func.count()).select_from(PluginMembershipModel)
            .where(PluginMembershipModel.plugin_id == skeleton.plugin_row_id))).scalar_one()
        assert m_count == 1
        op = await session.get(ExtensionInstallOperationModel, skeleton.operation_id)
        assert op.state == "in_progress"
        assert [s["state"] for s in op.steps] == ["planned", "planned"]
        assert op.steps[1]["expected_hash"] == entry_content_hash(
            {"transport": "streamable_http", "url": "https://safe.test/mcp"})
    assert await _events_for(factory, skeleton.operation_id) == ["plugin_expand_started"]


async def test_mark_step_updates_not_appends(factory):
    """R9#1/R48#2：mark_step 按 seq 更新既有项（attempting→done），绝不追加。"""
    store = PluginSagaStore(factory)
    skeleton = await store.create_install_skeleton(_ctx(_pid(), f"s-{uuid.uuid4().hex[:8]}"))
    await store.mark_step(skeleton.operation_id, 1, "attempting")
    await store.mark_step(skeleton.operation_id, 1, "done")
    op = await store.load_operation(skeleton.operation_id)
    assert len(op.steps) == 2
    assert op.steps[0]["state"] == "done" and op.steps[0]["done_at"] is not None


async def test_complete_install_publishes(factory):
    """§8.3-4：complete_install → 父行 active + op completed + plugin_expand_completed。"""
    store = PluginSagaStore(factory)
    plugin_ext_id = _pid()
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, f"s-{uuid.uuid4().hex[:8]}"))
    await store.complete_install(skeleton.operation_id)
    parent = await _parent(factory, plugin_ext_id)
    assert parent.status == "active" and parent.deleted_at is None
    op = await store.load_operation(skeleton.operation_id)
    assert op.state == "completed"
    assert await _events_for(factory, skeleton.operation_id) == [
        "plugin_expand_started", "plugin_expand_completed"]


async def test_complete_install_cas_rejects_soft_deleted_skeleton(factory):
    """硬化 #3(a)：complete_install CAS 追加 ``deleted_at IS NULL``——骨架被带外软删 → rowcount 0
    → ``PluginActivationConflictError``；op 不 completed（事务回滚）、软删父行**不复活**（不翻
    active，INV-D1-8）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id = _pid()
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, f"s-{uuid.uuid4().hex[:8]}"))
    async with factory() as session:                              # 带外软删父行（骨架非预期 live 态）
        async with session.begin():
            await session.execute(
                update(ExtensionModel)
                .where(ExtensionModel.id == skeleton.plugin_row_id)
                .values(deleted_at=datetime.now(timezone.utc)))
    with pytest.raises(PluginActivationConflictError):
        await store.complete_install(skeleton.operation_id)
    op = await store.load_operation(skeleton.operation_id)
    assert op.state == "in_progress"                             # 未 completed
    parent = await _parent(factory, plugin_ext_id)
    assert parent.status == "disabled"                           # 未翻 active（软删行不复活）
    # 唯一激活点未发射发布 audit（回滚）
    assert await _events_for(factory, skeleton.operation_id) == ["plugin_expand_started"]


async def test_complete_install_cas_rejects_non_disabled_skeleton(factory):
    """硬化 #3(b)：complete_install rowcount 门——骨架非 ``disabled``（带外已 active，如并发发布）
    → rowcount 0 → ``PluginActivationConflictError``；op 不 completed（不静默把非预期态翻 active）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id = _pid()
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, f"s-{uuid.uuid4().hex[:8]}"))
    async with factory() as session:                              # 带外把父行改成 active
        async with session.begin():
            await session.execute(
                update(ExtensionModel)
                .where(ExtensionModel.id == skeleton.plugin_row_id)
                .values(status="active"))
    with pytest.raises(PluginActivationConflictError):
        await store.complete_install(skeleton.operation_id)
    op = await store.load_operation(skeleton.operation_id)
    assert op.state == "in_progress"                             # rowcount 门拦截，op 未 completed


async def test_compensate_recycles_skeleton(factory):
    """§8.3-5：compensate → 父/成员软删 + membership 物理删 + op compensated +
    plugin_expand_compensated。INV-D1-8：compensate 前 in_progress ⇒ 父行 disabled（阻断）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"s-{uuid.uuid4().hex[:8]}"
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, member_ext_id))
    # INV-D1-8：in_progress 期父行阻断
    assert (await _parent(factory, plugin_ext_id)).status == "disabled"
    await store.compensate(skeleton.operation_id, details={
        "failed_step": 2, "compensated_targets": ["mcp_config:" + member_ext_id],
        "collided_targets": []})
    parent = await _parent(factory, plugin_ext_id)
    assert parent.deleted_at is not None                         # 父行软删（身份释放）
    async with factory() as session:
        member = (await session.execute(
            select(ExtensionModel).where(ExtensionModel.ext_id == member_ext_id))).scalar_one()
        assert member.deleted_at is not None                    # 成员软删
        mem_count = (await session.execute(
            select(func.count()).select_from(PluginMembershipModel)
            .where(PluginMembershipModel.plugin_id == skeleton.plugin_row_id))).scalar_one()
        assert mem_count == 0                                    # membership 物理删
    op = await store.load_operation(skeleton.operation_id)
    assert op.state == "compensated"
    assert await _events_for(factory, skeleton.operation_id) == [
        "plugin_expand_started", "plugin_expand_compensated"]


async def test_fail_keeps_parent_disabled(factory):
    """§3.4：fail → op failed + 父行**保持** disabled（阻断态，Admin 处置）。INV-D1-8。"""
    store = PluginSagaStore(factory)
    plugin_ext_id = _pid()
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, f"s-{uuid.uuid4().hex[:8]}"))
    await store.fail(skeleton.operation_id, error="boom")
    op = await store.load_operation(skeleton.operation_id)
    assert op.state == "failed" and op.error == "boom"
    parent = await _parent(factory, plugin_ext_id)
    assert parent.status == "disabled" and parent.deleted_at is None


async def test_record_content_collision_audit_stages(factory):
    """§3.5：record_content_collision_audit → install_rejected（extension_id=父行、stage 允许集）。"""
    store = PluginSagaStore(factory)
    plugin_ext_id = _pid()
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, f"s-{uuid.uuid4().hex[:8]}"))
    await store.record_content_collision_audit(
        skeleton.operation_id, stage="publish_reverify", member="search")
    async with factory() as session:
        rejected = (await session.execute(
            select(ExtensionAuditLogModel).where(
                ExtensionAuditLogModel.event == "install_rejected",
                ExtensionAuditLogModel.correlation_id == skeleton.operation_id))).scalar_one()
    assert rejected.extension_id == skeleton.plugin_row_id       # 非 NULL（与 preflight 自持区分）
    assert rejected.details["stage"] == "publish_reverify"


async def test_rollback_shared_kernel_compensates_done(factory):
    """_rollback_steps 共享内核（运行期与恢复同一算法）：done 成员经 delete hook + compensate。
    用 fake app_config/skill/write_port（内容删除面）验补偿转 compensated。"""
    store = PluginSagaStore(factory)
    plugin_ext_id, member_ext_id = _pid(), f"s-{uuid.uuid4().hex[:8]}"
    skeleton = await store.create_install_skeleton(_ctx(plugin_ext_id, member_ext_id))
    await store.mark_step(skeleton.operation_id, 1, "done")
    await store.mark_step(skeleton.operation_id, 2, "done")
    op = await store.load_operation(skeleton.operation_id)

    class _Cfg:
        def __init__(self):
            self.deleted = []

        async def delete_mcp_server(self, name, *, uninstall_context=None, missing_ok=False):
            self.deleted.append(name)

        async def delete_a2a_server(self, aid, *, uninstall_context=None, missing_ok=False):
            self.deleted.append(aid)

    cfg = _Cfg()
    await _rollback_steps(op, {"mcp": {}, "a2a": {}}, store,
                          app_config_service=cfg, skill_service=None, write_port=None)
    assert member_ext_id in cfg.deleted                         # done 成员经 delete hook
    assert (await store.load_operation(skeleton.operation_id)).state == "compensated"


def test_operation_snapshot_shape():
    """OperationSnapshot frozen 读模型契约（纯断言——无 DB）。"""
    snap = OperationSnapshot(
        id=uuid.uuid4(), operation_type="plugin_install",
        plugin_extension_id=uuid.uuid4(), plugin_ext_id="p", initiated_by="a",
        state="in_progress", steps=[], error=None)
    assert snap.state == "in_progress" and snap.plugin_ext_id == "p"


def test_build_saga_steps_pure():
    """build_saga_steps 纯函数（无 DB）：bundle seq1 + 成员，全 planned。"""
    steps = build_saga_steps(_ctx("p", "m"))
    assert [s["target"]["type"] for s in steps] == ["plugin_bundle", "mcp_config"]
    assert all(s["state"] == "planned" for s in steps)
