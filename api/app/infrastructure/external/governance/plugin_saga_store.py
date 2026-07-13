"""D1a §8.3/§3.4 Plugin install saga 持久化 store（骨架/步进/发布/补偿事务族）。

step schema 唯一权威（§3.4 R7#1 write-ahead intent）：``make_step`` + ``STEP_STATES`` +
``TARGET_TYPES`` + 纯函数 ``build_saga_steps``。DB 事务族：

- ``create_install_skeleton`` —— **单一事务** NOT NULL FK 插入序（R10#1）：①plugin 父行
  status=disabled → ②全部成员行 active + membership → ③operation in_progress + steps 预写
  全量 planned → ④audit ``plugin_expand_started``；
- ``mark_step`` —— 单独 UPDATE（attempting 写前置标 R48#2；done 恒不追加新项 R9#1）；
- ``complete_install`` —— 发布事务：plugin 行 disabled→active + op completed +
  ``plugin_expand_completed``；
- ``compensate`` —— 骨架回收事务：plugin 父行/剩余成员行软删 + membership 物理删 +
  op compensated + ``plugin_expand_compensated``（+ 复验/恢复碰撞 ``recovery_collision``）；
- ``fail`` —— op failed（父行保持 disabled=天然阻断，§3.4 failed 语义）；
- ``load_operation`` / ``record_content_collision_audit``。

分层：infrastructure——允许 SQLAlchemy/session（同 ``db_extension_registry``）。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.errors.exceptions import NotFoundError
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    OperationPendingError,
    RevisionConflictError,
)
from app.domain.services.extension_hashing import entry_content_hash
from app.infrastructure.external.governance.audit import insert_audit
from app.infrastructure.models.extension_governance import (
    ExtensionInstallOperationModel,
    ExtensionModel,
    PluginMembershipModel,
)

class PluginActivationConflictError(RuntimeError):
    """**硬化 #3**：§8.3-4 发布事务的 CAS 激活（``disabled`` + ``deleted_at IS NULL`` → ``active``）
    未命中骨架行（rowcount≠1）——骨架被带外改写/软删/并发发布，父行不在预期态。绝不把非预期态
    的父行翻 ``active``（INV-D1-8 唯一激活点）；抛出令 ``install()`` 外层 ``except`` 转补偿
    （本事务已回滚激活 UPDATE，op 保持 in_progress、父行保持 disabled）。"""


# R2b#1（b）：startup 崩溃恢复的**孤儿判定**staleness 阈值。`load_inflight_operations`
# 原返回全部 `state='in_progress'` 操作——多 pod 下 Pod B 会把 Pod A **正在运行**的 install
# 当孤儿补偿/回滚。只有 `updated_at` 陈旧（远超 saga 预期最大时长）的 op 才是真孤儿：
# 活 pod 的 saga 每步 `mark_step` 都刷 `updated_at`，故进行中的 op 恒「新鲜」不被采纳。
# spec §3.6 明示 D1a 假设单实例、多实例退化 best-effort——本阈值 + compensate/fail 的终态
# CAS 是多 pod 的最小防护；健壮的 lease/heartbeat 归 flip 前置（见 §3.6 部署诚实声明）。
# 值=生成器无版本、spec-silent 下选定的稳妥默认（saga 含每成员 ~20s probe + 内容写，
# 几分钟内收敛；30min 远超之，既不误采纳活 saga，又能最终清理真孤儿）。
INFLIGHT_STALE_THRESHOLD = timedelta(minutes=30)


# §3.4 唯一 schema（R7#1）——step 状态四值 + target 六类。
STEP_STATES = ("planned", "attempting", "done", "collided")
TARGET_TYPES = (
    "skill_dir", "mcp_config", "a2a_config", "plugin_bundle", "registry_row", "membership",
)


def make_step(seq: int, step: str, target_type: str, key: str, expected_hash: str | None) -> dict:
    """§3.4 唯一 step 工厂——所有权证明的数据载体（expected_hash 内容 target 非 NULL）。"""
    return {
        "seq": seq,
        "step": step,
        "target": {"type": target_type, "key": key},
        "state": "planned",
        "expected_hash": expected_hash,
        "done_at": None,
    }


def bundle_target_key(plugin_ext_id: str, version: str) -> str:
    """plugin_bundle target key = ``{plugin_ext_id}/{version}``（自持版本，崩溃恢复无需 ctx）。"""
    return f"{plugin_ext_id}/{version}"


def build_saga_steps(ctx: Any) -> list[dict]:
    """§8.3-2 ③：从不可变 ``PluginInstallContext`` 预写全量 planned steps（R6#7/R8#1）。

    仅覆盖**外部写目标**（R7#1：skill 目录 / config 条目 / plugin bundle 目录）——
    registry_row / membership 属 DB 事务（骨架/补偿内联，非 step 追踪）。expected_hash 每
    target（R49#2/R50#1）：plugin bundle/skill=快照 ``compute_content_hash`` 值；mcp/a2a=
    ``entry_content_hash(entry_dump)``（第三种 hash，含 secrets 值，§3.4）。seq 1=父 bundle
    首段（§8.3-3），成员 2..N+1——补偿逆序=成员先删、bundle 后删（LIFO）。
    """
    steps: list[dict] = [
        make_step(
            1, "write_plugin_bundle", "plugin_bundle",
            bundle_target_key(ctx.plugin_ext_id, ctx.version), ctx.parent_bundle_hash,
        )
    ]
    seq = 1
    for member in ctx.members:
        seq += 1
        if member.kind == "skill":
            steps.append(make_step(
                seq, "write_skill", "skill_dir", member.ext_id, member.observed_artifact_hash))
        elif member.kind == "mcp":
            steps.append(make_step(
                seq, "write_mcp", "mcp_config", member.ext_id,
                entry_content_hash(member.entry_dump)))
        elif member.kind == "a2a":
            steps.append(make_step(
                seq, "write_a2a", "a2a_config", member.ext_id,
                entry_content_hash(member.entry_dump)))
    return steps


# §8.3-6 uninstall step target 类型（成员 kind → target type，与 install 侧同源）。
_UNINSTALL_TARGET_TYPE_FOR_KIND = {"skill": "skill_dir", "mcp": "mcp_config", "a2a": "a2a_config"}


def build_uninstall_steps(members: list[tuple[str, str]]) -> list[dict]:
    """§8.3-6：从 managed_by_plugin=true 成员 ``(kind, ext_id)`` 预写全量 planned 成员删除步。

    仅覆盖成员内容 target（skill 目录 / mcp,a2a config 条目）——plugin bundle 目录 / membership /
    父行由 ``finalize_uninstall`` 最终事务处理，不在 step 追踪（uninstall 是 **forward** 语义，
    孤儿恢复标 failed 不逆补偿，故成员步 ``expected_hash=None``——删除幂等 missing_ok，无所有权三分）。
    seq 1..N 为成员序（forward 删除，非 LIFO）。
    """
    steps: list[dict] = []
    for seq, (kind, ext_id) in enumerate(members, start=1):
        steps.append(make_step(
            seq, f"delete_{kind}", _UNINSTALL_TARGET_TYPE_FOR_KIND[kind], ext_id, None))
    return steps


@dataclass(frozen=True)
class SkeletonResult:
    """``create_install_skeleton`` 产物——operation id + plugin 父行 UUID。"""
    operation_id: uuid.UUID
    plugin_row_id: uuid.UUID


@dataclass(frozen=True)
class OperationSnapshot:
    """store frozen 读模型（R3#3）——``begin``/``load_operation`` 统一返回该类型。"""
    id: uuid.UUID
    operation_type: str
    plugin_extension_id: uuid.UUID
    plugin_ext_id: str | None
    initiated_by: str
    state: str
    steps: list[dict[str, Any]]
    error: str | None


# ---- T24 §9.2 GET /v2/plugins 读模型（rows join membership join 最新 operation）------
@dataclass(frozen=True)
class PluginLastOperationRow:
    """最新 saga operation 摘要（updated_at DESC 取一）。"""
    type: str
    state: str
    error: str | None
    updated_at: str | None


@dataclass(frozen=True)
class PluginMemberDetailRow:
    """membership join 子 registry 行摘要（声明读者——不重算/不观测）。"""
    declared_component_id: str
    kind: str
    ext_id: str
    expected_hash: str | None
    installed_version: str | None
    managed_by_plugin: bool
    status: str
    scan_verdict: str | None
    scan_report: dict[str, Any] | None


@dataclass(frozen=True)
class PluginDetailRow:
    """live plugin 父行 + 成员 + 最新 operation（interfaces 层映射 ``PluginDetail`` DTO）。

    ``name`` 无 registry 列（manifest name 不落库）→ 恒 None。"""
    ext_id: str
    name: str | None
    version: str | None
    status: str
    artifact_hash: str | None
    row_revision: int
    last_operation: PluginLastOperationRow | None
    members: list[PluginMemberDetailRow] = field(default_factory=list)


class PluginSagaStore:
    """§8.3/§3.4 saga 事务族（session 自管：每方法 ``session.begin()`` 提交）。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ---------------------------------------------------------------- skeleton --
    async def create_install_skeleton(self, ctx: Any) -> SkeletonResult:
        """§8.3-2 单一事务——插入序满足 NOT NULL FK（R10#1）；父行 disabled 骨架。"""
        prov = ctx.parent_provenance
        async with self._session_factory() as session:
            async with session.begin():
                # ① plugin 父行 status=disabled（发布前 parent_blocked，INV-D1-8）
                plugin_row = ExtensionModel(
                    kind="plugin", ext_id=ctx.plugin_ext_id, status="disabled",
                    trust_origin=prov.trust_origin, source_type=prov.source_type,
                    source_ref=prov.source_ref, version=ctx.version,
                    hash_schema_version=HASH_SCHEMA_VERSION,
                    scan_verdict=ctx.aggregate_verdict, installed_by=ctx.initiated_by)
                session.add(plugin_row)
                await session.flush()

                # ② 全部成员行 active + membership（pins 在内容写 record_install update-only）
                for member in ctx.members:
                    m_prov = member.provenance
                    member_row = ExtensionModel(
                        kind=member.kind, ext_id=member.ext_id, status="active",
                        trust_origin=m_prov.trust_origin, source_type=m_prov.source_type,
                        source_ref=m_prov.source_ref, version=m_prov.version,
                        hash_schema_version=HASH_SCHEMA_VERSION,
                        scan_verdict=member.scan.verdict,
                        scan_report=member.scan.model_dump())
                    session.add(member_row)
                    await session.flush()
                    session.add(PluginMembershipModel(
                        plugin_id=plugin_row.id, child_extension_id=member_row.id,
                        declared_component_id=member.declared_component_id,
                        expected_hash=member.expected_hash_declared,
                        managed_by_plugin=True, installed_version=m_prov.version))

                # ③ operation in_progress + steps 预写全量 planned
                operation = ExtensionInstallOperationModel(
                    operation_type="plugin_install", plugin_extension_id=plugin_row.id,
                    initiated_by=ctx.initiated_by, state="in_progress",
                    steps=build_saga_steps(ctx))
                session.add(operation)
                await session.flush()

                # ④ audit plugin_expand_started（correlation=operation.id 贯穿全序）
                await insert_audit(
                    session, kind="plugin", ext_id=ctx.plugin_ext_id,
                    extension_id=plugin_row.id, event="plugin_expand_started",
                    actor_user_id=ctx.initiated_by, correlation_id=operation.id)
                return SkeletonResult(operation_id=operation.id, plugin_row_id=plugin_row.id)

    # -------------------------------------------------------------- mark_step --
    async def mark_step(self, operation_id: uuid.UUID, seq: int, state: str) -> None:
        """单独 UPDATE 既有 step 的 state（R48#2 写前置标 attempting；R9#1 done 不追加）。"""
        async with self._session_factory() as session:
            async with session.begin():
                op = await session.get(ExtensionInstallOperationModel, operation_id)
                if op is None:
                    return
                steps = [dict(s) for s in op.steps]
                for s in steps:
                    if s["seq"] == seq:
                        s["state"] = state
                        if state == "done":
                            s["done_at"] = datetime.now(timezone.utc).isoformat()
                        break
                await session.execute(
                    update(ExtensionInstallOperationModel)
                    .where(ExtensionInstallOperationModel.id == operation_id)
                    .values(steps=steps, updated_at=datetime.now(timezone.utc)))

    # ---------------------------------------------------------------- publish --
    async def complete_install(self, operation_id: uuid.UUID) -> None:
        """§8.3-4 发布事务：plugin 行 disabled→active + op completed + ``plugin_expand_completed``。

        **硬化 #3**：激活 UPDATE 是 INV-D1-8 唯一激活点。CAS 追加 ``deleted_at IS NULL``（软删行
        不得复活）并校验 rowcount——仅 rowcount==1（骨架确在预期 ``disabled``-live 态）才翻 op
        ``completed``；否则抛 ``PluginActivationConflictError`` 令上层 ``except`` 补偿（本事务回滚，
        父行保持 disabled、op 保持 in_progress）。"""
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            async with session.begin():
                op = await session.get(ExtensionInstallOperationModel, operation_id)
                if op is None:
                    return
                plugin_row = await session.get(ExtensionModel, op.plugin_extension_id)
                result = await session.execute(
                    update(ExtensionModel)
                    .where(ExtensionModel.id == op.plugin_extension_id,
                           ExtensionModel.status == "disabled",
                           ExtensionModel.deleted_at.is_(None))
                    .values(status="active",
                            row_revision=ExtensionModel.row_revision + 1,
                            updated_at=now))
                if result.rowcount != 1:                            # 骨架非预期态 → 不发布，转补偿
                    raise PluginActivationConflictError(
                        f"plugin activation CAS 未命中 disabled-live 骨架行"
                        f"（operation={operation_id}, rowcount={result.rowcount}）")
                await session.execute(
                    update(ExtensionInstallOperationModel)
                    .where(ExtensionInstallOperationModel.id == operation_id)
                    .values(state="completed", updated_at=now))
                await insert_audit(
                    session, kind="plugin",
                    ext_id=plugin_row.ext_id if plugin_row else None,
                    extension_id=op.plugin_extension_id, event="plugin_expand_completed",
                    actor_user_id=op.initiated_by, correlation_id=operation_id)

    # -------------------------------------------------------------- compensate --
    async def compensate(self, operation_id: uuid.UUID, *, details: dict[str, Any]) -> None:
        """§8.3-5 骨架回收：plugin 父行/剩余 live 成员行软删 + membership 物理删 +
        op compensated + ``plugin_expand_compensated``（+ ``recovery_collision`` 若 collided）。

        已写成员行由 _rollback_steps 的 delete hooks 软删（+``uninstalled``）——本事务只
        兜剩余（planned/collided 成员行 + 父行=无 hook）。details 键限 allowlist
        {failed_step, compensated_targets}（§3.5）。"""
        now = datetime.now(timezone.utc)
        collided_targets = details.get("collided_targets") or []
        async with self._session_factory() as session:
            async with session.begin():
                op = await session.get(ExtensionInstallOperationModel, operation_id)
                if op is None:
                    return
                # R2b#1（a）：终态 CAS 前置守卫——只补偿仍 `in_progress` 的 operation。多 pod
                # 下 Pod B 的 startup closure 持 load_inflight 快照调本方法时，Pod A 可能已
                # `complete_install`（op→completed、父行→active）。无守卫则本方法会**拆掉一个
                # 已成功的安装**（软删成员+父行、删 membership）并把 completed 覆盖成 compensated。
                # CAS 置 compensated WHERE state='in_progress'：miss（rowcount 0）→ 已终态 → 直接
                # 跳过全部 teardown（幂等空操作），杜绝覆盖。CAS 先行使 teardown 只在赢家事务内发生。
                cas = await session.execute(
                    update(ExtensionInstallOperationModel)
                    .where(ExtensionInstallOperationModel.id == operation_id,
                           ExtensionInstallOperationModel.state == "in_progress")
                    .values(state="compensated", updated_at=now)
                    .returning(ExtensionInstallOperationModel.id))
                if cas.scalar_one_or_none() is None:
                    return  # 已被并发 complete_install / 另一 pod 补偿 → 不 teardown 不覆盖
                plugin_row = await session.get(ExtensionModel, op.plugin_extension_id)
                # 剩余 live 成员行软删（已 hook-软删的过滤 deleted_at IS NULL 天然跳过）
                member_ids = (await session.execute(
                    select(PluginMembershipModel.child_extension_id)
                    .where(PluginMembershipModel.plugin_id == op.plugin_extension_id))
                ).scalars().all()
                if member_ids:
                    await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id.in_(member_ids),
                               ExtensionModel.deleted_at.is_(None))
                        .values(deleted_at=now, row_revision=ExtensionModel.row_revision + 1))
                # membership 物理删（R13#7 无软删列）
                await session.execute(
                    delete(PluginMembershipModel)
                    .where(PluginMembershipModel.plugin_id == op.plugin_extension_id))
                # plugin 父行软删（释放 (plugin, ext_id) 身份——补偿后可重装，R1#8）
                if plugin_row is not None and plugin_row.deleted_at is None:
                    await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id == op.plugin_extension_id)
                        .values(deleted_at=now, row_revision=ExtensionModel.row_revision + 1))
                # op state → compensated 已由上方终态 CAS 前置守卫原子写入（R2b#1a）——此处不再重复
                # 恢复期碰撞 per-operation 恰一条（§3.5 R52#6）——details.collided_targets 全列
                if collided_targets:
                    await insert_audit(
                        session, kind="plugin",
                        ext_id=plugin_row.ext_id if plugin_row else None,
                        extension_id=op.plugin_extension_id, event="install_rejected",
                        actor_user_id=op.initiated_by, correlation_id=operation_id,
                        details={"stage": "recovery_collision",
                                 "collided_targets": list(collided_targets)})
                await insert_audit(
                    session, kind="plugin",
                    ext_id=plugin_row.ext_id if plugin_row else None,
                    extension_id=op.plugin_extension_id, event="plugin_expand_compensated",
                    actor_user_id=op.initiated_by, correlation_id=operation_id,
                    details={k: details[k] for k in ("failed_step", "compensated_targets")
                             if k in details})

    # -------------------------------------------------------------------- fail --
    async def fail(self, operation_id: uuid.UUID, *, error: str) -> None:
        """补偿自身失败/config 不可读 → op failed（父行**保持** disabled=阻断态，§3.4）。

        R2b#1（a）：终态 CAS `WHERE state='in_progress'`——多 pod startup closure 对已被
        另一 pod `complete_install` 的 op 调 fail 时，miss（rowcount 0）→ 幂等空操作，绝不把
        completed/compensated 覆盖成 failed。fail 无 teardown，故仅需 WHERE 守卫（无早退分支）。"""
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(ExtensionInstallOperationModel)
                    .where(ExtensionInstallOperationModel.id == operation_id,
                           ExtensionInstallOperationModel.state == "in_progress")
                    .values(state="failed", error=error, updated_at=datetime.now(timezone.utc)))

    # ---------------------------------------------------------- load_operation --
    async def load_operation(self, operation_id: uuid.UUID) -> OperationSnapshot | None:
        async with self._session_factory() as session:
            op = await session.get(ExtensionInstallOperationModel, operation_id)
            if op is None:
                return None
            plugin_row = await session.get(ExtensionModel, op.plugin_extension_id)
            return OperationSnapshot(
                id=op.id, operation_type=op.operation_type,
                plugin_extension_id=op.plugin_extension_id,
                plugin_ext_id=plugin_row.ext_id if plugin_row else None,
                initiated_by=op.initiated_by, state=op.state,
                steps=[dict(s) for s in op.steps], error=op.error)

    # -------------------------------------------------- list_plugin_details --
    async def list_plugin_details(self) -> list[PluginDetailRow]:
        """§9.2 GET /v2/plugins：live plugin 父行 join membership join 最新 operation。

        纯读（无锁、无事务写）——声明读者：membership 子行的 registry 状态/pin/scan 逐字投影，
        不重算/不观测。plugin 父行按 ``ext_id`` 稳定序；成员按 ``(kind, ext_id)`` 稳定序；
        最新 operation 取 ``updated_at DESC`` 一条（无 operation → None）。软删父行不列。"""
        async with self._session_factory() as session:
            plugins = (await session.execute(
                select(ExtensionModel)
                .where(ExtensionModel.kind == "plugin",
                       ExtensionModel.deleted_at.is_(None))
                .order_by(ExtensionModel.ext_id))).scalars().all()
            details: list[PluginDetailRow] = []
            for p in plugins:
                member_rows = (await session.execute(
                    select(ExtensionModel, PluginMembershipModel)
                    .join(PluginMembershipModel,
                          PluginMembershipModel.child_extension_id == ExtensionModel.id)
                    .where(PluginMembershipModel.plugin_id == p.id)
                    .order_by(ExtensionModel.kind, ExtensionModel.ext_id))).all()
                members = [
                    PluginMemberDetailRow(
                        declared_component_id=mem.declared_component_id,
                        kind=child.kind, ext_id=child.ext_id,
                        expected_hash=mem.expected_hash,
                        installed_version=mem.installed_version,
                        managed_by_plugin=mem.managed_by_plugin,
                        status=child.status, scan_verdict=child.scan_verdict,
                        scan_report=child.scan_report)
                    for child, mem in member_rows
                ]
                latest = (await session.execute(
                    select(ExtensionInstallOperationModel)
                    .where(ExtensionInstallOperationModel.plugin_extension_id == p.id)
                    .order_by(ExtensionInstallOperationModel.updated_at.desc())
                    .limit(1))).scalar_one_or_none()
                last_op = (
                    PluginLastOperationRow(
                        type=latest.operation_type, state=latest.state,
                        error=latest.error,
                        updated_at=latest.updated_at.isoformat() if latest.updated_at else None)
                    if latest is not None else None)
                details.append(PluginDetailRow(
                    ext_id=p.ext_id, name=None, version=p.version, status=p.status,
                    artifact_hash=p.artifact_hash, row_revision=p.row_revision,
                    last_operation=last_op, members=members))
            return details

    # ============================================================ §8.3-6 uninstall ==
    async def begin_uninstall(
        self, plugin_ext_id: str, *, initiated_by: str, expected_row_revision: int | None,
    ) -> OperationSnapshot:
        """§8.3-6 首步单一 DB 事务（R6#4——消除"operation 已建而父级未阻断"崩溃窗口）。分支：

        - 既有 ``failed plugin_uninstall`` → 原子 CAS ``state='in_progress' WHERE state='failed'``
          **复活原 operation**（保 id/steps，correlation 连续；清 error）；CAS 输→并发已复活→409。
        - 既有 ``in_progress`` operation（任意类型）→ 409（部分唯一索引语义，主动检出给清晰错误）。
        - 新建路（``failed plugin_install`` 不阻断——§3.4 唯一例外）：父行 CAS-bump row_revision
          （expected_row_revision 守卫；active→disabled + audit ``disabled`` / 已阻断态仅 bump，R47#2）
          + ``plugin_uninstall`` operation in_progress（成员步预写 planned）。
        """
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            async with session.begin():
                parent = await self._load_live_plugin(session, plugin_ext_id)
                if parent is None:
                    raise NotFoundError(f"plugin {plugin_ext_id} not found")
                existing = (await session.execute(
                    select(ExtensionInstallOperationModel).where(
                        ExtensionInstallOperationModel.plugin_extension_id == parent.id,
                        ExtensionInstallOperationModel.state.in_(("in_progress", "failed"))))
                ).scalars().all()
                if any(o.state == "in_progress" for o in existing):
                    raise OperationPendingError(
                        f"plugin {plugin_ext_id} 存在 in_progress operation（等待完成或重试）")
                failed_uninstall = next(
                    (o for o in existing
                     if o.state == "failed" and o.operation_type == "plugin_uninstall"), None)
                if failed_uninstall is not None:
                    result = await session.execute(
                        update(ExtensionInstallOperationModel)
                        .where(ExtensionInstallOperationModel.id == failed_uninstall.id,
                               ExtensionInstallOperationModel.state == "failed")
                        .values(state="in_progress", error=None, updated_at=now)
                        .returning(ExtensionInstallOperationModel.id))
                    if result.scalar_one_or_none() is None:
                        raise OperationPendingError(
                            f"plugin {plugin_ext_id} failed uninstall 已被并发复活（重试）")
                    return OperationSnapshot(
                        id=failed_uninstall.id, operation_type="plugin_uninstall",
                        plugin_extension_id=parent.id, plugin_ext_id=plugin_ext_id,
                        initiated_by=failed_uninstall.initiated_by, state="in_progress",
                        steps=[dict(s) for s in failed_uninstall.steps], error=None)
                # 新建路：父行 CAS-bump（active→disabled / 已阻断态保持 status 仍 bump）
                was_active = parent.status == "active"
                new_status = "disabled" if was_active else parent.status
                cas = await session.execute(
                    update(ExtensionModel)
                    .where(ExtensionModel.id == parent.id,
                           ExtensionModel.row_revision == expected_row_revision,
                           ExtensionModel.deleted_at.is_(None))
                    .values(status=new_status,
                            row_revision=ExtensionModel.row_revision + 1, updated_at=now)
                    .returning(ExtensionModel.row_revision))
                if cas.scalar_one_or_none() is None:
                    raise RevisionConflictError(
                        f"plugin {plugin_ext_id} row_revision mismatch（并发治理动作）")
                # 先建 operation 行（拿到 id）——父行 CAS 已在其前，"父行先阻断"崩溃一致性不变；
                # 全程同一 session.begin() 事务（parent block + steps + op create + disabled audit
                # 原子提交，R6#4 首步单一事务合同）。
                members = await self._load_managed_members(session, parent.id)
                operation = ExtensionInstallOperationModel(
                    operation_type="plugin_uninstall", plugin_extension_id=parent.id,
                    initiated_by=initiated_by, state="in_progress",
                    steps=build_uninstall_steps(members))
                session.add(operation)
                await session.flush()
                # disabled 迁移由本 uninstall op 因果驱动 → 必须 correlation=operation.id（否则
                # correlation_id NULL 使该 audit 脱离 op 事件流，_events_for(op.id) 检不到）。
                if was_active:
                    await insert_audit(
                        session, kind="plugin", ext_id=plugin_ext_id, extension_id=parent.id,
                        event="disabled", actor_user_id=initiated_by,
                        before={"status": "active"}, after={"status": "disabled"},
                        correlation_id=operation.id)
                return OperationSnapshot(
                    id=operation.id, operation_type="plugin_uninstall",
                    plugin_extension_id=parent.id, plugin_ext_id=plugin_ext_id,
                    initiated_by=initiated_by, state="in_progress",
                    steps=[dict(s) for s in operation.steps], error=None)

    async def finalize_uninstall(self, operation_id: uuid.UUID) -> None:
        """§8.3-6 最终步单事务：membership 物理删 + 软删 plugin 父行（+ ``uninstalled`` audit）+
        op ``completed`` + 旧 ``failed plugin_install`` → ``compensated``（error 列追加
        ``cleaned_up_via=<op_id>``，零 schema 变更 R53#2——否则父行已软删而 failed 仍要求阻断，
        D1-8 违反且 §14 全终态不可达）。plugin bundle 目录（§8.2）随本步删除（事务提交后 best-effort）。"""
        now = datetime.now(timezone.utc)
        bundle_key: str | None = None
        async with self._session_factory() as session:
            async with session.begin():
                op = await session.get(ExtensionInstallOperationModel, operation_id)
                if op is None:
                    return
                parent = await session.get(ExtensionModel, op.plugin_extension_id)
                # membership 物理删（R13#7 无软删列）
                await session.execute(
                    delete(PluginMembershipModel)
                    .where(PluginMembershipModel.plugin_id == op.plugin_extension_id))
                # 软删 plugin 父行（释放身份，补偿后可重装）+ uninstalled audit
                if parent is not None and parent.deleted_at is None:
                    await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id == op.plugin_extension_id)
                        .values(deleted_at=now, row_revision=ExtensionModel.row_revision + 1))
                    await insert_audit(
                        session, kind="plugin", ext_id=parent.ext_id,
                        extension_id=op.plugin_extension_id, event="uninstalled",
                        actor_user_id=op.initiated_by, correlation_id=operation_id)
                    bundle_key = bundle_target_key(parent.ext_id, parent.version or "")
                await session.execute(
                    update(ExtensionInstallOperationModel)
                    .where(ExtensionInstallOperationModel.id == operation_id)
                    .values(state="completed", updated_at=now))
                # 旧 failed install → compensated（error 追加 cleaned_up_via，同事务闭合）
                failed_installs = (await session.execute(
                    select(ExtensionInstallOperationModel).where(
                        ExtensionInstallOperationModel.plugin_extension_id == op.plugin_extension_id,
                        ExtensionInstallOperationModel.operation_type == "plugin_install",
                        ExtensionInstallOperationModel.state == "failed"))).scalars().all()
                for fi in failed_installs:
                    await session.execute(
                        update(ExtensionInstallOperationModel)
                        .where(ExtensionInstallOperationModel.id == fi.id)
                        .values(state="compensated",
                                error=(fi.error or "") + f" cleaned_up_via={operation_id}",
                                updated_at=now))
        # plugin bundle 目录删（事务外 best-effort——DB 终态已提交，fs 残留不阻塞语义）
        # Known limitation（Minor #3）：本删除在 COMMIT 之后；若此处失败/进程在 COMMIT 与删除间崩溃，
        # bundle 目录泄漏。因 _recheck_collisions_dual_source 把 _bundle_dir_exists 当碰撞源，泄漏目录
        # 会令同 {plugin_ext_id}/{version} 的**重装在 409 窗口**内被拒，直到 staging/bundle 被清理。
        if bundle_key is not None:
            from app.application.services.plugin_install_service import _remove_bundle_dir
            _remove_bundle_dir(bundle_key)

    async def load_inflight_operations(self) -> list[OperationSnapshot]:
        """§3.4 startup 恢复：``state='in_progress'`` **且陈旧**的孤儿操作快照（closure 按 type 分流）。

        R2b#1（b）：加 `updated_at < now - INFLIGHT_STALE_THRESHOLD` staleness 门——原「返回全部
        in_progress」会让多 pod 的 startup closure 把另一 pod **正在运行**的 install 当孤儿补偿/
        回滚（甚至删掉其正写入的内容）。活 pod 的 saga 每步 `mark_step` 刷 `updated_at`，故进行中的
        op 恒新鲜、被本门排除；只有久无进展的 op 才是真孤儿被采纳。**单实例快速重启的代价**：崩溃后
        新 boot 时孤儿的 `updated_at` 仍「新鲜」，须待其老化过阈值（或下次 boot）才被收尾——期间父行
        保持 disabled=阻断，无准入，安全（healthy 的多 pod lease/ownership 归 flip 前置，§3.6）。"""
        cutoff = datetime.now(timezone.utc) - INFLIGHT_STALE_THRESHOLD
        async with self._session_factory() as session:
            ops = (await session.execute(
                select(ExtensionInstallOperationModel).where(
                    ExtensionInstallOperationModel.state == "in_progress",
                    ExtensionInstallOperationModel.updated_at < cutoff))).scalars().all()
            snapshots: list[OperationSnapshot] = []
            for op in ops:
                parent = await session.get(ExtensionModel, op.plugin_extension_id)
                snapshots.append(OperationSnapshot(
                    id=op.id, operation_type=op.operation_type,
                    plugin_extension_id=op.plugin_extension_id,
                    plugin_ext_id=parent.ext_id if parent else None,
                    initiated_by=op.initiated_by, state=op.state,
                    steps=[dict(s) for s in op.steps], error=op.error))
            return snapshots

    async def sweep_terminal_staging(self) -> None:
        """§8.3-3/R50#3 断言③：终态 op（completed/compensated/failed）的 staging 目录整删；
        非终态（in_progress）保留。DB 查全部 op 状态 → 委托 application 层共享 fs 逻辑
        （``_sweep_staging_dirs`` 单源——staging 根常量归属 plugin_install_service，避免双份）。"""
        async with self._session_factory() as session:
            rows = (await session.execute(
                select(ExtensionInstallOperationModel.id,
                       ExtensionInstallOperationModel.state))).all()
        op_states = {r.id: r.state for r in rows}
        from app.application.services.plugin_install_service import _sweep_staging_dirs
        _sweep_staging_dirs(op_states)

    async def _load_live_plugin(self, session: AsyncSession, plugin_ext_id: str):
        result = await session.execute(
            select(ExtensionModel).where(
                ExtensionModel.kind == "plugin", ExtensionModel.ext_id == plugin_ext_id,
                ExtensionModel.deleted_at.is_(None)))
        return result.scalar_one_or_none()

    async def _load_managed_members(
        self, session: AsyncSession, plugin_row_id: uuid.UUID,
    ) -> list[tuple[str, str]]:
        """§8.4：仅 ``managed_by_plugin=true`` 成员（kind, ext_id），确定序（kind, ext_id）。"""
        result = await session.execute(
            select(ExtensionModel.kind, ExtensionModel.ext_id)
            .join(PluginMembershipModel,
                  PluginMembershipModel.child_extension_id == ExtensionModel.id)
            .where(PluginMembershipModel.plugin_id == plugin_row_id,
                   PluginMembershipModel.managed_by_plugin.is_(True))
            .order_by(ExtensionModel.kind, ExtensionModel.ext_id))
        return [(r.kind, r.ext_id) for r in result.all()]

    # ------------------------------------------------ collision/reject audits --
    async def record_content_collision_audit(
        self,
        operation_id: uuid.UUID,
        *,
        stage: str,
        member: str | None = None,
        target_key: str | None = None,
        collided_targets: list[str] | None = None,
    ) -> None:
        """§3.5：install_rejected（extension_id=父行非 NULL、correlation=op.id）——
        stage ∈ {content_write_collision, publish_reverify, recovery_collision}（allowlist）。"""
        async with self._session_factory() as session:
            async with session.begin():
                op = await session.get(ExtensionInstallOperationModel, operation_id)
                if op is None:
                    return
                plugin_row = await session.get(ExtensionModel, op.plugin_extension_id)
                details: dict[str, Any] = {"stage": stage}
                if member is not None:
                    details["member"] = member
                if collided_targets is not None:
                    details["collided_targets"] = list(collided_targets)
                await insert_audit(
                    session, kind="plugin",
                    ext_id=plugin_row.ext_id if plugin_row else None,
                    extension_id=op.plugin_extension_id, event="install_rejected",
                    actor_user_id=op.initiated_by, correlation_id=operation_id,
                    details=details)


class DbPluginRejectAuditSink:
    """T24：``PluginRejectAuditSink`` 生产实现（关闭 T21 注入的 Protocol 占位）。

    真实 install（非 dry_run）被 preflight / 锁内重检拒绝时，写**自持** ``install_rejected``
    audit——``extension_id`` NULL + ``ext_id`` NULL（身份解析前的唯一合法 NULL 场景，R13#5/
    audit ``insert_audit`` 断言允许）。details 经 ``sanitize_details`` allowlist
    （{stage, member, category, collided_targets, source_ref}）+ ``source_ref``
    强制 canonicalize（INV-D1-7）。``source_type`` 不在 allowlist（``kind='plugin'`` 已载体）
    → 天然 drop。自管 session（每次 ``session.begin()`` 提交）；``dry_run`` 时服务侧不调用。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record_install_rejected(
        self, *, source_type: str, source_ref: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        if source_ref is not None:
            merged.setdefault("source_ref", source_ref)   # allowlist + canonicalize 在 sink 内
        async with self._session_factory() as session:
            async with session.begin():
                await insert_audit(
                    session, kind="plugin", ext_id=None, extension_id=None,
                    event="install_rejected", details=merged)
