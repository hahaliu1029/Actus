"""D1a 治理写/读 port 实现（§2 前置表 + §3.6 CAS + §5.2 写集 + §6.1 writer 所有权）。"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.errors.exceptions import BadRequestError, NotFoundError
from app.domain.external.extension_admission import (
    AuditLogEntry,
    AuditPage,
    GovernanceCounters,
    GovernanceRowSnapshot,
    InstallContext,
    UninstallContext,
)
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    InvalidStateTransitionError,
    ManagedByPluginError,
    MissingObservationError,
    OperationPendingError,
    RevisionConflictError,
)
from app.domain.services.extension_admission_logic import (
    REQUIRED_OBSERVED_CATEGORIES,
    pin_presence,
)
from app.infrastructure.external.governance.audit import insert_audit
from app.infrastructure.models.extension_governance import (
    ExtensionAuditLogModel,
    ExtensionInstallOperationModel,
    ExtensionModel,
    PluginMembershipModel,
)

logger = logging.getLogger(__name__)

PIN_WRITE_SET: dict[str, tuple[str, ...]] = {
    "mcp": ("surface_hash", "config_fingerprint"),
    "a2a": ("surface_hash", "config_fingerprint"),
    "skill": ("artifact_hash",),
    "plugin": ("artifact_hash",),
}
OBSERVED_CLEAR_COLUMNS = (
    "observed_surface_hash", "observed_artifact_hash",
    "observed_config_fingerprint", "observed_hash_schema_version",
)
_OBSERVED_FOR_CATEGORY = {
    "surface": "observed_surface_hash",
    "artifact": "observed_artifact_hash",
    "config_fingerprint": "observed_config_fingerprint",
}
_PIN_FOR_CATEGORY = {
    "surface": "surface_hash",
    "artifact": "artifact_hash",
    "config_fingerprint": "config_fingerprint",
}
_INSTALL_CAS_RETRIES = 3

# R2b#4：source_missing 二次确认软删的**时间**宽限门（§6.3 两次独立确认语义的时间地基）。
# 单靠 `source_missing_at IS NOT NULL`（strike 计数）在多 pod 下会被压缩：Pod A startup
# 标记后释放 advisory lock，Pod B 立即抢锁并在零经过时间内软删——两撑塌成毫秒级。加此
# 时间下限使宽限与 pod 拓扑无关：仅当 `now - source_missing_at >= 本阈值` 才允许软删。
# spec §6.3 只规定「两次独立确认」未给具体时长；本值为 spec-silent 下选定的稳妥默认——
# 足以跨过 config 手编中间态（F7 竞态）与滚动部署重启周期，又不过久延迟清理。
SOURCE_MISSING_GRACE_PERIOD = timedelta(minutes=15)


class DbExtensionRegistryWritePort:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ---------- 安装/卸载 ----------

    async def record_install(self, kind: str, ext_id: str, install_context: InstallContext) -> None:
        ctx = install_context
        pins = {col: getattr(ctx, col) for col in PIN_WRITE_SET[kind]}
        now = datetime.now(timezone.utc)
        for attempt in range(_INSTALL_CAS_RETRIES):
            async with self._session_factory() as session:
                async with session.begin():
                    row = await _load_live(session, kind, ext_id)
                    if row is None:
                        model = ExtensionModel(
                            kind=kind, ext_id=ext_id, status="active",
                            trust_origin=ctx.trust_origin, source_type=ctx.source_type,
                            source_ref=ctx.source_ref, version=ctx.version,
                            hash_schema_version=ctx.hash_schema_version,
                            scan_verdict=ctx.scan.verdict if ctx.scan else None,
                            scan_report=ctx.scan.model_dump() if ctx.scan else None,
                            installed_by=ctx.actor_user_id,
                            pinned_at=now if any(pins.values()) else None,
                            pinned_by=ctx.actor_user_id if any(pins.values()) else None,
                            **pins,
                        )
                        session.add(model)
                        await session.flush()
                        await _install_audits(session, kind, ext_id, model.id, ctx)
                        return
                    # 既有行：事务内 membership 守卫（R32#4——standalone 不得接管 plugin 成员行）
                    if ctx.correlation_id is None and await _has_membership(session, row.id):
                        raise ManagedByPluginError(f"{kind}/{ext_id} is managed by a plugin")
                    # update-only：pin/scan/provenance + 清 observed 四列；恒不改 status（R44#3）；
                    # 恒 CAS-bump（R48#1，read-CAS-retry 有界 R49#3）
                    values: dict[str, Any] = dict(
                        trust_origin=ctx.trust_origin, source_type=ctx.source_type,
                        source_ref=ctx.source_ref, version=ctx.version,
                        hash_schema_version=ctx.hash_schema_version,
                        scan_verdict=ctx.scan.verdict if ctx.scan else None,
                        scan_report=ctx.scan.model_dump() if ctx.scan else None,
                        installed_by=ctx.actor_user_id,
                        pinned_at=now if any(pins.values()) else None,
                        pinned_by=ctx.actor_user_id if any(pins.values()) else None,
                        row_revision=ExtensionModel.row_revision + 1,
                        source_missing_at=None,   # F4：重装=源重现 → 清 missing 标记
                                                  # （配合 record_reconciled_missing 的 WHERE 守卫防并发误删）
                        **pins,
                        **{col: None for col in OBSERVED_CLEAR_COLUMNS},   # R46#6
                    )
                    result = await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id == row.id,
                               ExtensionModel.row_revision == row.row_revision)
                        .values(**values)
                        .returning(ExtensionModel.row_revision))
                    if result.scalar_one_or_none() is not None:
                        await _install_audits(session, kind, ext_id, row.id, ctx)
                        return
            # CAS 失败：重读重试（不对外 409——非行政端点合同，R49#3）
        raise RevisionConflictError(
            f"record_install CAS exhausted after {_INSTALL_CAS_RETRIES} retries: {kind}/{ext_id}")

    async def record_delete(self, kind: str, ext_id: str, *, uninstall_context: UninstallContext) -> None:
        # §3.6 CAS：软删必须带 row_revision 守卫（F4——否则读 rev=7 的陈旧 delete 会覆盖
        # 并发 reinstall bump 到 8 的 live 行）。镜像 record_install 的读-CAS-有界重试(3)：
        # CAS miss → 重读；行已软删/消失 → 幂等成功（当前语义）；live 行 rev 变 → 重试；耗尽 → 409。
        for _attempt in range(_INSTALL_CAS_RETRIES):
            async with self._session_factory() as session:
                async with session.begin():
                    row = await _load_live(session, kind, ext_id)
                    if row is None:
                        return  # 幂等：已软删/不存在 → 零写零 audit（成功路）
                    # R1#17 纵深第二层（主守卫在 service 预检）：standalone 路（correlation None）
                    # 不得软删 plugin 成员行——§8.4 成员单独 uninstall v1 拒绝
                    if (uninstall_context.correlation_id is None
                            and await _has_membership(session, row.id)):
                        raise ManagedByPluginError(f"{kind}/{ext_id} is managed by a plugin")
                    result = await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id == row.id,
                               ExtensionModel.row_revision == row.row_revision)
                        .values(deleted_at=datetime.now(timezone.utc),
                                row_revision=ExtensionModel.row_revision + 1)
                        .returning(ExtensionModel.row_revision))
                    if result.scalar_one_or_none() is not None:
                        await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                           event="uninstalled",
                                           actor_user_id=uninstall_context.actor_user_id,
                                           correlation_id=uninstall_context.correlation_id)
                        return
            # CAS 失败：重读重试（不对外 409——非行政端点合同，同 record_install R49#3）
        raise RevisionConflictError(
            f"record_delete CAS exhausted after {_INSTALL_CAS_RETRIES} retries: {kind}/{ext_id}")

    # ---------- reconcile 记账族（actor=NULL 系统）----------

    async def record_reconciled_seen(self, kind, ext_id, *, source_type, source_ref, version, trust_origin) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                if await _load_live(session, kind, ext_id) is not None:
                    return
                model = ExtensionModel(
                    kind=kind, ext_id=ext_id, status="active",
                    trust_origin=trust_origin, source_type=source_type,
                    source_ref=source_ref, version=version,
                    hash_schema_version=HASH_SCHEMA_VERSION)   # §3.6-pre：建行必填=当前常量
                session.add(model)
                await session.flush()
                await insert_audit(session, kind=kind, ext_id=ext_id,
                                   extension_id=model.id, event="reconciled_seen")

    async def mark_source_missing(self, kind, ext_id) -> None:
        await self._flag_source(kind, ext_id, missing=True)

    async def mark_source_restored(self, kind, ext_id) -> None:
        await self._flag_source(kind, ext_id, missing=False)

    async def _flag_source(self, kind, ext_id, *, missing: bool) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live(session, kind, ext_id)
                if row is None:
                    return
                already = row.source_missing_at is not None
                if already == missing:
                    return  # 幂等
                await session.execute(
                    update(ExtensionModel).where(ExtensionModel.id == row.id)
                    .values(source_missing_at=datetime.now(timezone.utc) if missing else None))
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="source_missing" if missing else "source_restored")

    async def record_reconciled_missing(self, kind, ext_id) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live(session, kind, ext_id)
                if row is None:
                    return
                if kind == "plugin":
                    # R32#5：plugin 父行保持（软删会孤儿化成员）——事件照写，处置差异化
                    await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                       event="reconciled_missing",
                                       details={"disposition": "retained_plugin"})
                    return
                # F4：仅当行「仍标记 missing」才软删——防启动快照与本删之间的并发
                # 重装（record_install 已清 source_missing_at）被这条陈旧 reconcile 覆盖。
                # R2b#4：再加**时间**宽限门 `source_missing_at < now - GRACE`——两次独立确认
                # （§6.3）在多 pod 下会被压缩为毫秒级（Pod A 标记后释放 advisory lock，Pod B
                # 抢锁即以「IS NOT NULL」为真立删）；时间下限使宽限与 pod 拓扑无关。
                # WHERE 守卫 miss（rowcount 0）= 行已恢复**或仍在宽限内** → no-op，连 audit 也
                # 不写（幂等；宽限内的行保留 source_missing_at，待下一轮经过宽限后才软删）。
                now = datetime.now(timezone.utc)
                result = await session.execute(
                    update(ExtensionModel)
                    .where(ExtensionModel.id == row.id,
                           ExtensionModel.source_missing_at.isnot(None),
                           ExtensionModel.source_missing_at < now - SOURCE_MISSING_GRACE_PERIOD)
                    .values(deleted_at=now,
                            row_revision=ExtensionModel.row_revision + 1)
                    .returning(ExtensionModel.row_revision))
                if result.scalar_one_or_none() is None:
                    return  # 并发重装已恢复该行 / 仍在时间宽限内 → reconcile 删除 no-op
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="reconciled_missing",
                                   details={"disposition": "soft_deleted"})

    async def reset_pins_after_config_drift(self, kind, ext_id, *, row_revision: int) -> bool:
        """§5.2 交接：CAS 清 surface/config 两类 pin + 连带清 observed_surface_hash（R17#4）
        + 同事务 audit config_changed。False=CAS 失败（调用方放弃/409，语义 §5.2/§6.1）。"""
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live(session, kind, ext_id)
                if row is None:
                    return False
                result = await session.execute(
                    update(ExtensionModel)
                    .where(ExtensionModel.id == row.id,
                           ExtensionModel.row_revision == row_revision)
                    .values(surface_hash=None, config_fingerprint=None,
                            observed_surface_hash=None,
                            row_revision=ExtensionModel.row_revision + 1)
                    .returning(ExtensionModel.row_revision))
                if result.scalar_one_or_none() is None:
                    return False
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="config_changed")
                return True

    # ---------- 行政动作（§2 前置表；返回新 row_revision）----------

    async def quarantine(self, kind, ext_id, *, expected_row_revision, actor_user_id, note=None) -> int:
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live_or_404(session, kind, ext_id)
                if row.status == "quarantined":
                    raise InvalidStateTransitionError("already quarantined")
                # F5：before 快照必须在 UPDATE **之前**捕获进纯局部——ORM update() 的
                # synchronize_session 会把 in-memory row 同步到新值，UPDATE 后再读 row.status
                # 得到的是 after 值（before==after，审计失去转移证据）。
                before = {"status": row.status}
                new_rev = await _cas_transition(
                    session, row, expected_row_revision,
                    status="quarantined", quarantine_reason="admin_manual")
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="quarantined", actor_user_id=actor_user_id,
                                   before=before,
                                   after={"status": "quarantined", "quarantine_reason": "admin_manual"},
                                   details={"note": note} if note else None)
                return new_rev

    async def reapprove(self, kind, ext_id, *, expected_row_revision, actor_user_id) -> int:
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live_or_404(session, kind, ext_id)
                if row.status != "quarantined":
                    raise InvalidStateTransitionError("reapprove requires quarantined")
                if kind == "plugin":
                    await _raise_if_operation_pending(session, row.id)     # R46#2
                _require_valid_observations(kind, row)                      # §5.2 全类别原子性
                # F5：before 快照 + pins 均在 UPDATE 前从 in-memory row 捕获（observed 列不受
                # 本次 UPDATE 改动，但统一按「UPDATE 前捕获」以免 ORM 同步语义漂移）。
                before = {"status": row.status}
                pins = {_PIN_FOR_CATEGORY[c]: getattr(row, _OBSERVED_FOR_CATEGORY[c])
                        for c in REQUIRED_OBSERVED_CATEGORIES[kind]}
                new_rev = await _cas_transition(
                    session, row, expected_row_revision,
                    status="active", quarantine_reason=None,
                    hash_schema_version=HASH_SCHEMA_VERSION,
                    pinned_at=datetime.now(timezone.utc), pinned_by=actor_user_id, **pins)
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="reapproved", actor_user_id=actor_user_id,
                                   before=before, after={"status": "active", **pins})
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="pin_established", actor_user_id=actor_user_id)
                return new_rev

    async def set_governance_enabled(self, kind, ext_id, *, enabled, expected_row_revision, actor_user_id) -> int:
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live_or_404(session, kind, ext_id)
                before = {"status": row.status}   # F5：UPDATE 前捕获（ORM 同步会污染 after 读）
                if enabled:
                    if row.status != "disabled":
                        raise InvalidStateTransitionError("enable requires disabled")
                    if kind == "plugin":
                        await _raise_if_operation_pending(session, row.id)   # R47#1
                    new_status, event = "active", "enabled"
                else:
                    if row.status != "active":
                        # quarantined 不允许 disable 掩埋隔离证据（§2 表）
                        raise InvalidStateTransitionError("disable requires active")
                    new_status, event = "disabled", "disabled"
                new_rev = await _cas_transition(session, row, expected_row_revision, status=new_status)
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event=event, actor_user_id=actor_user_id,
                                   before=before, after={"status": new_status})
                return new_rev

    async def approve_pin(self, kind, ext_id, *, expected_row_revision, actor_user_id) -> str:
        async with self._session_factory() as session:
            async with session.begin():
                row = await _load_live(session, kind, ext_id)
                if row is None or row.status != "active":
                    return "skipped_invalid_state"     # R12#3/R13#3（批量语义；单项 404 由调用方）
                try:
                    _require_valid_observations(kind, row)
                except MissingObservationError:
                    return "skipped_no_observation"
                pins = {_PIN_FOR_CATEGORY[c]: getattr(row, _OBSERVED_FOR_CATEGORY[c])
                        for c in REQUIRED_OBSERVED_CATEGORIES[kind]}
                result = await session.execute(
                    update(ExtensionModel)
                    .where(ExtensionModel.id == row.id,
                           ExtensionModel.row_revision == expected_row_revision)
                    .values(hash_schema_version=HASH_SCHEMA_VERSION,
                            pinned_at=datetime.now(timezone.utc), pinned_by=actor_user_id,
                            row_revision=ExtensionModel.row_revision + 1, **pins)
                    .returning(ExtensionModel.row_revision))
                if result.scalar_one_or_none() is None:
                    return "conflict"
                await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=row.id,
                                   event="pin_established", actor_user_id=actor_user_id)
                return "pinned"


# ---------- module helpers ----------

def _encode_audit_cursor(model: ExtensionAuditLogModel) -> str:
    """复合游标编码 ``"{created_at.isoformat()}|{id}"``（id=UUID 不含 ``|``，isoformat 亦无）。"""
    return f"{model.created_at.isoformat()}|{model.id}"


def _decode_audit_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    created_str, id_str = cursor.rsplit("|", 1)
    return datetime.fromisoformat(created_str), uuid.UUID(id_str)


def _to_audit_entry(model: ExtensionAuditLogModel) -> AuditLogEntry:
    return AuditLogEntry(
        id=model.id, kind=model.kind, ext_id=model.ext_id,
        actor_user_id=model.actor_user_id, event=model.event,
        before=model.before, after=model.after, details=model.details,
        correlation_id=model.correlation_id, created_at=model.created_at)


async def _load_live(session: AsyncSession, kind: str, ext_id: str) -> ExtensionModel | None:
    result = await session.execute(
        select(ExtensionModel).where(
            ExtensionModel.kind == kind, ExtensionModel.ext_id == ext_id,
            ExtensionModel.deleted_at.is_(None)))
    return result.scalar_one_or_none()


async def _load_live_or_404(session, kind, ext_id) -> ExtensionModel:
    row = await _load_live(session, kind, ext_id)
    if row is None:
        raise NotFoundError(f"extension {kind}/{ext_id} not found")
    return row


async def _has_membership(session: AsyncSession, extension_id: uuid.UUID) -> bool:
    result = await session.execute(
        select(PluginMembershipModel.id)
        .where(PluginMembershipModel.child_extension_id == extension_id).limit(1))
    return result.scalar_one_or_none() is not None


async def _raise_if_operation_pending(session: AsyncSession, plugin_extension_id: uuid.UUID) -> None:
    result = await session.execute(
        select(ExtensionInstallOperationModel.id)
        .where(ExtensionInstallOperationModel.plugin_extension_id == plugin_extension_id,
               ExtensionInstallOperationModel.state.in_(("in_progress", "failed")))
        .limit(1))
    if result.scalar_one_or_none() is not None:
        raise OperationPendingError("plugin has non-terminal operation")


def _require_valid_observations(kind: str, row: ExtensionModel) -> None:
    """§5.2：该 kind 全部必需 observed 非 NULL 且版本=当前，否则 MissingObservationError。"""
    if row.observed_hash_schema_version != HASH_SCHEMA_VERSION:
        raise MissingObservationError("observed hash schema version stale or absent")
    for category in REQUIRED_OBSERVED_CATEGORIES[kind]:
        if getattr(row, _OBSERVED_FOR_CATEGORY[category]) is None:
            raise MissingObservationError(f"missing observed {category}")


async def _cas_transition(session, row, expected_row_revision: int, **values) -> int:
    result = await session.execute(
        update(ExtensionModel)
        .where(ExtensionModel.id == row.id,
               ExtensionModel.row_revision == expected_row_revision)
        .values(row_revision=ExtensionModel.row_revision + 1, **values)
        .returning(ExtensionModel.row_revision))
    new_rev = result.scalar_one_or_none()
    if new_rev is None:
        raise RevisionConflictError("row_revision mismatch")
    return new_rev


async def _install_audits(session, kind, ext_id, extension_id, ctx: InstallContext) -> None:
    """R30#1：record_install 唯一生成安装审计事实。"""
    await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=extension_id,
                       event="installed", actor_user_id=ctx.actor_user_id,
                       correlation_id=ctx.correlation_id,
                       details={"probe_failed": ctx.probe_failed, "forced": ctx.forced})
    if any(getattr(ctx, col) for col in PIN_WRITE_SET[kind]):
        await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=extension_id,
                           event="pin_established", actor_user_id=ctx.actor_user_id,
                           correlation_id=ctx.correlation_id)
    if ctx.scan is not None:
        await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=extension_id,
                           event="scan_recorded", actor_user_id=ctx.actor_user_id,
                           correlation_id=ctx.correlation_id)
    if ctx.acknowledged:
        await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=extension_id,
                           event="acknowledged", actor_user_id=ctx.actor_user_id,
                           correlation_id=ctx.correlation_id)
    if ctx.forced:
        await insert_audit(session, kind=kind, ext_id=ext_id, extension_id=extension_id,
                           event="force_installed", actor_user_id=ctx.actor_user_id,
                           correlation_id=ctx.correlation_id)


class DbExtensionRegistryReadPort:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_row(self, kind, ext_id) -> GovernanceRowSnapshot | None:
        rows = await self._snapshot_rows(kind=kind, ext_id=ext_id)
        return rows[0] if rows else None

    async def list_live_rows(self) -> list[GovernanceRowSnapshot]:
        return await self._snapshot_rows()

    async def governance_counters(self) -> GovernanceCounters:
        """R32#9：unpinned/missing 仅 active 行；unpinned 含 pin_stale（R43#4）；
        missing=任一必需 observed NULL 或版本≠当前（R43#3）。"""
        rows = await self._snapshot_rows()
        unpinned = missing = quarantined = 0
        for r in rows:
            if r.status == "quarantined":
                quarantined += 1
            if r.status != "active":
                continue
            required = REQUIRED_OBSERVED_CATEGORIES[r.kind]
            if any(pin_presence(getattr(r, _PIN_FOR_CATEGORY[c]), r.hash_schema_version,
                                HASH_SCHEMA_VERSION) != "pinned" for c in required):
                unpinned += 1
            if (r.observed_hash_schema_version != HASH_SCHEMA_VERSION
                    or any(getattr(r, _OBSERVED_FOR_CATEGORY[c]) is None for c in required)):
                missing += 1
        return GovernanceCounters(unpinned_count=unpinned,
                                  missing_observation_count=missing,
                                  quarantined_count=quarantined)

    async def list_audit(self, *, kind=None, ext_id=None, event=None,
                         cursor=None, limit=50) -> AuditPage:
        """§9.2 复合游标翻页：``(created_at, id) > (:c, :i) ORDER BY created_at, id
        LIMIT :n``（ix_extension_audit_log_created_id 支撑）。多取 1 行判 has_more →
        next_cursor（末页 None）。可选 kind/ext_id/event 过滤。"""
        conditions = []
        if kind is not None:
            conditions.append(ExtensionAuditLogModel.kind == kind)
        if ext_id is not None:
            conditions.append(ExtensionAuditLogModel.ext_id == ext_id)
        if event is not None:
            conditions.append(ExtensionAuditLogModel.event == event)
        if cursor:
            try:
                c_created, c_id = _decode_audit_cursor(cursor)
            except ValueError as exc:
                # R2b#2：畸形游标（rsplit/fromisoformat/UUID 任一失败）抛裸 ValueError——
                # 无 handler 映射 → 全局 500。翻译为 BadRequestError（400 客户端错误），
                # 与其余坏输入处置一致（客户端传了无法解析的 cursor）。
                raise BadRequestError(f"invalid audit cursor: {cursor!r}") from exc
            conditions.append(
                tuple_(ExtensionAuditLogModel.created_at, ExtensionAuditLogModel.id)
                > tuple_(c_created, c_id))
        stmt = (
            select(ExtensionAuditLogModel)
            .where(*conditions)
            .order_by(ExtensionAuditLogModel.created_at, ExtensionAuditLogModel.id)
            .limit(limit + 1))
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        has_more = len(rows) > limit
        page = list(rows[:limit])
        next_cursor = _encode_audit_cursor(page[-1]) if has_more and page else None
        return AuditPage(
            entries=[_to_audit_entry(r) for r in page], next_cursor=next_cursor)

    async def _snapshot_rows(self, kind=None, ext_id=None) -> list[GovernanceRowSnapshot]:
        parent = ExtensionModel.__table__.alias("parent")
        stmt = (
            select(ExtensionModel, parent.c.ext_id.label("parent_ext_id"))
            .select_from(ExtensionModel)
            .outerjoin(PluginMembershipModel,
                       PluginMembershipModel.child_extension_id == ExtensionModel.id)
            .outerjoin(parent, parent.c.id == PluginMembershipModel.plugin_id)
            .where(ExtensionModel.deleted_at.is_(None))
        )
        if kind is not None:
            stmt = stmt.where(ExtensionModel.kind == kind, ExtensionModel.ext_id == ext_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            out = []
            for model, parent_ext_id in result.all():
                out.append(GovernanceRowSnapshot(
                    id=model.id, kind=model.kind, ext_id=model.ext_id, status=model.status,
                    quarantine_reason=model.quarantine_reason, trust_origin=model.trust_origin,
                    source_type=model.source_type, source_ref=model.source_ref,
                    version=model.version, artifact_hash=model.artifact_hash,
                    surface_hash=model.surface_hash, config_fingerprint=model.config_fingerprint,
                    hash_schema_version=model.hash_schema_version,
                    observed_surface_hash=model.observed_surface_hash,
                    observed_artifact_hash=model.observed_artifact_hash,
                    observed_config_fingerprint=model.observed_config_fingerprint,
                    observed_hash_schema_version=model.observed_hash_schema_version,
                    last_observed_at=model.last_observed_at,
                    last_verified_at=model.last_verified_at,
                    last_mismatch_at=model.last_mismatch_at,
                    pinned_at=model.pinned_at, pinned_by=model.pinned_by,
                    scan_verdict=model.scan_verdict, scan_report=model.scan_report,
                    source_missing_at=model.source_missing_at, installed_by=model.installed_by,
                    row_revision=model.row_revision, deleted_at=model.deleted_at,
                    parent_plugin_ext_id=parent_ext_id))
            return out
