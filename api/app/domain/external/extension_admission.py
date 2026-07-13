"""D1a §4.1 治理 Port 合同（admission / write / read 三面）。

- ExtensionAdmissionPort：受限 admission 写 port——唯一允许的写 =
  原子 record_observation_and_maybe_quarantine（observed_*/last_* + 条件 quarantine + audit，
  单 DB 事务，实现侧 §3.6）。禁止 pin 写/行政写（INV-D1-3，AST gate T8）。
- ExtensionRegistryWritePort：治理写（安装/卸载/reconcile 记账/行政动作/approve/reset）。
- ExtensionRegistryReadPort：只读投影（治理端点/B9 governance block 消费）。
  spec 未命名读面——plan 定名，不动 admission/write 二分。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol, Sequence
from uuid import UUID

from app.domain.models.extension_governance import (
    AdmissionDecisionReason,
    GovernanceScanSummary,
    GovernedExtensionKind,
    PinApprovalOutcome,
)


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: AdmissionDecisionReason
    row_revision: int | None          # 观测/判定后的行 revision（行不存在=None；R15#2）
    observation_outcome: Literal["persisted", "unchanged", "conflict", "none"] = "none"
    config_drift_detected: bool = False   # R46#5：可比 pin 前置下 observed≠pin；独立于 reason 优先序


@dataclass(frozen=True)
class UninstallContext:
    """R26#1：一切卸载路径都带 context——saga=operation.id / standalone=None。"""
    correlation_id: UUID | None
    actor_user_id: str                # 恒必填（users.id 是 str/String(255)，R23）


@dataclass(frozen=True)
class InstallContext:
    """R25#2：安装写入上下文（pin 建立的唯一载体）。"""
    actor_user_id: str
    correlation_id: UUID | None       # standalone=None；plugin saga=operation.id
    source_type: str                  # §3.1 词表（SOURCE_TYPES 惯例集合）
    source_ref: str | None            # canonicalize_source_ref 后的值（T5 落地，R6#A1 归属修）
    version: str | None
    trust_origin: str                 # TRUST_ORIGINS 惯例集合
    artifact_hash: str | None         # per-kind pin 写集（§5.2）——kind 不适用=None
    surface_hash: str | None
    config_fingerprint: str | None
    hash_schema_version: int          # = 当前 HASH_SCHEMA_VERSION
    scan: GovernanceScanSummary | None
    probe_failed: bool = False        # R30#1：审计事实三字段
    acknowledged: bool = False
    forced: bool = False


@dataclass(frozen=True)
class Observation:
    """R1#15：带类型观测快照——canonical 化前原始载荷，port 内 canonicalize+hash。"""
    category: Literal["surface", "artifact", "config_fingerprint"]
    payload: Any
    schema_version: int
    under_config_fingerprint: str | None = None   # R34#2：mcp/a2a surface 观测必带；artifact/config 恒 None


@dataclass(frozen=True)
class GovernanceCounters:
    """GET /v2/extensions/governance 读模型（R32#9 scope=仅 active 行）。"""
    unpinned_count: int               # active 且 enforce 会以 unpinned/pin_stale 拒绝（R43#4）
    missing_observation_count: int    # active 且任一必需 observed 为 NULL 或版本≠当前（R43#3）
    quarantined_count: int


@dataclass(frozen=True)
class AuditLogEntry:
    """extension_audit_log 行的只读快照（§9.2 GET /audit 读模型）。"""
    id: UUID
    kind: str
    ext_id: str | None
    actor_user_id: str | None
    event: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    details: dict[str, Any] | None
    correlation_id: UUID | None
    created_at: datetime


@dataclass(frozen=True)
class AuditPage:
    """audit 复合游标翻页结果（cursor 稳定=(created_at, id)；末页 next_cursor=None）。"""
    entries: list[AuditLogEntry]
    next_cursor: str | None = None


@dataclass(frozen=True)
class GovernanceRowSnapshot:
    """registry 行的只读快照（§9.1 governance block 与治理端点的读模型来源）。"""
    id: UUID
    kind: str
    ext_id: str
    status: str
    quarantine_reason: str | None
    trust_origin: str
    source_type: str
    source_ref: str | None
    version: str | None
    artifact_hash: str | None
    surface_hash: str | None
    config_fingerprint: str | None
    hash_schema_version: int
    observed_surface_hash: str | None
    observed_artifact_hash: str | None
    observed_config_fingerprint: str | None
    observed_hash_schema_version: int | None
    last_observed_at: datetime | None
    last_verified_at: datetime | None
    last_mismatch_at: datetime | None
    pinned_at: datetime | None
    pinned_by: str | None
    scan_verdict: str | None
    scan_report: dict[str, Any] | None
    source_missing_at: datetime | None
    installed_by: str | None
    row_revision: int
    deleted_at: datetime | None
    parent_plugin_ext_id: str | None = None   # membership join 派生（§8.5——不写入成员行）


class ExtensionAdmissionPort(Protocol):
    mode: str   # R1#4：实现必须公开生效 mode（"shadow"/"enforce"）——gates 双保险
                # 异常兜底按 mode 分流（INV-D1-6：enforce fail-closed / shadow fail-open）

    async def check_many(
        self,
        kind: GovernedExtensionKind,
        ext_ids: Sequence[str],
        config_fingerprints: Mapping[str, str] | None = None,
    ) -> dict[str, AdmissionDecision]:
        """批量准入判定（G1/G4/G4b/G5 状态查）。

        config_fingerprints（R1#3）：mcp/a2a 调用方随查随带**已 canonical/hash 的**
        指纹字符串（G1 处用 §5.1 canonicalizer 纯函数算出，port 内不重复哈希 R9#3）；
        每个入参指纹在 port 内构造 Observation(category="config_fingerprint")
        进入与 verify_observation 完全相同的原子记录路径（§3.6）。
        """
        ...

    async def verify_observation(
        self,
        kind: GovernedExtensionKind,
        ext_id: str,
        obs: Observation,
    ) -> AdmissionDecision:
        """统一 surface/artifact 校验：canonicalize → hash → 与 pin 比对 →
        写 observed_*/audit →（enforce∧active∧mismatch，仅 surface/artifact）quarantine。"""
        ...


class ExtensionRegistryWritePort(Protocol):
    """治理写面。行政动作违反 §2 前置表 → 抛 domain 异常
    （InvalidStateTransitionError/RevisionConflictError/MissingObservationError/
    OperationPendingError，T1）；行不存在 → 返回 None 语义由实现层定义为抛
    NotFound 类异常（T8 用 application.errors.NotFoundError 先例）。"""

    async def record_install(
        self, kind: GovernedExtensionKind, ext_id: str, install_context: "InstallContext",
    ) -> None:
        """安装/重装落行（§6.1）：insert 或 update-only（既有行）；update 恒
        CAS-bump row_revision（R48#1，内部 read-CAS-retry 有界重试 3 次 R49#3）、
        只写 pin/scan/provenance + 清空 observed 三列与 observed_hash_schema_version
        （R46#6）、恒不改 status（R44#3）；事务内 membership 守卫（R32#4：
        standalone context 落在带 membership 的行 → raise ManagedByPluginError 回滚）；
        audit installed(+pin_established/acknowledged/force_installed/scan_recorded
        按 context 事实字段，R30#1)。"""
        ...

    async def record_delete(
        self, kind: GovernedExtensionKind, ext_id: str, *, uninstall_context: "UninstallContext",
    ) -> None:
        """软删 + audit uninstalled（双字段=correlation_id/actor，R22#F2）；
        幂等：行已软删/不存在 → 零写零 audit。"""
        ...

    # ---- reconcile 记账族（§6.1-2/§6.3；actor=NULL 系统）----
    async def record_reconciled_seen(
        self, kind: GovernedExtensionKind, ext_id: str, *,
        source_type: str, source_ref: str | None, version: str | None, trust_origin: str,
    ) -> None:
        """无 context 补建 unpinned 行 + audit reconciled_seen（已存在 live 行=幂等零写）。"""
        ...

    async def mark_source_missing(self, kind: GovernedExtensionKind, ext_id: str) -> None:
        """source_missing_at 置位 + audit source_missing（已置位=幂等零写）。"""
        ...

    async def mark_source_restored(self, kind: GovernedExtensionKind, ext_id: str) -> None:
        """清 source_missing_at + audit source_restored（未置位=幂等零写）。"""
        ...

    async def record_reconciled_missing(self, kind: GovernedExtensionKind, ext_id: str) -> None:
        """二次确认缺源（§6.3 R33#5 分流）：standalone=软删；plugin=保持行
        （R32#5）——audit reconciled_missing details.disposition ∈ {soft_deleted, retained_plugin}。"""
        ...

    async def reset_pins_after_config_drift(
        self, kind: GovernedExtensionKind, ext_id: str, *, row_revision: int,
    ) -> bool:
        """§5.2 pin-reset 交接：CAS（WHERE row_revision=入参）清 surface/config 两类 pin
        + 连带清 observed_surface_hash（R17#4）+ 同事务 audit config_changed。
        返回 CAS 是否成功（False=放弃不重试，调用方语义 §5.2/§6.1）。"""
        ...

    # ---- 行政动作（§2 前置表 + §3.6 CAS；返回新 row_revision）----
    async def quarantine(
        self, kind: GovernedExtensionKind, ext_id: str, *,
        expected_row_revision: int, actor_user_id: str, note: str | None = None,
    ) -> int:
        """reason 恒 admin_manual（R17#5）；note sanitize ≤500 字符进 details（R18#5）。"""
        ...

    async def reapprove(
        self, kind: GovernedExtensionKind, ext_id: str, *,
        expected_row_revision: int, actor_user_id: str,
    ) -> int:
        """quarantined→active + observed→pin（全类别原子性 §5.2；缺有效观测=
        MissingObservationError；plugin 非终态 operation=OperationPendingError R46#2）。"""
        ...

    async def set_governance_enabled(
        self, kind: GovernedExtensionKind, ext_id: str, *,
        enabled: bool, expected_row_revision: int, actor_user_id: str,
    ) -> int:
        """active↔disabled（§2 表；enable 对 plugin 非终态 operation=OperationPendingError R47#1）。"""
        ...

    async def approve_pin(
        self, kind: GovernedExtensionKind, ext_id: str, *,
        expected_row_revision: int, actor_user_id: str,
    ) -> PinApprovalOutcome:
        """observed→pin 单行转正（仅 active；R12#3/R13#3 状态前置由调用方批量语义处理）。"""
        ...


class ExtensionRegistryReadPort(Protocol):
    async def get_row(
        self, kind: GovernedExtensionKind, ext_id: str,
    ) -> GovernanceRowSnapshot | None:
        """live 行快照（软删=None）；含 parent_plugin_ext_id join 派生。"""
        ...

    async def list_live_rows(self) -> list[GovernanceRowSnapshot]:
        """全部 live 行（B9 governance 源 + 批量端点 all:true 扫描基础）。"""
        ...

    async def governance_counters(self) -> GovernanceCounters:
        """R32#9/R43#3/R43#4 计数语义（仅 active 行；unpinned 含 pin_stale）。"""
        ...

    async def list_audit(
        self,
        *,
        kind: str | None = None,
        ext_id: str | None = None,
        event: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> AuditPage:
        """§9.2 GET /audit 复合游标翻页：``WHERE (created_at, id) > (:c, :i)
        ORDER BY created_at, id LIMIT :n``（cursor 编码 ``"{created_at.isoformat()}|{id}"``）；
        可选 kind/ext_id/event 过滤。返回本页 + next_cursor（末页 None）。"""
        ...
