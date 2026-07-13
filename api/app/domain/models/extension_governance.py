"""D1a 扩展治理权威词表、受控 schema 与领域异常（spec §2）。

词表单权威（INV-D1-1）：ExtensionStatus/QuarantineReason/GovernanceMode/
ExtensionAuditEvent/AdmissionDecisionReason/SyncOutcome/PinApprovalOutcome
七套只在本文件定义；kind 词表的唯一 Literal 是 runtime_extension.ExtensionKind，
GovernedExtensionKind 是它的别名 import（非第二定义）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.runtime_extension import ExtensionKind

# 词表单权威消歧（R17#1）：别名，非新定义——registry/B9 聚合/wire 三处共用同一定义
GovernedExtensionKind = ExtensionKind  # = Literal["mcp", "a2a", "skill", "plugin"]

ExtensionStatus = Literal["active", "quarantined", "disabled", "deleted"]
QuarantineReason = Literal[
    "pin_mismatch",   # TOFU 校验失败（rug-pull 嫌疑）
    "admin_manual",   # Admin 手动隔离
]
GovernanceMode = Literal["off", "shadow", "enforce"]

# canonicalizer 契约版本（§5.1）；pin 的 hash_schema_version < 当前 → 视同 unpinned（pin_stale）
HASH_SCHEMA_VERSION = 1

ExtensionAuditEvent = Literal[
    "installed", "install_rejected", "uninstalled", "config_changed",
    "enabled", "disabled", "quarantined", "reapproved",
    "pin_established", "pin_mismatch", "config_drift_detected",
    "observed_first", "acknowledged",
    "scan_recorded", "force_installed",
    "reconciled_seen", "source_missing", "source_restored", "reconciled_missing",
    "plugin_expand_started", "plugin_expand_completed", "plugin_expand_compensated",
]

AdmissionDecisionReason = Literal[
    "ok", "unknown", "unpinned", "pin_stale", "quarantined", "disabled",
    "deleted", "parent_blocked", "config_drift", "pin_mismatch",
    "registry_unavailable", "mode_off",
]

# 三分归类（R33#1，§4.1 mode 语义消费）：穷尽 12 值、两两不交（契约测试锁）
NEUTRAL_REASONS = frozenset({"ok", "mode_off"})
DETECTION_REASONS = frozenset({
    "unknown", "unpinned", "pin_stale", "config_drift",
    "pin_mismatch", "registry_unavailable",
})
ADMINISTRATIVE_REASONS = frozenset({
    "quarantined", "disabled", "deleted", "parent_blocked",
})

SyncOutcome = Literal[
    "uploaded", "already_current", "no_bundle",
    "r3_rejected_old_bundle", "governance_rejected",
]

PinApprovalOutcome = Literal[
    "pinned", "skipped_no_observation", "skipped_invalid_state", "conflict",
]

# F13 惯例集合（str 域，非 Literal——skill 侧自由 str 现状；registry 列值由治理写入方字面量决定）
TRUST_ORIGINS = frozenset({"builtin", "user_installed", "agent_created"})
SOURCE_TYPES = frozenset({"local", "github", "mcp_registry", "config", "generated", "plugin"})


class GovernanceScanFinding(BaseModel):
    """§3.1 R6#8：scan_report 允许的唯一 finding 形态——禁原始 match 文本。"""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(max_length=256)
    severity: str = Field(max_length=256)
    pattern_id: str = Field(max_length=256)
    path: str = Field(max_length=256)
    line: int | None = None


class GovernanceScanSummary(BaseModel):
    """持久化与对外 DTO 的唯一 scan 形态（原始 ScanReport 仅内存使用，§7.3）。"""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["safe", "caution", "dangerous"]
    finding_count: int
    findings: list[GovernanceScanFinding] = Field(default_factory=list, max_length=50)


class GovernanceError(Exception):
    """治理领域异常基类（interfaces 层 exception handler 映射 HTTP，T20）。"""


class RevisionConflictError(GovernanceError):
    """CAS 失配（§3.6）→ HTTP 409 并发冲突。"""


class InvalidStateTransitionError(GovernanceError):
    """§2 动作×状态前置条件表外的组合 → HTTP 409 invalid_state。"""


class MissingObservationError(GovernanceError):
    """reapprove 缺必需有效观测（§5.2 全类别原子性）→ HTTP 409 missing_observation。"""


class OperationPendingError(GovernanceError):
    """plugin 行存在非终态 operation（R46#2/R47#1）→ HTTP 409 operation_pending。"""


class ManagedByPluginError(GovernanceError):
    """standalone 安装/卸载命中 plugin 成员行（§6.1/§7.1）→ HTTP 409 managed_by_plugin。"""
