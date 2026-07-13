"""B9 运行时扩展聚合 domain 模型——ExtensionKind 与中间 DTO 的权威定义（spec §2 R9#4）；wire schema 在 interfaces/schemas/runtime_extensions.py 镜像"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ExtensionKind = Literal["mcp", "a2a", "skill", "plugin"]  # domain 权威定义；interfaces 复用此 Literal（D1a §9.1 扩 plugin）

ConfigReasonCode = Literal[
    "enabled", "disabled_global", "disabled_user", "disabled_both",
    "user_enablement_unknown", "config_unreadable",
    "not_applicable_plugin",   # D1a §9.1：plugin 元容器不可执行/不可用户启停（enablement N/A）
    "governance_blocked",      # D1a §9.1：quarantined/disabled/parent_blocked → 有效停用（唯一治理泄漏）
]
HealthState = Literal["reachable", "unreachable", "ok", "error", "unknown", "skipped"]
LivenessState = Literal["in_use", "idle", "unknown", "not_applicable"]
StatsUnavailableReason = Literal["disabled", "redis_unavailable", "unsupported", "admin_only"]
ProbeErrorCode = Literal["timeout", "connect_failed", "auth_failed", "protocol_error", "spawn_failed"]


@dataclass(frozen=True)
class ExtensionConfigInfo:
    enabled_global: bool
    enabled_user: bool | None
    effective_enabled: bool
    reason_code: ConfigReasonCode


@dataclass(frozen=True)
class ExtensionHealthInfo:
    kind: Literal["probe", "integrity"]
    state: HealthState
    last_checked_at: datetime | None = None
    latency_ms: int | None = None
    error_code: str | None = None
    error_message: str | None = None          # 已脱敏（Task 13 保证；投影层只负责 Admin-only 裁剪）
    relative_file: str | None = None          # 不变式：kind="probe" ⇒ 恒 None（R12#8）
    stale: bool = False
    consecutive_failures: int = 0
    next_probe_at: datetime | None = None


@dataclass(frozen=True)
class ExtensionLivenessInfo:
    state: LivenessState
    active_run_count: int = 0


@dataclass(frozen=True)
class ExtensionStatsInfo:
    available: bool
    unavailable_reason: StatsUnavailableReason | None = None
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_active_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None


@dataclass(frozen=True)
class ExtensionGovernanceInfo:
    """D1a §9.1 R46#8：registry 治理块（Admin-only 投影）——GovernanceRowSnapshot 的
    读模型平移 + pin_presence 派生布尔。非 Admin 投影层置 None（三层剥离第①层）。"""
    status: str
    trust_origin: str
    # pinned/unpinned/pin_stale：对该 kind 必需 pin 写集聚合派生（全 pinned→pinned；
    # 任一 unpinned→unpinned；任一 stale→pin_stale）
    pinned: bool
    unpinned: bool
    pin_stale: bool
    scan_verdict: str | None
    quarantine_reason: str | None
    last_mismatch_at: datetime | None
    last_verified_at: datetime | None
    row_revision: int
    observed_surface_hash: str | None
    observed_artifact_hash: str | None
    observed_config_fingerprint: str | None
    pinned_at: datetime | None
    pinned_by: str | None
    installed_by: str | None
    source_type: str
    source_ref: str | None
    version: str | None
    source_missing_at: datetime | None
    parent_plugin_ext_id: str | None = None


@dataclass(frozen=True)
class ExtensionItemInfo:
    kind: ExtensionKind
    id: str                                   # mcp=server_name；a2a=config uuid；skill=skill.id
    name: str
    description: str | None
    config: ExtensionConfigInfo
    health: ExtensionHealthInfo
    liveness: ExtensionLivenessInfo
    stats: ExtensionStatsInfo
    details: dict[str, Any]                   # 已按角色投影后的 shape（§6 冻结）
    governance: ExtensionGovernanceInfo | None = None  # D1a：Admin-only；mode=off/非 Admin=None


@dataclass(frozen=True)
class RuntimeExtensionsSnapshot:
    items: tuple[ExtensionItemInfo, ...]
    snapshot_at: datetime
    probe_enabled: bool                       # 有效能力（flag on 且循环成功启动），非 raw 配置值
    stats_enabled: bool


@dataclass(frozen=True)
class LivenessSnapshot:                       # RuntimeLivenessRegistry.snapshot() 返回值
    active: dict[tuple[str, str], frozenset[str]]   # (kind, id) -> run_id 集合
    degraded: bool
