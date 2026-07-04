"""B9 运行时扩展聚合 domain 模型——ExtensionKind 与中间 DTO 的权威定义（spec §2 R9#4）；wire schema 在 interfaces/schemas/runtime_extensions.py 镜像"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ExtensionKind = Literal["mcp", "a2a", "skill"]          # domain 权威定义；interfaces 复用此 Literal

ConfigReasonCode = Literal[
    "enabled", "disabled_global", "disabled_user", "disabled_both",
    "user_enablement_unknown", "config_unreadable",
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
