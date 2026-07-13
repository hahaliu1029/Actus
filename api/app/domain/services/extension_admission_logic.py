"""D1a admission 判定内核（纯函数——DB 壳在 infrastructure T7）。"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from app.domain.models.extension_governance import (
    ADMINISTRATIVE_REASONS,
    DETECTION_REASONS,
    AdmissionDecisionReason,
)

# §5.2 per-kind 必需观测类别（approve/reapprove 全类别原子性同源）
REQUIRED_OBSERVED_CATEGORIES: dict[str, frozenset[str]] = {
    "mcp": frozenset({"surface", "config_fingerprint"}),
    "a2a": frozenset({"surface", "config_fingerprint"}),
    "skill": frozenset({"artifact"}),
    "plugin": frozenset({"artifact"}),
}

PinPresence = Literal["pinned", "unpinned", "pin_stale"]


def pin_presence(
    pin_value: str | None, pin_schema_version: int | None, current_version: int,
) -> PinPresence:
    if pin_value is None:
        return "unpinned"
    if pin_schema_version != current_version:
        return "pin_stale"   # R1#15：pin 的 canonicalizer 版本过期=视同 unpinned
    return "pinned"


def should_persist_observation(
    *,
    stored_value: str | None,
    new_value: str,
    last_observed_at: datetime | None,
    now: datetime,
    window_seconds: int,
    stored_schema_version: int | None,
    current_schema_version: int,
) -> bool:
    """§4.2 有界观测判定（R3#10+R6#2+R40#1+R44#1a）：
    ①首次 ②值变化 ③超采样窗口 ④行内观测版本非 NULL 且≠当前（一次性强制）。
    窗口内匹配与稳态失配一视同仁=零写。"""
    if stored_value is None:
        return True
    if new_value != stored_value:
        return True
    if stored_schema_version is not None and stored_schema_version != current_schema_version:
        return True
    if last_observed_at is None:
        return True
    return (now - last_observed_at).total_seconds() > window_seconds


def select_reason(
    *,
    row_exists: bool,
    status: str | None,
    parent_blocked: bool,
    detection: AdmissionDecisionReason | None,
) -> AdmissionDecisionReason:
    """§4.1 最终 reason 优先序（非执行短路——观测持久化资格独立于行政状态，R36#1/R37#1）：
    unknown → 行政/结构类（quarantined/disabled/deleted/parent_blocked 首命中）→ 检测类 → ok。"""
    if not row_exists:
        return "unknown"
    if status in ("quarantined", "disabled", "deleted"):
        return status  # type: ignore[return-value]
    if parent_blocked:
        return "parent_blocked"
    if detection is not None:
        return detection
    return "ok"


def decide_admitted(mode: str, reason: AdmissionDecisionReason) -> bool:
    """三分分流（§4.1 R32#1/R33#1）：中性恒准入；行政类双 mode 强制不准入；
    检测类 shadow fail-open / enforce 拦。"""
    if reason in ADMINISTRATIVE_REASONS:
        return False
    if reason in DETECTION_REASONS:
        return mode != "enforce"
    return True   # NEUTRAL_REASONS
