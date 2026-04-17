from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

MemorySystemNotificationEventType = Literal[
    "memory_gate_paused",
    "quota_exceeded",
    "fs_permanent_failure",
]


@dataclass(frozen=True)
class MemorySystemNotification:
    """Post-flow 用户通知——与 ``MemoryAuditLog`` 分离。

    Audit log 记录 "做了什么"（幂等事实流），notification 记录 "要告诉
    用户什么" 并带 read lifecycle。合并两者会把 read_at 污染到只写的
    审计路径上，每条 audit 都得判一次 read？—— 不对称，分表更干净。

    M1 PR-4+8 起用的三个 event_type（见 design §183）：
    - ``memory_gate_paused``：gate circuit breaker OPEN，当前 flush 块
      被完整丢弃；payload 含 ``consecutive_failures`` / ``cooldown_until``
    - ``quota_exceeded``：per-user 每日 auto-promote cap 打满，当天
      后续自动写入被降级/跳过
    - ``fs_permanent_failure``：FsMemoryWriter 重试 5 次仍失败，DB 行
      存在但文件未落盘（M1 期间只能靠 reconciler 再试）
    后续 milestone 可能新增，所以 DB CHECK 不收紧枚举，白名单校验在
    Pydantic schema 层面做（interface 入口）。
    """

    id: str
    user_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    read_at: datetime | None = None
