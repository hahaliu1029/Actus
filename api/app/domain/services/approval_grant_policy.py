"""R5 CS4 approval grant 策略辅助。

集中放置 grant 过期时间、命令/目录匹配等纯函数，供 Writer / Reader
复用，保证语义一致。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from typing import Optional

# R5 Phase 1 固定 session grant 过期时间 24h（design doc §Open Questions 2）
SESSION_GRANT_TTL = timedelta(hours=24)


def to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """把 aware datetime 归一成 naive UTC wall-clock。

    Actus 约定：所有 DateTime 列存 naive UTC（``TIMESTAMP WITHOUT TIME ZONE``）。
    domain 层保留 aware UTC 以便语义清晰，**进 DB 前**调用本函数剥 tzinfo。

    规则：
    - ``None`` → ``None``
    - naive datetime → 原样返回（假定已是 UTC 墙钟；调用方不应传 naive-local）
    - aware datetime → ``astimezone(UTC).replace(tzinfo=None)``（转 UTC 后剥 tzinfo）

    使用场景：
    - ``DBApprovalGrantRepository.create()`` 写 ``expires_at`` 前归一
    - ``DBApprovalGrantRepository.find_active_grants()`` 计算 now 的比较值
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def session_grant_expires_at(now: Optional[datetime] = None) -> datetime:
    """返回 session scope grant 的 ``expires_at``：UTC now + 24h。

    使用 ``datetime.now(timezone.utc)``（非已弃用的 ``utcnow()``）。
    ``now`` 参数仅供测试注入固定时间，线上调用不传。
    """
    base = now if now is not None else datetime.now(timezone.utc)
    return base + SESSION_GRANT_TTL


def match_command_and_dir(
    primary_arg: str,
    dir_arg: Optional[str],
    pattern: str,
    dir_pattern: str,
) -> bool:
    """Grant 命中判定。

    语义锁定到 ``ApprovalGrant.matches``：
    - ``primary_arg`` 必须 fnmatch ``pattern``
    - ``dir_pattern`` 为空串：不约束目录
    - ``dir_pattern`` 非空：``dir_arg`` 不能为 None 且要 fnmatch ``dir_pattern``
    """
    if not fnmatch(primary_arg, pattern):
        return False
    if dir_pattern:
        if dir_arg is None:
            return False
        if not fnmatch(dir_arg, dir_pattern):
            return False
    return True
