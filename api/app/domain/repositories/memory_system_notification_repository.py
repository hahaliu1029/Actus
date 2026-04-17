from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.domain.models.memory_system_notification import (
        MemorySystemNotification,
    )


class MemorySystemNotificationRepository(Protocol):
    """``memory_system_notifications`` 的仓库协议。

    选用 Protocol 与 MemoryChunkRepository 一致；Clean Architecture 上
    domain 只声明签名，infra 提供 PostgreSQL 实现。
    """

    async def create(
        self,
        *,
        notification_id: str,
        user_id: str,
        event_type: str,
        payload: dict,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> MemorySystemNotification:
        """插入一条通知。

        ``created_at`` / ``expires_at`` 为 None 时由 DB 默认值填充（now()
        与 now() + 30d）。显式传入用于测试或补推历史通知场景。
        """
        ...

    async def list_unread(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[MemorySystemNotification]:
        """返回当前用户未读（``read_at IS NULL``）且未过期（``expires_at >
        now()``）的通知，按 ``created_at DESC`` 排序。

        走 partial index ``ix_memory_system_notifications_user_unread``
        实现 O(unread) 查询；已过期行虽仍留在表里，但这里主动过滤掉，
        避免前端看到 30 天前的陈旧告警。
        """
        ...

    async def count_unread(self, user_id: str) -> int:
        """当前用户未读 + 未过期通知的**总数**，与 ``list_unread`` 相同
        的 WHERE 条件但不受 ``limit`` 截断。

        前端 polling 拿到 ``items``（被 limit=50 截断）+ ``count_unread``
        （真实总数）就能正确渲染 badge："50+"。如果只用 ``len(items)``
        代替，会在未读超过 50 时悄悄低估。
        """
        ...

    async def mark_read(
        self,
        *,
        notification_id: str,
        user_id: str,
    ) -> bool:
        """把某条通知标记为已读（``read_at = NOW()``）。

        返回是否真的命中一行（存在 + 属于该用户 + 先前未读）。二次点击
        / 并发点击会落到这个分支，返回 False，调用方视为幂等 no-op。
        越权（user_id 不匹配）也返回 False——不暴露 "存在但非本人"，
        让越权与 "不存在" 对外语义一致，避免通过状态码枚举用户空间。
        """
        ...

    async def purge_expired(self, *, limit: int = 1000) -> int:
        """清理 ``expires_at <= now()`` 的行。返回实际删除数量。

        PR-4+8 不自动挂 cron——先留一条手动 CLI（post-M3 再接）。提供这
        个方法是为了写测试 + 将来挂后台任务时有现成入口。
        """
        ...
