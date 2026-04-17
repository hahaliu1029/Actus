from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select, update

from app.domain.models.memory_system_notification import (
    MemorySystemNotification,
)
from app.domain.repositories.memory_system_notification_repository import (
    MemorySystemNotificationRepository,
)
from app.infrastructure.models.memory_system_notification import (
    MemorySystemNotificationModel,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class DBMemorySystemNotificationRepository(MemorySystemNotificationRepository):
    """PostgreSQL 实现 of :class:`MemorySystemNotificationRepository`.

    Commit + rollback 由调用方管理（和 DBMemoryChunkRepository 一致）。
    Service 层持有 AsyncSession 生命周期，这里只发语句。
    """

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

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
        values: dict = {
            "id": notification_id,
            "user_id": user_id,
            "event_type": event_type,
            "payload": payload,
        }
        # 两个时间戳为 None 时让 DB server_default 接管（now() /
        # now()+30d），避免 Python/DB 时区漂移。显式传入走测试/补推路径。
        if created_at is not None:
            values["created_at"] = created_at
        if expires_at is not None:
            values["expires_at"] = expires_at

        row = MemorySystemNotificationModel(**values)
        self.db_session.add(row)
        await self.db_session.flush()
        await self.db_session.refresh(row)
        return _to_domain(row)

    async def list_unread(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[MemorySystemNotification]:
        stmt = (
            select(MemorySystemNotificationModel)
            .where(
                MemorySystemNotificationModel.user_id == user_id,
                MemorySystemNotificationModel.read_at.is_(None),
                MemorySystemNotificationModel.expires_at > func.now(),
            )
            .order_by(MemorySystemNotificationModel.created_at.desc())
            .limit(limit)
        )
        result = await self.db_session.execute(stmt)
        return [_to_domain(row) for row in result.scalars().all()]

    async def count_unread(self, user_id: str) -> int:
        # 走与 list_unread 完全一致的 WHERE 条件，让 partial index
        # ``ix_memory_system_notifications_user_unread`` 同时服务两个
        # 查询。``func.count()`` 比 ``len(list_unread())`` 便宜一个量级，
        # 因为没有行物化 + 没有 ORM hydration。
        stmt = select(func.count()).where(
            MemorySystemNotificationModel.user_id == user_id,
            MemorySystemNotificationModel.read_at.is_(None),
            MemorySystemNotificationModel.expires_at > func.now(),
        )
        result = await self.db_session.execute(stmt)
        return int(result.scalar_one())

    async def mark_read(
        self,
        *,
        notification_id: str,
        user_id: str,
    ) -> bool:
        # 只有首次由未读翻已读才视为命中；把 read_at IS NULL 放在 WHERE 里
        # 使得并发两次 mark_read 里第二次拿到 rowcount=0，调用方据此幂等
        # 返回 "已读过" 语义。若改成无条件 UPDATE，并发场景下两个请求都
        # 会拿到 1，调用方无从分辨。
        stmt = (
            update(MemorySystemNotificationModel)
            .where(
                MemorySystemNotificationModel.id == notification_id,
                MemorySystemNotificationModel.user_id == user_id,
                MemorySystemNotificationModel.read_at.is_(None),
            )
            .values(read_at=func.now())
        )
        result = await self.db_session.execute(stmt)
        return (result.rowcount or 0) > 0

    async def purge_expired(self, *, limit: int = 1000) -> int:
        # 子查询选出目标行 id，再按 id 批删。Postgres 在 DELETE 上不支持
        # LIMIT，必须 ``WHERE id IN (subquery LIMIT n)`` 或 CTE。用子查询
        # 版本保持单语句可读；limit 是保护，避免一次性 purge 百万行阻塞表。
        sub = (
            select(MemorySystemNotificationModel.id)
            .where(MemorySystemNotificationModel.expires_at <= func.now())
            .limit(limit)
            .scalar_subquery()
        )
        stmt = delete(MemorySystemNotificationModel).where(
            MemorySystemNotificationModel.id.in_(sub)
        )
        result = await self.db_session.execute(stmt)
        return result.rowcount or 0


def _to_domain(row: MemorySystemNotificationModel) -> MemorySystemNotification:
    return MemorySystemNotification(
        id=row.id,
        user_id=row.user_id,
        event_type=row.event_type,
        payload=dict(row.payload or {}),
        created_at=row.created_at,
        expires_at=row.expires_at,
        read_at=row.read_at,
    )
