"""/v2/notifications —— post-flow 系统通知读 + mark-read API。

设计取舍（见 design §165-198）：

- 仅两个路由：``GET /unread`` 轮询拉未读列表，``POST /{id}/mark-read``
  显式消费。**没有** GET /{id} 单条取 detail——列表 payload 已经完整
  （event_type + payload），前端直接渲染；单条 detail 会拆成两次 roundtrip
  且没有 "详情 diff" 需求。
- 未读列表按 ``created_at DESC`` 返回，上限 50。过了 50 条还没读完通常
  意味着用户已经忽略通知系统，再多也只是压 DOM——保留 purge_expired
  作为生命周期收尾。
- mark-read 采幂等语义：第二次点击返回 ``marked_read=false`` 而非 409，
  避免前端处理 "双击导致错误弹窗" 的 UX 毛刺。越权（非本人通知）也走
  false 分支，不暴露 "存在但非本人"。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from app.interfaces.dependencies import (
    CurrentUser,
    rate_limit_read,
    rate_limit_write,
)
from app.interfaces.schemas import Response
from app.interfaces.schemas.notification_schemas import (
    MarkReadResponse,
    NotificationItem,
    UnreadListResponse,
)
from app.interfaces.service_dependencies import (
    get_memory_system_notification_repository,
)

if TYPE_CHECKING:
    from app.domain.repositories.memory_system_notification_repository import (
        MemorySystemNotificationRepository,
    )

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/notifications", tags=["系统通知"])


@router.get(
    path="/unread",
    response_model=Response[UnreadListResponse],
    summary="获取当前用户未读系统通知",
    description=(
        "按 created_at DESC 返回；过滤已过期（expires_at <= now()）的记录。"
        "前端 header 小铃铛建议每 30s 轮询一次。"
    ),
    dependencies=[Depends(rate_limit_read)],
)
async def list_unread(
    current_user: CurrentUser,
    repo: "MemorySystemNotificationRepository" = Depends(
        get_memory_system_notification_repository
    ),
    limit: int = Query(50, ge=1, le=50),
) -> Response[UnreadListResponse]:
    # ``unread_count`` 必须是全量未读数，不是 items 切片长度——前端 badge
    # 在未读 > limit 时还要显示准确数字（"50+"）。两个查询走同一 partial
    # index，成本可忽略。
    rows = await repo.list_unread(current_user.id, limit=limit)
    total_unread = await repo.count_unread(current_user.id)
    items = [NotificationItem.model_validate(r) for r in rows]
    return Response.success(
        data=UnreadListResponse(items=items, unread_count=total_unread)
    )


@router.post(
    path="/{notification_id}/mark-read",
    response_model=Response[MarkReadResponse],
    summary="把单条通知标记为已读",
    description=(
        "幂等：重复请求 / 越权均返回 marked_read=false，无 404/409 区分—— "
        "避免用户重复点击时触发错误弹窗。"
    ),
    dependencies=[Depends(rate_limit_write)],
)
async def mark_read(
    notification_id: str,
    current_user: CurrentUser,
    repo: "MemorySystemNotificationRepository" = Depends(
        get_memory_system_notification_repository
    ),
) -> Response[MarkReadResponse]:
    marked = await repo.mark_read(
        notification_id=notification_id,
        user_id=current_user.id,
    )
    return Response.success(data=MarkReadResponse(marked_read=marked))
