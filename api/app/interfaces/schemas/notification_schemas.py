"""/v2/notifications 路由 I/O schemas。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# M1 PR-4+8 已知 event_type 枚举。前端按这个集合做 i18n / icon 映射；
# 但 schema **不用 Literal 收紧**，因为：
# - 后续 milestone（M4 trust_score_decayed 等）会在 DB 里新增 event_type
# - 新旧前后端滚动发布时，老前端读到新 event_type 不该让整个 /unread
#   响应 500。用 ``str`` 放行 + 前端对未知类型降级显示（通用 banner）
#   是"forward-compat by default"的正确姿势
# - 严格白名单如果需要，可以在 emit 侧（写路径）做，不是 read 侧
KNOWN_NOTIFICATION_EVENT_TYPES = frozenset(
    {"memory_gate_paused", "quota_exceeded", "fs_permanent_failure"}
)


class NotificationItem(BaseModel):
    """List/Response 元素——对应一条通知的对外视图。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    event_type: str = Field(
        description=(
            "Event kind. Known values in M1: memory_gate_paused, "
            "quota_exceeded, fs_permanent_failure. Future milestones may "
            "add more — clients must handle unknown values gracefully."
        )
    )
    payload: dict = Field(default_factory=dict)
    created_at: datetime
    expires_at: datetime
    read_at: datetime | None = None


class UnreadListResponse(BaseModel):
    """``GET /v2/notifications/unread`` 响应体。"""

    items: list[NotificationItem]
    unread_count: int


class MarkReadResponse(BaseModel):
    """``POST /v2/notifications/{id}/mark-read`` 响应体。"""

    marked_read: bool
