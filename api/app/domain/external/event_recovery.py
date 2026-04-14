# api/app/domain/external/event_recovery.py
from dataclasses import dataclass
from typing import List, Protocol

from app.domain.models.event import Event


@dataclass(frozen=True)
class EventRecoveryResult:
    """从 task 实时流恢复的事件结果"""
    events: List[Event]
    has_more: bool


class EventRecoveryPort(Protocol):
    """事件恢复端口协议。

    从 task 的实时流中获取指定 event_id 之后的事件。
    task_id 是业务标识，stream key 格式由实现层决定。
    """

    async def get_recent_events(
        self, task_id: str, after_event_id: str | None
    ) -> EventRecoveryResult:
        """获取 task 实时流中 after_event_id 之后的事件。

        Args:
            task_id: 任务 ID（业务标识，非 stream key）
            after_event_id: 起始 event_id（exclusive）。None 表示从头读取。

        Returns:
            EventRecoveryResult 包含事件列表和分页标识
        """
        ...
