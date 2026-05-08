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
        self,
        task_id: str,
        after_event_id: str | None,
        after_seq: int | None = None,  # B3-core PR-1 §6.7
    ) -> EventRecoveryResult:
        """获取 task 实时流中 after_event_id (or after_seq) 之后的事件。

        Args:
            task_id: 任务 ID（业务标识，非 stream key）
            after_event_id: 起始 event_id（exclusive）。None 表示从头读取。
            after_seq: B3-core PR-1 §3.3 — preferred monotonic cursor for
                events with seq. When ``after_event_id`` is also provided,
                implementations may use it as the legacy ``seq is None`` floor
                so mixed streams do not drop unsequenced events after the
                client's last event id.

        Returns:
            EventRecoveryResult 包含事件列表和分页标识
        """
        ...
