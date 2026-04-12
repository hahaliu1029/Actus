import logging
from typing import List

from pydantic import TypeAdapter

from app.domain.external.event_recovery import EventRecoveryPort, EventRecoveryResult
from app.domain.models.event import Event
from app.infrastructure.external.message_queue.redis_stream_message_queue import (
    RedisStreamMessageQueue,
)

logger = logging.getLogger(__name__)

_event_adapter = TypeAdapter(Event)
_DEFAULT_MAX_COUNT = 10000


class RedisEventRecovery(EventRecoveryPort):
    """从 Redis Stream 恢复 task 事件的实现。

    stream key 格式 `task:output:{task_id}` 封装在此层。
    """

    def __init__(self, max_count: int = _DEFAULT_MAX_COUNT) -> None:
        self._max_count = max_count

    async def get_recent_events(
        self, task_id: str, after_event_id: str | None
    ) -> EventRecoveryResult:
        stream_name = f"task:output:{task_id}"
        queue = RedisStreamMessageQueue(stream_name)

        start_id = after_event_id if after_event_id else "-"
        events: List[Event] = []

        try:
            async for message_id, event_str in queue.get_range(
                start_id=start_id, count=self._max_count
            ):
                if event_str is None:
                    continue
                # xrange inclusive：跳过 ID == after_event_id 的首条
                if after_event_id and message_id == after_event_id:
                    continue

                try:
                    event = _event_adapter.validate_json(event_str)
                    # 回填 Redis Stream ID（关键！与 agent_service.py:616 同模式）
                    event.id = message_id
                    events.append(event)
                except Exception:
                    logger.warning(
                        "event_recovery: 反序列化失败 stream=%s id=%s",
                        stream_name,
                        message_id,
                    )
                    continue
        except Exception:
            logger.warning(
                "event_recovery: 读取 stream 失败 stream=%s", stream_name
            )
            return EventRecoveryResult(events=[], has_more=False)

        has_more = len(events) >= self._max_count
        return EventRecoveryResult(events=events, has_more=has_more)
