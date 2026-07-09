"""C7 §4.5 R15 — research child lifecycle 的 best-effort 注入 sink。

research 管线独立、不经 domain event_queue（F18）；本 sink 把 subagent
lifecycle 注入 parent 主流：AgentService._emit_event（producer INCR+XADD）+
PG session.events。**仅当 parent 有活跃 task_id 时可投递**（_emit_event 返回
None = skip）；skip 不静默——结构化日志 + lifecycle_research_sink_skipped_total
计数器（缺口在投递维度非词表维度，spec §1 局限 5）。

投递保证与 coordinator 管线（经咽喉强保证）不同——本管线 best-effort，
验收是可观测断言非投递断言（§12-5）。sink 永不 raise。
"""
import logging
from typing import Any, Callable, Optional

from app.domain.models.event import LifecycleEvent
from app.domain.models.lifecycle import (
    LifecycleDetailV1,
    LifecycleEventKind,
    LifecycleType,
)
from app.domain.services.lifecycle_emit import build_lifecycle_event
from app.interfaces.schemas.subagent import ChildOutcome

logger = logging.getLogger(__name__)

_DONE_MAP = {
    ChildOutcome.COMPLETED: (LifecycleEventKind.COMPLETED, None),
    ChildOutcome.FAILED: (LifecycleEventKind.FAILED, "worker_failed"),
    ChildOutcome.TIMED_OUT: (LifecycleEventKind.FAILED, "watchdog_timeout"),
    ChildOutcome.CANCELLED: (LifecycleEventKind.CANCELLED, None),
    # 防御值：接口注释明言不应出现（schemas/subagent.py:18）
    ChildOutcome.WAITING: (LifecycleEventKind.FAILED, "waiting_unsupported"),
}


def _default_skip_counter() -> Any:
    """镜像 memory_recall_telemetry 的 lazy OTel 模式——setup 前是 no-op proxy。"""
    try:
        from app.infrastructure.observability.otel_meter import OtelMeter
        return OtelMeter().create_counter(
            "lifecycle_research_sink_skipped_total",
            description="research lifecycle events skipped (no active parent task)",
        )
    except Exception:  # noqa: BLE001 — 观测组件缺失不阻断业务
        class _Noop:
            def add(self, n: int, attributes: Any = None) -> None:
                return None
        return _Noop()


class ResearchLifecycleSink:
    def __init__(
        self,
        *,
        emit_event: Callable[..., Any],          # AgentService._emit_event(session_id, event)
        uow_factory: Callable[[], Any],
        flags_getter: Callable[[], Any],          # () -> LifecycleRuntimeConfig（快照原子换新）
        skip_counter: Optional[Any] = None,
    ) -> None:
        self._emit_event = emit_event
        self._uow_factory = uow_factory
        self._flags_getter = flags_getter
        self._skip_counter = skip_counter if skip_counter is not None else _default_skip_counter()

    def _enabled(self) -> bool:
        try:
            cfg = self._flags_getter()
            return bool(
                getattr(cfg, "lifecycle_events_enabled", False)
                and getattr(cfg, "lifecycle_subagent_events_enabled", False)
            )
        except Exception:  # noqa: BLE001
            return False

    async def child_started(self, *, parent_session_id: str, child_session_id: str) -> bool:
        if not self._enabled():
            return False  # flag-off 不是 skip，是零构造（INV-C7-3）
        ev = build_lifecycle_event(
            LifecycleType.SUBAGENT, LifecycleEventKind.STARTED,
            unit_id=child_session_id, parent_unit_id=parent_session_id,
        )
        return await self._deliver(parent_session_id, ev)

    async def child_done(
        self, *, parent_session_id: str, child_session_id: str, outcome: ChildOutcome,
    ) -> bool:
        if not self._enabled():
            return False
        kind, reason = _DONE_MAP.get(outcome, (LifecycleEventKind.FAILED, "unknown_terminal_outcome"))
        detail = None
        if reason in ("waiting_unsupported", "unknown_terminal_outcome"):
            detail = LifecycleDetailV1(original_outcome=getattr(outcome, "value", str(outcome)))
        ev = build_lifecycle_event(
            LifecycleType.SUBAGENT, kind,
            unit_id=child_session_id, parent_unit_id=parent_session_id,
            reason=reason, detail=detail,
        )
        return await self._deliver(parent_session_id, ev)

    async def _deliver(self, parent_session_id: str, event: LifecycleEvent) -> bool:
        try:
            stream_id = await self._emit_event(parent_session_id, event)
            if stream_id is None:
                # skip 不静默（R15）：无活跃 task/emitter 不可用 → 计数 + 结构化日志
                self._skip_counter.add(1, attributes={"parent_session_id": parent_session_id})
                logger.warning(
                    "lifecycle_research_sink_skipped: parent=%s unit=%s event=%s "
                    "(no active task_id — delivery gap is observable by design)",
                    parent_session_id, event.unit_id, event.event.value,
                )
                return False
            async with self._uow_factory() as uow:
                await uow.session.add_event(parent_session_id, event)
            return True
        except Exception:  # noqa: BLE001 — 观测面永不破坏 research 主流程
            logger.warning(
                "research lifecycle sink delivery failed: parent=%s unit=%s",
                parent_session_id, event.unit_id, exc_info=True,
            )
            return False
