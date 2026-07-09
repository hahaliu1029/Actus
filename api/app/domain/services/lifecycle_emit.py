"""C7 §2/§9 INV-C7-2 — LifecycleEvent 唯一构造入口。

生产代码禁止绕过本模块构造 LifecycleEvent（AST gate 锁定）。
显式豁免：本模块、Redis recovery 的 TypeAdapter(Event) 反序列化（合法重建）、测试代码。
"""
from typing import Optional

from app.domain.models.event import BaseEvent, LifecycleEvent
from app.domain.models.lifecycle import (
    REASON_CODES,
    STATE_FOR,
    SUPPORTED_EVENTS,
    LifecycleContractError,
    LifecycleCorrelationV1,
    LifecycleDetailV1,
    LifecycleEventKind,
    LifecycleType,
)


def build_lifecycle_event(
    lifecycle_type: LifecycleType,
    event: LifecycleEventKind,
    *,
    unit_id: str,
    epoch: int = 0,
    reason: Optional[str] = None,
    detail: Optional[LifecycleDetailV1] = None,
    parent_unit_id: Optional[str] = None,
    correlation: Optional[LifecycleCorrelationV1] = None,
    source: Optional[BaseEvent] = None,
) -> LifecycleEvent:
    """构造并校验一条 LifecycleEvent（不发射——发射走 runner 咽喉，spec §6）。

    校验（全部 raise LifecycleContractError，禁 assert——python -O 剥除断言）：
    - (lifecycle_type, event) 必须 ∈ SUPPORTED_EVENTS（INV-C7-1）
    - 非 TASK 且 epoch != 0 → raise；epoch < 0 → raise（INV-C7-8 四象限）
    - reason 必须 ∈ REASON_CODES 或 None（敏感信息禁入，spec §3.1）
    - unit_id 非空
    - state 由 STATE_FOR 派生，调用方不可指定（R10#A4）
    """
    supported = SUPPORTED_EVENTS.get(lifecycle_type, frozenset())
    if event not in supported:
        raise LifecycleContractError(
            f"unsupported lifecycle pair: ({lifecycle_type.value}, {event.value})"
        )
    if epoch < 0:
        raise LifecycleContractError(f"epoch must be >= 0, got {epoch}")
    if lifecycle_type is not LifecycleType.TASK and epoch != 0:
        raise LifecycleContractError(
            f"epoch must be 0 for non-task lifecycle, got {epoch} for {lifecycle_type.value}"
        )
    if reason is not None and reason not in REASON_CODES:
        raise LifecycleContractError(f"reason {reason!r} not in REASON_CODES")
    if not unit_id:
        raise LifecycleContractError("unit_id must be non-empty")

    return LifecycleEvent(
        lifecycle_type=lifecycle_type,
        event=event,
        state=STATE_FOR[(lifecycle_type, event)],
        unit_id=unit_id,
        epoch=epoch,
        reason=reason,
        detail=detail,
        parent_unit_id=parent_unit_id,
        correlation=correlation,
        source_event_type=(source.type if source is not None else None),
        source_event_id=(source.id if source is not None else None),
        source_seq=(getattr(source, "seq", None) if source is not None else None),
    )
