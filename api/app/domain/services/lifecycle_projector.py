"""C7 §6 — runner 咽喉 lifecycle 投影器。

调用契约（spec §6，R10#A5/A6）：
- 由 AgentTaskRunner._put_and_add_event 的 post-stamp hook 在 source 事件
  seq 已盖、id 已定、persist 已落位之后**同协程内联**调用；
- project() 返回 Sequence[LifecycleEvent]（reduce 是 1→N 展开且含 async repo
  反查——非 Optional、非纯函数）；
- 投影器自身无状态（跨重启重复终态由前端 reducer sticky 规则权威去重，
  INV-C7-5；emitter 侧 in-process best-effort 去重在 runner hook）。
分支覆盖：PR2 plan/step；PR3 tool；PR5 subagent（coordinator 三事件）。
"""
import logging
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from app.domain.models.event import (
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    CoordinatorWorkerSpawnedEvent,
    Event,
    LifecycleEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.lifecycle import (
    LifecycleCorrelationV1,
    LifecycleDetailV1,
    LifecycleEventKind,
    LifecycleType,
)
from app.domain.services.lifecycle_emit import build_lifecycle_event

logger = logging.getLogger(__name__)

# coordinator_run_id -> {work_unit_id: child_session_id}（PR5 注入，spec R3#2/R4#3）
ChildLookup = Callable[[str], Awaitable[Dict[str, str]]]


class LifecycleProjector:
    def __init__(
        self,
        *,
        parent_session_id: str = "",
        subagent_enabled: Optional[Callable[[], bool]] = None,
        child_lookup: Optional[ChildLookup] = None,
    ) -> None:
        self._parent_session_id = parent_session_id
        self._subagent_enabled = subagent_enabled or (lambda: False)
        self._child_lookup = child_lookup

    async def project(self, event: Event) -> Sequence[LifecycleEvent]:
        # R3#5 自投影硬守卫：映射表误加 lifecycle 源会造成递归——直接空集
        if isinstance(event, LifecycleEvent):
            return []
        if isinstance(event, PlanEvent):
            return self._project_plan(event)
        if isinstance(event, StepEvent):
            return self._project_step(event)
        if isinstance(event, ToolEvent):
            return self._project_tool(event)
        if isinstance(event, (CoordinatorWorkerSpawnedEvent, CoordinatorReduceEvent, CoordinatorSiblingCancelEvent)):
            if not self._subagent_enabled():
                return []  # R10#A9 运行期 AND 门（master 已在 hook 入口查过）
            if isinstance(event, CoordinatorWorkerSpawnedEvent):
                return self._project_worker_spawned(event)
            if isinstance(event, CoordinatorReduceEvent):
                return await self._project_reduce(event)
            return await self._project_sibling_cancel(event)
        return []

    # --- §4.1 plan（双 planner 路径经咽喉天然覆盖；steps snapshot 不派生 step） ---
    def _project_plan(self, event: PlanEvent) -> List[LifecycleEvent]:
        if event.status == PlanEventStatus.CREATED:
            return [build_lifecycle_event(
                LifecycleType.PLAN, LifecycleEventKind.STARTED,
                unit_id=event.plan.id,
                detail=LifecycleDetailV1(note="plan_created_not_yet_executing"),
                source=event,
            )]
        if event.status == PlanEventStatus.UPDATED:
            return [build_lifecycle_event(
                LifecycleType.PLAN, LifecycleEventKind.PROGRESS,
                unit_id=event.plan.id, reason="plan_updated", source=event,
            )]
        if event.status == PlanEventStatus.COMPLETED:
            return [build_lifecycle_event(
                LifecycleType.PLAN, LifecycleEventKind.COMPLETED,
                unit_id=event.plan.id, source=event,
            )]
        return []

    # --- §4.2 step（仅 main_graph 源，F11；failed 经 COMPLETED+success=False） ---
    def _project_step(self, event: StepEvent) -> List[LifecycleEvent]:
        if event.status == StepEventStatus.STARTED:
            return [build_lifecycle_event(
                LifecycleType.STEP, LifecycleEventKind.STARTED,
                unit_id=event.step.id, source=event,
            )]
        if event.status == StepEventStatus.COMPLETED and event.step.success is True:
            return [build_lifecycle_event(
                LifecycleType.STEP, LifecycleEventKind.COMPLETED,
                unit_id=event.step.id, source=event,
            )]
        if event.status == StepEventStatus.COMPLETED or event.status == StepEventStatus.FAILED:
            # COMPLETED+success=False（生产失败路径，R5#P1）或防御性 FAILED（无生产源）
            return [build_lifecycle_event(
                LifecycleType.STEP, LifecycleEventKind.FAILED,
                unit_id=event.step.id, reason="step_failed", source=event,
            )]
        return []

    # --- §4.3 tool（源：ToolEvent + artifact.outcome variant；unit_id=tool_call_id） ---
    def _project_tool(self, event: ToolEvent) -> List[LifecycleEvent]:
        if event.status == ToolEventStatus.CALLING:
            return [build_lifecycle_event(
                LifecycleType.TOOL, LifecycleEventKind.STARTED,
                unit_id=event.tool_call_id, source=event,
            )]
        if event.status == ToolEventStatus.RUNNING:
            return [build_lifecycle_event(
                LifecycleType.TOOL, LifecycleEventKind.PROGRESS,
                unit_id=event.tool_call_id, source=event,
            )]
        if event.status != ToolEventStatus.CALLED:
            return []

        variant, reason_type = self._resolve_tool_outcome(event)
        if variant == "asked":
            # Asked 走 ToolConfirmationEvent 独立通道（tool_event_envelope_v1.py:277-281 同源纪律）
            return []
        if variant == "allow_success":
            kind, reason = LifecycleEventKind.COMPLETED, None
        elif variant == "allow_error":
            kind = LifecycleEventKind.FAILED
            reason = "tool_timeout" if reason_type == "timeout" else "tool_error"
        elif variant == "denied":
            # 用户确认拒绝与策略拒绝均此形态（R8#P3b）；取消非失败
            kind, reason = LifecycleEventKind.CANCELLED, "denied"
        elif variant == "passthrough":
            kind, reason = LifecycleEventKind.COMPLETED, "passthrough"
        else:
            logger.warning(
                "tool lifecycle: unresolvable CALLED outcome (tool_call_id=%s) — skip",
                event.tool_call_id,
            )
            return []
        return [build_lifecycle_event(
            LifecycleType.TOOL, kind, unit_id=event.tool_call_id,
            reason=reason, source=event,
        )]

    @staticmethod
    def _resolve_tool_outcome(event: ToolEvent) -> "tuple[Optional[str], Optional[str]]":
        """(variant, allow_error 的 reason.type)。artifact dict 懒读 + try/except 兜底
        （镜像 event.py:184-189 注释的消费纪律）；artifact 缺失回退 legacy
        function_result.success（镜像 tool_event_envelope_v1.py:308 判别）。"""
        artifact = event.artifact
        if isinstance(artifact, dict):
            try:
                outcome = artifact.get("outcome")
                if isinstance(outcome, dict):
                    variant = outcome.get("variant")
                    if isinstance(variant, str):
                        reason = outcome.get("reason")
                        rtype = reason.get("type") if isinstance(reason, dict) else None
                        return variant, rtype
            except Exception:  # noqa: BLE001 — 畸形 artifact 不得让投影 raise
                pass
        fr = event.function_result
        if fr is not None:
            return ("allow_success" if getattr(fr, "success", False) else "allow_error"), None
        return None, None

    # --- §4.5 subagent（unit_id=child_session_id；coordinator 管线） ---

    _REDUCE_OUTCOME_MAP = {
        "success": (LifecycleEventKind.COMPLETED, None),
        "failed": (LifecycleEventKind.FAILED, "worker_failed"),
        "timed_out": (LifecycleEventKind.FAILED, "watchdog_timeout"),
        "needs_authorization": (LifecycleEventKind.FAILED, "needs_authorization"),
        "cancelled": (LifecycleEventKind.CANCELLED, None),
    }

    def _subagent_correlation(self, work_unit_id, coordinator_run_id) -> LifecycleCorrelationV1:
        return LifecycleCorrelationV1(
            work_unit_id=work_unit_id, coordinator_run_id=coordinator_run_id,
        )

    def _project_worker_spawned(self, event: CoordinatorWorkerSpawnedEvent) -> List[LifecycleEvent]:
        if not event.child_session_id:
            logger.warning("subagent lifecycle: spawned event missing child_session_id — skip")
            return []
        return [build_lifecycle_event(
            LifecycleType.SUBAGENT, LifecycleEventKind.STARTED,
            unit_id=event.child_session_id,
            parent_unit_id=event.parent_session_id or self._parent_session_id,
            correlation=self._subagent_correlation(event.work_unit_id, event.coordinator_run_id),
            source=event,
        )]

    async def _resolve_children(self, coordinator_run_id: Optional[str]) -> Dict[str, str]:
        """每个 source 事件最多一次批查（R4#3）；失败/缺 run_id → 空 map + log（R3#2 合法降级）。"""
        if not coordinator_run_id or self._child_lookup is None:
            logger.warning("subagent lifecycle: no coordinator_run_id/lookup — skip expansion")
            return {}
        try:
            return await self._child_lookup(coordinator_run_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "subagent lifecycle: child lookup failed (run=%s) — skip expansion",
                coordinator_run_id, exc_info=True,
            )
            return {}

    async def _project_reduce(self, event: CoordinatorReduceEvent) -> List[LifecycleEvent]:
        # GroupOutcome（CONFLICT/INCOMPLETE/MIXED/…）不映射 per-child（R4#6）；
        # 展开只由 per_worker_outcomes 驱动——INCOMPLETE 下无 outcome 的 child
        # 停 started（§1 局限 1，文档化不硬造）。
        if not event.per_worker_outcomes:
            return []
        children = await self._resolve_children(event.coordinator_run_id)
        out: List[LifecycleEvent] = []
        for wu_id, outcome in event.per_worker_outcomes.items():
            child_id = children.get(wu_id)
            if not child_id:
                logger.warning("subagent lifecycle: no child row for wu=%s — skip", wu_id)
                continue
            kind, reason = self._REDUCE_OUTCOME_MAP.get(
                getattr(outcome, "value", str(outcome)),
                (LifecycleEventKind.FAILED, "unknown_terminal_outcome"),
            )
            detail = None
            if reason in ("needs_authorization", "unknown_terminal_outcome"):
                detail = LifecycleDetailV1(original_outcome=getattr(outcome, "value", str(outcome)))
            out.append(build_lifecycle_event(
                LifecycleType.SUBAGENT, kind, unit_id=child_id,
                reason=reason, detail=detail,
                parent_unit_id=event.parent_session_id or self._parent_session_id,
                correlation=self._subagent_correlation(wu_id, event.coordinator_run_id),
                source=event,
            ))
        return out

    async def _project_sibling_cancel(self, event: CoordinatorSiblingCancelEvent) -> List[LifecycleEvent]:
        # F17：事件不携带 child_session_id/work_unit_id（业务载荷是 wu id 列表）——
        # 经咽喉 repo 反查投影（R10#A1 定案，取代 R2 时代源处投影旧案）
        if not event.cancelled_work_unit_ids:
            return []
        children = await self._resolve_children(event.coordinator_run_id)
        out: List[LifecycleEvent] = []
        for wu_id in event.cancelled_work_unit_ids:
            child_id = children.get(wu_id)
            if not child_id:
                logger.warning("subagent lifecycle: no child row for cancelled wu=%s — skip", wu_id)
                continue
            out.append(build_lifecycle_event(
                LifecycleType.SUBAGENT, LifecycleEventKind.CANCELLED,
                unit_id=child_id, reason="sibling_cancel",
                parent_unit_id=event.parent_session_id or self._parent_session_id,
                correlation=self._subagent_correlation(wu_id, event.coordinator_run_id),
                source=event,
            ))
        return out
