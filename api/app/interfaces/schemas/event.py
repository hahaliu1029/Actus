from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Self, Type, Union, get_args

from app.domain.models.event import (
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
    CoordinatorApplyEvent,
    CoordinatorDispatchEvent,
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    CoordinatorWorkerSpawnedEvent,
    Event,
    HealthEvent,
    HealthStatus,
    PlanEvent,
    SandboxStateChangedEvent,
    SessionModeChangedEvent,
    StepEvent,
    ToolConfirmationEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.file import File
from app.domain.models.mailbox_envelope import CostAggregate
from app.domain.models.plan import ExecutionStatus
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.services.tools.tool_source_resolver import ToolSource


class BaseEventData(BaseModel):
    """基础事件数据"""

    event_id: Optional[str] = None  # 事件id
    created_at: datetime = Field(default_factory=datetime.now)  # 事件时间
    # B3-core PR-1 §3.3 — producer-side monotonic cursor stamped via INCR session:seq:{sid}.
    # Optional/None for legacy events that pre-date PR-1.
    seq: Optional[int] = None

    # pydantic v2写法，序列化时将datetime转换为时间戳
    model_config = ConfigDict(json_encoders={datetime: lambda v: int(v.timestamp())})

    @classmethod
    def base_event_data(cls, event: Event) -> Dict[str, Any]:
        """类方法，用于将事件Domain模型转换成基础事件数据字典"""
        return {
            "event_id": event.id,
            "created_at": int(event.created_at.timestamp()),
            # B3-core PR-1 §3.3 — propagate seq from domain event to SSE wire envelope.
            "seq": getattr(event, "seq", None),
        }

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """从事件Domain模型中构建基础事件数据.

        ``base_event_data`` already supplies ``event_id``/``created_at``/``seq``;
        the second spread must exclude the same domain-level fields to avoid
        ``TypeError: multiple values for keyword argument`` (B3-core PR-1 §3.3
        added ``seq`` to both sides of the spread).
        """
        return cls(
            **cls.base_event_data(event),
            **event.model_dump(mode="json", exclude={"id", "type", "created_at", "seq"}),
        )


class BaseSSEEvent(BaseModel):
    """基础流式事件数据类型"""

    event: str  # 事件类型
    data: BaseEventData  # 数据

    def to_sse_data_json(self) -> str:
        """默认 SSE 序列化. 子类若需要特殊 wire 处理可 override."""
        return self.data.model_dump_json()

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """将事件Domain模型转换成基础流式事件"""
        # 1.获取事件数据的类型，如果没有则使用基础事件数据BaseEventData
        data_class: Type[BaseEventData] = cls.__annotations__.get("data", BaseEventData)

        # 2.调用构造函数完成初始化
        return cls(
            event=event.type,
            data=data_class.from_event(event),
        )


class CommonEventData(BaseEventData):
    """通用事件数据，让结构允许填充额外的数据"""

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: int(v.timestamp()),
        },
        extra="allow",
    )


class CommonSSEEvent(BaseSSEEvent):
    """通用事件"""

    event: str
    data: CommonEventData


class MessageEventData(BaseEventData):
    """消息事件数据"""

    role: Literal["user", "assistant", "system"] = "assistant"
    message: str = ""
    stream_id: Optional[str] = None
    partial: bool = False
    attachments: List[File] = Field(default_factory=list)


class MessageSSEEvent(BaseSSEEvent):
    """流式消息事件数据响应结构"""

    event: Literal["message"] = "message"
    data: MessageEventData

    @classmethod
    def from_event(cls, event: Event) -> Self:
        return cls(
            data=MessageEventData(
                **BaseEventData.base_event_data(event),
                role=event.role,
                message=event.message,
                stream_id=event.stream_id,
                partial=event.partial,
                attachments=event.attachments,
            )
        )


class TitleEventData(BaseEventData):
    """标题事件数据"""

    title: str


class TitleSSEEvent(BaseSSEEvent):
    """标题流式事件"""

    event: Literal["title"] = "title"
    data: TitleEventData


class StepEventData(BaseEventData):
    """步骤事件数据"""

    id: str  # 步骤id
    status: ExecutionStatus  # 步骤执行状态
    description: str  # 步骤描述


class StepSSEEvent(BaseSSEEvent):
    """步骤流式事件"""

    event: Literal["step"] = "step"
    data: StepEventData

    @classmethod
    def from_event(cls, event: StepEvent) -> Self:
        return cls(
            data=StepEventData(
                **BaseEventData.base_event_data(event),
                status=event.step.status,
                id=event.step.id,
                description=event.step.description,
            )
        )


class PlanEventData(BaseEventData):
    """计划事件数据"""

    steps: List[StepEventData]


class PlanSSEEvent(BaseSSEEvent):
    """计划流式事件"""

    event: Literal["plan"] = "plan"
    data: PlanEventData

    @classmethod
    def from_event(cls, event: PlanEvent) -> Self:
        return cls(
            data=PlanEventData(
                **BaseEventData.base_event_data(event),
                steps=[
                    StepEventData(
                        **BaseEventData.base_event_data(event),
                        id=step.id,
                        status=step.status,
                        description=step.description,
                    )
                    for step in event.plan.steps
                ],
            )
        )


ToolStatusV1 = Literal[
    "ok",          # outcome: success
    "error",       # outcome: error (reason.type="exception")
    "denied",      # outcome: denied by policy
    "timeout",     # outcome: error (reason.type="timeout")
    "passthrough", # outcome: passthrough multimodal
    # 注意：没有 "asked" — asked outcome 走独立 ToolConfirmationEvent
]


class DecisionReasonWire(BaseModel):
    """CS3 wire-shape projection of DecisionReason.

    Projector 会发出以下 7 个 type 值：
    - domain 6 值 (R2 已定义): approval_policy, smart_approve, ast_validator,
      risk_enforce, exception, timeout
    - wire-only fallback 1 值: unknown_variant (projector 的
      _project_unknown_variant_fallback 产出)

    type 字段有意保留为开放 str（不是 Literal），以便未来 R2 新增
    reason 类型时老 server 反序列化不硬抛。前端 tolerant reader
    遇到 7 值以外的 type 降级显示 "unknown" 徽章（I-R4.5）。
    """
    type: str
    code: str = ""
    message: str = ""

    model_config = ConfigDict(extra="forbid")


class FunctionResultV1(BaseModel):
    """扁平 function_result, projector 从 tool artifact 的 outcome 投影而来."""
    status: ToolStatusV1
    message: str = ""
    data: Any = None
    retryable: bool = False
    user_action_required: bool = False  # v1 envelope 恒 False（asked outcome 走 tool_confirmation）
    reason: Optional[DecisionReasonWire] = None
    # passthrough 多模态 blocks, 严格 by_alias wire format:
    # [{"type": "image_url", "image_url": {...}}, {"type": "file", "file": {...}}]
    result_blocks: Optional[list[dict[str, Any]]] = None

    model_config = ConfigDict(extra="forbid")


class ToolEventEnvelopeV1(BaseEventData):
    """CS3 contract surface v1. 冻结 frontend / Flutter / N2 SSE / B10 / B12 读取路径的 wire shape.

    Wire 字段名向后兼容（F1 fix）: 通过 Pydantic Field(alias=...) 保留现有 wire
    短名 name / function / args, Python 属性名改为描述性长名便于代码可读.
    by_alias=True 序列化时出 wire 仍是 name / function / args, 现有前端零改.

    Versioning 术语区分 (Round 2c 厘清):
    - release bump (项目 semver/CHANGELOG): 改本类字段集都是 release 层面的 minor
      或 major, 和 envelope_version 无关.
    - envelope_version bump (本字段的数值变化): 只在 breaking 改动时触发.

    字段集演化规则:
    - Additive optional 字段新增 (*: Optional[T] = None 到 ToolEventEnvelopeV1
      或 FunctionResultV1): release minor, envelope_version 保持 1.
    - Breaking 改动 (必填字段新增 / 字段重命名 / 字段删除 / Literal 值域收缩 /
      字段类型变更): release major, envelope_version bump 到 2.

    envelope_version 类型用 int + validator (>=1), 不用 Literal[1],
    以便未来 v2 事件流经老 server 时 Pydantic 不硬抛.
    """
    envelope_version: int = 1

    tool_call_id: str
    tool_name: str = Field(alias="name")
    tool_source: Optional[ToolSource] = None
    function_name: str = Field(alias="function")
    function_args: dict[str, Any] = Field(alias="args")
    status: Literal["calling", "running", "called"]
    activity_description: str = ""
    display_icon: Optional[str] = None
    render_style: Optional[
        Literal["text", "code", "table", "image", "document"]
    ] = None
    media_type: Optional[str] = None

    function_result: Optional[FunctionResultV1] = None
    content: Optional[dict[str, Any]] = None

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    @field_validator("envelope_version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"envelope_version must be >= 1, got {v}")
        return v


class ToolEventData(BaseEventData):
    """工具事件数据"""

    tool_call_id: str  # 工具调用id
    name: str  # 工具箱名字
    status: ToolEventStatus  # 工具状态
    function: str  # 工具名字
    args: Dict[str, Any]  # 工具参数
    content: Optional[Any] = None  # 工具调用结果


class ToolSSEEvent(BaseSSEEvent):
    """工具流式事件 (R4 after).

    I-R4.6: wire 序列化必须用 by_alias=True 输出短字段名 (name/function/args).
    """

    event: Literal["tool"] = "tool"
    data: ToolEventEnvelopeV1

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_event(cls, event: ToolEvent) -> "ToolSSEEvent":
        from app.application.services.tool_event_envelope_v1 import (
            project_tool_event_to_envelope_v1,
        )
        return cls(data=project_tool_event_to_envelope_v1(event))

    def to_sse_data_json(self) -> str:
        """SSE 序列化入口: 走 by_alias=True 保留 wire 短名 (I-R4.6)."""
        return self.data.model_dump_json(by_alias=True, exclude_none=False)


class DoneEventData(BaseEventData):
    """结束事件数据"""

    metrics: Optional[Dict[str, Any]] = None


class DoneSSEEvent(BaseSSEEvent):
    """停止流式事件"""

    event: Literal["done"] = "done"
    data: DoneEventData

    @classmethod
    def from_event(cls, event) -> Self:
        return cls(
            data=DoneEventData(
                **BaseEventData.base_event_data(event),
                metrics=getattr(event, "metrics", None),
            )
        )


class FinishingSSEEvent(BaseSSEEvent):
    """FINISHING 流式事件"""

    event: Literal["finishing"] = "finishing"


class WaitSSEEvent(BaseSSEEvent):
    """等待人类输入流式事件"""

    event: Literal["wait"] = "wait"
    data: CommonEventData


class ControlEventData(BaseEventData):
    """接管控制事件数据"""

    action: ControlAction
    scope: Optional[ControlScope] = None
    source: ControlSource
    reason: Optional[str] = None
    handoff_mode: Optional[str] = None
    request_status: Optional[str] = None
    takeover_id: Optional[str] = None
    expires_at: Optional[int] = None


class ControlSSEEvent(BaseSSEEvent):
    """接管控制流式事件"""

    event: Literal["control"] = "control"
    data: ControlEventData

    @classmethod
    def from_event(cls, event: ControlEvent) -> Self:
        expires_at = (
            int(event.expires_at.timestamp()) if event.expires_at is not None else None
        )
        return cls(
            data=ControlEventData(
                **BaseEventData.base_event_data(event),
                action=event.action,
                scope=event.scope,
                source=event.source,
                reason=event.reason,
                handoff_mode=event.handoff_mode,
                request_status=event.request_status,
                takeover_id=event.takeover_id,
                expires_at=expires_at,
            )
        )


class ToolConfirmationEventData(BaseEventData):
    """危险工具确认请求事件数据"""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    risk_level: str
    risk_reason: str
    matched_patterns: list[str]
    suggested_alternative: str | None = None
    approval_options: list[str] = ["once", "session", "always", "deny"]
    timeout_seconds: int


class ToolConfirmationSSEEvent(BaseSSEEvent):
    """危险工具确认请求流式事件"""

    event: Literal["tool_confirmation"] = "tool_confirmation"
    data: ToolConfirmationEventData

    @classmethod
    def from_event(cls, event: ToolConfirmationEvent) -> "ToolConfirmationSSEEvent":
        return cls(
            data=ToolConfirmationEventData(
                **BaseEventData.base_event_data(event),
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                tool_args=event.tool_args,
                risk_level=event.risk_level,
                risk_reason=event.risk_reason,
                matched_patterns=event.matched_patterns,
                suggested_alternative=event.suggested_alternative,
                approval_options=event.approval_options,
                timeout_seconds=event.timeout_seconds,
            )
        )


class ErrorEventData(BaseEventData):
    """错误事件数据"""

    error: str


class ErrorSSEEvent(BaseSSEEvent):
    """错误流式事件"""

    event: Literal["error"] = "error"
    data: ErrorEventData


class HealthEventData(BaseEventData):
    """执行健康状态事件数据"""

    status: HealthStatus
    reason: str
    last_node: Optional[str] = None
    idle_seconds: Optional[float] = None
    tool_failures: int = 0
    action: str = "monitoring"
    metrics: Optional[Dict[str, Any]] = None


class HealthSSEEvent(BaseSSEEvent):
    """执行健康状态流式事件"""

    event: Literal["health"] = "health"
    data: HealthEventData

    @classmethod
    def from_event(cls, event: HealthEvent) -> Self:
        return cls(
            data=HealthEventData(
                **BaseEventData.base_event_data(event),
                status=event.status,
                reason=event.reason,
                last_node=event.last_node,
                idle_seconds=event.idle_seconds,
                tool_failures=event.tool_failures,
                action=event.action,
                metrics=event.metrics,
            )
        )


class SandboxStateChangedEventData(BaseEventData):
    """Sandbox 绑定状态变更事件数据"""

    old_state: str
    new_state: str
    generation: int
    sandbox_id: Optional[str] = None
    reason: Optional[str] = None


class SandboxStateChangedSSEEvent(BaseSSEEvent):
    """Sandbox 状态变更流式事件"""

    event: Literal["sandbox_state_changed"] = "sandbox_state_changed"
    data: SandboxStateChangedEventData

    @classmethod
    def from_event(cls, event: SandboxStateChangedEvent) -> Self:
        return cls(
            data=SandboxStateChangedEventData(
                **BaseEventData.base_event_data(event),
                old_state=event.old_state,
                new_state=event.new_state,
                generation=event.generation,
                sandbox_id=event.sandbox_id,
                reason=event.reason,
            )
        )


class SessionModeChangedEventData(BaseEventData):
    """A4-0 unified control-mode-changed payload."""

    to: str
    from_mode: Optional[str] = None
    reason: str
    mode_revision: Optional[int] = None


class SessionModeChangedSSEEvent(BaseSSEEvent):
    """A4-0 control-mode-changed流式事件."""

    event: Literal["session_mode_changed"] = "session_mode_changed"
    data: SessionModeChangedEventData


# ---------------------------------------------------------------------------
# [C2 PR-8 §13] Coordinator SSE events.
#
# Each domain-side ``Coordinator*Event`` composes ``CoordinatorLineageMixin``,
# which carries five optional lineage tags. The matching EventData subclasses
# below declare every domain-event field (lineage + event-specific payload) so
# ``BaseEventData.from_event`` — which spreads
# ``event.model_dump(..., exclude={"id","type","created_at","seq"})`` into the
# data constructor — can populate them without ``TypeError``.
#
# Field-type choices are deliberately JSON-safe: enums are declared as ``str``
# (model_dump(mode="json") emits enum ``.value``), nested ``CostAggregate``
# stays typed because Pydantic re-validates dicts back into the model.
# ---------------------------------------------------------------------------


class CoordinatorDispatchEventData(BaseEventData):
    """[C2 PR-8 §13.3] Coordinator dispatch payload."""

    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    work_unit_id: Optional[str] = None
    step_id: str
    work_unit_count: int
    work_unit_ids: List[str]
    phases: List[str]


class CoordinatorDispatchSSEEvent(BaseSSEEvent):
    """[C2 PR-8 §13.3] Coordinator dispatch SSE event."""

    event: Literal["coordinator_dispatch"] = "coordinator_dispatch"
    data: CoordinatorDispatchEventData


class CoordinatorWorkerSpawnedEventData(BaseEventData):
    """[C2 PR-8 §13.3] Worker-spawned payload."""

    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    work_unit_id: Optional[str] = None
    objective: str
    phase: str
    allowed_tools: List[str]
    write_lease_count: int


class CoordinatorWorkerSpawnedSSEEvent(BaseSSEEvent):
    """[C2 PR-8 §13.3] Worker-spawned SSE event."""

    event: Literal["coordinator_worker_spawned"] = "coordinator_worker_spawned"
    data: CoordinatorWorkerSpawnedEventData


class CoordinatorReduceEventData(BaseEventData):
    """[C2 PR-8 §13.3] Reduce payload.

    ``group_outcome`` and per-worker outcome values arrive as enum ``.value``
    strings from ``model_dump(mode="json")``; ``cost_total`` round-trips back
    into ``CostAggregate`` via Pydantic validation.
    """

    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    work_unit_id: Optional[str] = None
    group_outcome: str
    per_worker_outcomes: Dict[str, str]
    diagnostics_summary: str
    conflict_paths: List[str] = Field(default_factory=list)
    cost_total: CostAggregate


class CoordinatorReduceSSEEvent(BaseSSEEvent):
    """[C2 PR-8 §13.3] Reduce SSE event."""

    event: Literal["coordinator_reduce"] = "coordinator_reduce"
    data: CoordinatorReduceEventData


class CoordinatorApplyEventData(BaseEventData):
    """[C2 PR-8 §13.3] Patch-apply progress payload."""

    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    work_unit_id: Optional[str] = None
    apply_status: str
    file_count: int = 0
    total_bytes: int = 0
    failed_at_path: Optional[str] = None
    rollback_status: Optional[str] = None


class CoordinatorApplySSEEvent(BaseSSEEvent):
    """[C2 PR-8 §13.3] Patch-apply SSE event."""

    event: Literal["coordinator_apply"] = "coordinator_apply"
    data: CoordinatorApplyEventData


class CoordinatorSiblingCancelEventData(BaseEventData):
    """[C2 PR-8 §13.3] Sibling-cancel payload."""

    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    work_unit_id: Optional[str] = None
    triggered_by_work_unit_id: str
    triggered_by_outcome: str
    cancelled_work_unit_ids: List[str]
    reason: str


class CoordinatorSiblingCancelSSEEvent(BaseSSEEvent):
    """[C2 PR-8 §13.3] Sibling-cancel SSE event."""

    event: Literal["coordinator_sibling_cancel"] = "coordinator_sibling_cancel"
    data: CoordinatorSiblingCancelEventData


# 定义Agent流式事件类型集合
AgentSSEEvent = Union[
    CommonSSEEvent,
    MessageSSEEvent,
    TitleSSEEvent,
    StepSSEEvent,
    PlanSSEEvent,
    ToolSSEEvent,
    DoneSSEEvent,
    FinishingSSEEvent,
    ErrorSSEEvent,
    WaitSSEEvent,
    ControlSSEEvent,
    HealthSSEEvent,
    ToolConfirmationSSEEvent,
    SandboxStateChangedSSEEvent,
    SessionModeChangedSSEEvent,
    CoordinatorDispatchSSEEvent,
    CoordinatorWorkerSpawnedSSEEvent,
    CoordinatorReduceSSEEvent,
    CoordinatorApplySSEEvent,
    CoordinatorSiblingCancelSSEEvent,
]


@dataclass
class EventMapping:
    """事件映射数据类，用于存储事件映射信息，涵盖流式事件类型、数据类、事件类型字符串"""

    sse_event_class: Type[BaseSSEEvent]
    data_class: Type[BaseEventData]
    event_type: str


class EventMapper:
    """事件映射类，利用Python自身提供的自省机制，将业务逻辑中的Event转换成适合流式传输的AgentSSEEvent"""

    # 缓存映射(type: EventMapping)
    _cache_mapping: Optional[Dict[str, EventMapping]] = None

    @staticmethod
    def _get_event_type_mapping() -> Dict[str, EventMapping]:
        """通过反射动态构建从事件类型字符串到AgentSSEEvent的映射"""
        # 1.判断缓存映射是否存在，如果存在则直接返回
        if EventMapper._cache_mapping is not None:
            return EventMapper._cache_mapping

        # 2.获取AgentSSEEvent的所有可能存在类
        sse_event_classes = get_args(AgentSSEEvent)
        mapping = {}

        # 3.循环遍历AgentSSEEvent可能的所有类逐个处理
        for sse_event_class in sse_event_classes:
            # 4.跳过基类
            if sse_event_class == BaseSSEEvent:
                continue

            # 5.检查类是否包含event属性
            if (
                hasattr(sse_event_class, "__annotations__")
                and "event" in sse_event_class.__annotations__
            ):
                # 6.提取事件字段
                event_field = sse_event_class.__annotations__["event"]

                # 7.提取事件的具体值(Literal的值)
                if hasattr(event_field, "__args__") and len(event_field.__args__) > 0:
                    event_type = event_field.__args__[0]

                    # 8.提取sse的载荷数据
                    data_class = None
                    if (
                        hasattr(sse_event_class, "__annotations__")
                        and "data" in sse_event_class.__annotations__
                    ):
                        data_class = sse_event_class.__annotations__["data"]

                    # 9.构建并注册映射关系
                    mapping[event_type] = EventMapping(
                        sse_event_class=sse_event_class,
                        data_class=data_class,
                        event_type=event_type,
                    )

        # 10.更新类级缓存
        EventMapper._cache_mapping = mapping
        return mapping

    @staticmethod
    def event_to_sse_event(event: Event) -> AgentSSEEvent:
        """将领域事件转换为Agent流式事件模型"""
        # 1.获取事件映射表
        event_type_mapping = EventMapper._get_event_type_mapping()

        # 2.根据传递进来的事件获取映射类
        event_mapping = event_type_mapping.get(event.type)

        # 3.如果找到了类型映射则进行转换
        if event_mapping:
            sse_event = event_mapping.sse_event_class.from_event(event)
            return sse_event

        # 4.如果没找到类型则使用通用类型
        return CommonSSEEvent.from_event(event)

    @staticmethod
    def events_to_sse_events(events: List[Event]) -> List[AgentSSEEvent]:
        """将领域事件模型列表转换为SSE流式事件列表"""
        return list(
            filter(
                lambda x: x is not None,
                [EventMapper.event_to_sse_event(event) for event in events],
            )
        )
