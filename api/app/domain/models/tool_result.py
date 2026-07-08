from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_serializer

from app.domain.services.tools.tool_source_resolver import ToolSource

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """LEGACY — DEPRECATED as of R2 (2026-04-15).

    This model remains for soft coexistence during R2 rollout. NEW code in:
    - api/app/domain/services/tools/**
    - api/app/domain/services/graphs/react_graph.py

    MUST use ToolOutcome (discriminated union) from this same module instead:
        from app.domain.models.tool_result import (
            AllowSuccess, AllowError, Denied, Asked, Passthrough,
            ToolOutcome, ToolArtifact,
        )

    Existing consumers in api/app/application/** remain temporarily allowed
    until R4 / Phase 2 PermissionEngine 落地时统一清理.

    Exit condition: R4 (CS3 ToolEventEnvelope freeze) or Phase 2 PermissionEngine,
    whichever comes first. Not allowed to remain indefinitely.
    """

    success: bool = True  # 是否成功调用
    message: Optional[str] = ""  # 额外的信息提示
    data: Optional[T] = None  # 工具的执行结果/数据

    @classmethod
    def from_sandbox(
        cls, code: int, msg: str, data: Optional[T], **kwargs
    ) -> "ToolResult":
        """将从沙箱中返回的API数据转换成工具结果"""
        return cls(
            success=True if code < 300 else False,
            message=msg,
            data=data,
        )


# ============================================================
# R2 CS2 — ToolStatus taxonomy (new typed union)
# See docs/superpowers/specs/2026-04-15-r2-toolstatus-taxonomy-design.md
# ============================================================


DecisionReasonType = Literal[
    "approval_policy",   # ApprovalStateReader/Writer 审批决策 (grant 持久化)
    "smart_approve",     # summary_llm 产出的 approve/deny/escalate
    "ast_validator",     # N1 pre-execution 静态拒绝
    "risk_enforce",      # SkillTool.risk_mode=enforce_confirmation 内部决策
    "exception",         # wrapper try/except 捕获的程序异常
    "timeout",           # ExecutionWatchdog / wrapper timeout
]


class DecisionReason(BaseModel):
    """CS2 的 2 轴 contract 之一: 决策/失败来源.

    Invariants:
    - `type` 是分支字段, 用于 graph routing / UI badge / 穷举测试
    - `code` 是诊断字段, 禁止 branch on this (只能用于 log / audit / telemetry)
    - `message` 是 human-readable 原因, 用于 UI 展示和 LLM 降级提示
    """

    type: DecisionReasonType
    code: str = ""
    message: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ============================================================
# MultimodalPayload — Passthrough 的 typed artifact
# 贴 Actus 现有 wire format (LangChain content block style)
# 严格 1:1 匹配:
#   image.py:85 ("image_url" + nested {"url", "detail"})
#   pdf.py:152-158 ("file" + nested {"filename", "file_data"})
#   video.py:178-187 (复用 image_url)
# ============================================================


class ImageUrlPayload(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"

    model_config = ConfigDict(extra="forbid")


class ImageUrlBlock(BaseModel):
    """Wire format: {"type": "image_url", "image_url": {"url": "...", "detail": "auto"}}"""
    kind: Literal["image_url"] = Field("image_url", alias="type")
    image_url: ImageUrlPayload

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FilePayload(BaseModel):
    filename: str
    file_data: str  # "data:application/pdf;base64,..." 格式

    model_config = ConfigDict(extra="forbid")


class FileBlock(BaseModel):
    """Wire format: {"type": "file", "file": {"filename": "...", "file_data": "..."}}"""
    kind: Literal["file"] = Field("file", alias="type")
    file: FilePayload

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TextBlock(BaseModel):
    """Wire format: {"type": "text", "text": "..."}"""
    kind: Literal["text"] = Field("text", alias="type")
    text: str

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


MultimodalBlock = Annotated[
    TextBlock | ImageUrlBlock | FileBlock,
    Field(discriminator="kind"),
]


class DocumentThumbnail(BaseModel):
    """B12 P5 文档预览缩略图（首页 presigned URL，非 base64）。"""
    url: str
    media_type: str
    page: int

    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentPreview(BaseModel):
    """B12 P5 结构化文档预览 metadata，挂 MultimodalPayload.document_preview。

    extraction path（有页图）→ thumbnail = 首页 image_url（presigned URL）;
    native path（无页图）→ thumbnail=None（仅 filename/page_count）。
    """
    filename: str
    media_type: str
    page_count: int | None = None
    thumbnail: DocumentThumbnail | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class MultimodalPayload(BaseModel):
    """Passthrough variant 的结构化 artifact payload. blocks 可以为空."""
    blocks: list[MultimodalBlock]
    # B12 P2/P5: optional metadata. None-omitting serializer 保证值为 None 时
    # 不出现在 dump 输出 → flag-OFF 旧 passthrough artifact 结构 identical
    # （INV-B12-1；对照 Asked._omit_none_confirmation_id）。
    media_type: str | None = None
    document_preview: DocumentPreview | None = None

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _omit_none_b12_fields(self, handler: Any) -> dict:
        """Omit B12 optional fields when None. Mirrors Asked (PE-0). handler(self)
        preserves nested block by_alias/mode=json; only pops top-level None keys."""
        data: dict = handler(self)
        if data.get("media_type") is None:
            data.pop("media_type", None)
        if data.get("document_preview") is None:
            data.pop("document_preview", None)
        return data


# ============================================================
# ToolOutcome — 5 变体 discriminated union
# 不含调用上下文 (tool_call_id / tool_name / tool_source),
# 那些在 ToolArtifact 里 (Task 4)
# ============================================================

# CS2.2-2.4 reason.type 白名单. 在变体 field_validator 里强制,
# 防止 wrapper 写出不一致的 (variant, reason) 组合.
_ASKED_ALLOWED_REASON_TYPES: frozenset[str] = frozenset(
    {"approval_policy", "smart_approve", "risk_enforce"}
)
_ALLOW_ERROR_ALLOWED_REASON_TYPES: frozenset[str] = frozenset(
    {"exception", "timeout"}
)
_DENIED_ALLOWED_REASON_TYPES: frozenset[str] = frozenset(
    {"approval_policy", "smart_approve", "ast_validator", "risk_enforce"}
)


class AllowSuccess(BaseModel):
    variant: Literal["allow_success"] = "allow_success"
    content: str                             # LLM-facing
    data: dict[str, Any] | None = None       # structured artifact payload

    model_config = ConfigDict(extra="forbid")


class AllowError(BaseModel):
    variant: Literal["allow_error"] = "allow_error"
    content: str                             # LLM-facing human message
    reason: DecisionReason                    # CS2.3: type ∈ {exception, timeout}
    retryable: bool = False
    data: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason")
    @classmethod
    def _validate_reason_type(cls, v: DecisionReason) -> DecisionReason:
        if v.type not in _ALLOW_ERROR_ALLOWED_REASON_TYPES:
            raise ValueError(
                f"CS2.3 violation: AllowError.reason.type must be one of "
                f"{sorted(_ALLOW_ERROR_ALLOWED_REASON_TYPES)}, got {v.type!r}."
            )
        return v


class Denied(BaseModel):
    variant: Literal["denied"] = "denied"
    content: str                             # LLM-facing explanation
    reason: DecisionReason                    # CS2.4: type ∈ {approval_policy, smart_approve, ast_validator, risk_enforce}

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason")
    @classmethod
    def _validate_reason_type(cls, v: DecisionReason) -> DecisionReason:
        if v.type not in _DENIED_ALLOWED_REASON_TYPES:
            raise ValueError(
                f"CS2.4 violation: Denied.reason.type must be one of "
                f"{sorted(_DENIED_ALLOWED_REASON_TYPES)}, got {v.type!r}."
            )
        return v


class Asked(BaseModel):
    variant: Literal["asked"] = "asked"
    content: str
    reason: DecisionReason                    # CS2.2: type ∈ {approval_policy, smart_approve, risk_enforce}
    # NOT ast_validator (AST is fail-closed deny), NOT exception/timeout (those are AllowError)
    confirmation_id: str | None = None  # PE-0: stable identifier used by
                                        # ConfirmationQueue + frontend
                                        # tool-confirmation-card. None for
                                        # legacy callers — back-compat with
                                        # R2 CS2 golden matrix payloads.

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason")
    @classmethod
    def _validate_reason_type(cls, v: DecisionReason) -> DecisionReason:
        if v.type not in _ASKED_ALLOWED_REASON_TYPES:
            raise ValueError(
                f"CS2.2 violation: Asked.reason.type must be one of "
                f"{sorted(_ASKED_ALLOWED_REASON_TYPES)}, got {v.type!r}. "
                "AST validator must produce Denied (fail-closed deny); "
                "exception/timeout must produce AllowError."
            )
        return v

    @model_serializer(mode="wrap")
    def _omit_none_confirmation_id(self, handler: Any) -> dict:
        """Omit confirmation_id from wire output when None.

        Back-compat with R2/R4 golden fixtures which predate PE-0 and do not
        contain the confirmation_id field.  When confirmation_id is set to a
        real value it is included normally so the frontend tool-confirmation-card
        and ConfirmationQueue can read it.
        """
        data: dict = handler(self)
        if data.get("confirmation_id") is None:
            data.pop("confirmation_id", None)
        return data


class Passthrough(BaseModel):
    variant: Literal["passthrough"] = "passthrough"
    content: str                             # LLM-facing summary text
    data: MultimodalPayload                   # CS2.5: typed, enforced by static type annotation

    model_config = ConfigDict(extra="forbid")


ToolOutcome = Annotated[
    AllowSuccess | AllowError | Denied | Asked | Passthrough,
    Field(discriminator="variant"),
]

# 模块级 singleton adapter. 用途:
#   1. golden matrix JSON 加载
#   2. checkpointer state 里的 ToolArtifact.outcome 字段 (dict) → typed variant
#   3. CS3 R4 envelope 向后兼容反序列化
TOOL_OUTCOME_ADAPTER: TypeAdapter[ToolOutcome] = TypeAdapter(ToolOutcome)


# ============================================================
# ToolArtifact — 调用上下文 + outcome
# 只在 react_graph.tool_node Layer 3 构造
# (wrapper 层不感知 tool_call_id, 由 LangChain StructuredTool 通过 ToolCall dict 传递)
# ============================================================


class ToolArtifact(BaseModel):
    """CS2 call-context wrapper around ToolOutcome.

    Invariant (CS2.6): tool_call_id / tool_name / tool_source 只在这里,
    不在 ToolOutcome 任一变体.

    This is what gets written into `ToolMessage.artifact` field by Layer 3.
    """
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_source: ToolSource
    outcome: ToolOutcome

    model_config = ConfigDict(extra="forbid")


# 同理需要 ToolArtifact 的模块级 TypeAdapter 用于 state 反序列化路径.
# 两级反序列化: 先 ToolArtifact, 其 outcome 字段再用 TOOL_OUTCOME_ADAPTER
TOOL_ARTIFACT_ADAPTER: TypeAdapter[ToolArtifact] = TypeAdapter(ToolArtifact)
