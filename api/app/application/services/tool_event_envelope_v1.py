"""R4 CS3 ToolEventEnvelope v1 projector.

唯一 ToolEvent (domain) → ToolEventEnvelopeV1 (interfaces wire) 转换点.

See docs/superpowers/specs/2026-04-16-r4-cs3-tool-event-envelope-v1-design.md
§Projector 设计 for full contract.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import ValidationError

from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    MultimodalPayload,
    Passthrough,
    TOOL_ARTIFACT_ADAPTER,
    ToolArtifact,
    ToolOutcome,
    ToolResult,
)
from app.interfaces.schemas.event import (
    DecisionReasonWire,
    FunctionResultV1,
    ToolEventEnvelopeV1,
    ToolStatusV1,
)

logger = logging.getLogger(__name__)


# ============================================================
# Safe variant extractor for malformed artifact fallback
# ============================================================


def _safe_extract_variant(raw: Any) -> str:
    """从可能 malformed 的 artifact dict 中安全提取 variant 名.

    不假设 raw 是 dict, 也不假设 raw["outcome"] 是 dict.
    任意层失败返回 "unknown", 不抛 AttributeError.
    """
    if not isinstance(raw, dict):
        return "unknown"
    outcome = raw.get("outcome")
    if not isinstance(outcome, dict):
        return "unknown"
    variant = outcome.get("variant", "unknown")
    return variant if isinstance(variant, str) and variant else "unknown"


# ============================================================
# DecisionReason → wire form
# ============================================================


def _wire_reason(reason: Optional[DecisionReason]) -> Optional[DecisionReasonWire]:
    """透明透传 reason.type/code/message 到 wire, 不做值变换.

    未知 type 的降级由前端 tolerant reader 处理 (I-R4.5 / R-1 缓解),
    projector 层不 rewrite reason.type.
    """
    if reason is None:
        return None
    return DecisionReasonWire(
        type=reason.type,
        code=reason.code,
        message=reason.message,
    )


# ============================================================
# Shared top-field builder (DRY, F2A fix)
# ============================================================


def _project_common_top_fields(
    event: ToolEvent,
    status_lit: Literal["calling", "called"],
) -> dict[str, Any]:
    """顶层 envelope metadata 字段, 所有路径共用. 用长名 + populate_by_name."""
    return {
        "event_id": event.id,
        "created_at": event.created_at,
        # B3-core PR-1 §3.3 — propagate seq from domain event to wire envelope.
        # ToolEventEnvelopeV1 inherits seq from BaseEventData; without this line
        # the projector silently drops it and the SSE wire payload misses seq.
        "seq": getattr(event, "seq", None),
        "envelope_version": 1,
        "tool_call_id": event.tool_call_id,
        "tool_name": event.tool_name,              # wire alias → "name"
        "tool_source": event.tool_source,
        "function_name": event.function_name,      # wire alias → "function"
        "function_args": event.function_args,      # wire alias → "args"
        "status": status_lit,
        "activity_description": event.activity_description,
        "display_icon": event.display_icon,
        "render_style": event.render_style,
        "media_type": event.media_type,
        "content": (
            event.tool_content.model_dump(mode="json")
            if event.tool_content else None
        ),
    }


# ============================================================
# Variant-specific helpers (Round 2c completion)
# ============================================================


def _function_result_from_outcome(outcome: ToolOutcome) -> FunctionResultV1:
    """ToolOutcome 5 variant → FunctionResultV1 的分发表实现.

    严格对应 spec §映射表. Asked variant 在调用方 _project_from_artifact
    里已经 AssertionError 拦截过, 所以这里 match 不含 Asked 分支.
    """
    if isinstance(outcome, AllowSuccess):
        return FunctionResultV1(
            status="ok",
            message=outcome.content,
            data=outcome.data,
        )
    if isinstance(outcome, AllowError):
        # AllowError.reason.type is domain-constrained to {"exception", "timeout"}
        # (see tool_result.py:_validate_reason_type), so the else branch is exhaustive.
        status: ToolStatusV1 = (
            "timeout" if outcome.reason.type == "timeout" else "error"
        )
        return FunctionResultV1(
            status=status,
            message=outcome.content,
            data=outcome.data,
            retryable=outcome.retryable,
            reason=_wire_reason(outcome.reason),
        )
    if isinstance(outcome, Denied):
        return FunctionResultV1(
            status="denied",
            message=outcome.content,
            user_action_required=False,    # v1 envelope 恒 False (F3 fix)
            reason=_wire_reason(outcome.reason),
        )
    if isinstance(outcome, Passthrough):
        return FunctionResultV1(
            status="passthrough",
            message=outcome.content,
            result_blocks=[
                b.model_dump(by_alias=True) for b in outcome.data.blocks
            ],
        )
    if isinstance(outcome, Asked):
        raise AssertionError(
            "_function_result_from_outcome: Asked variant should have been "
            "intercepted by caller; Asked flows through ToolConfirmationEvent, "
            "not ToolEventEnvelopeV1."
        )
    # Future variants not yet mapped — caller's unknown_variant fallback handles
    raise AssertionError(
        f"_function_result_from_outcome: unreachable variant "
        f"{type(outcome).__name__}"
    )


# ============================================================
# render_style + media_type derivation helpers (Task 7)
# ============================================================


def _derive_render_style_from_passthrough(
    payload: MultimodalPayload,
) -> tuple[
    Optional[Literal["text", "code", "table", "image", "document"]],
    Optional[str],
]:
    """R4 只覆盖 Passthrough 3 种基础情况 + 混合 tiebreaker, B12 扩展 code/table."""
    if not payload.blocks:
        return (None, None)
    block_kinds = {b.kind for b in payload.blocks}

    # 混合 file+image: file 优先 tiebreaker
    if "file" in block_kinds:
        for b in payload.blocks:
            if b.kind == "file":
                file_data = b.file.file_data
                if file_data.startswith("data:application/pdf"):
                    return ("document", "application/pdf")
                return ("document", None)

    # 纯 image (无 file)
    if "image_url" in block_kinds:
        return ("image", None)

    # 纯 text
    if block_kinds == {"text"}:
        return ("text", None)

    return (None, None)


def _derive_render_style_from_outcome(
    outcome: ToolOutcome,
) -> tuple[
    Optional[Literal["text", "code", "table", "image", "document"]],
    Optional[str],
]:
    """render_style + media_type 推导入口.

    R4 scope: 仅 Passthrough variant 尝试推导, 其他返回 (None, None) 留给 B12.
    """
    if isinstance(outcome, Passthrough):
        return _derive_render_style_from_passthrough(outcome.data)
    return (None, None)


# ============================================================
# Projector branches (4 路径)
# ============================================================


def _project_from_artifact(
    event: ToolEvent,
    artifact: ToolArtifact,
    status_lit: Literal["calling", "called"],
) -> ToolEventEnvelopeV1:
    top = _project_common_top_fields(event, status_lit)
    outcome = artifact.outcome

    if isinstance(outcome, Asked):
        raise AssertionError(
            "projector contract violation: ToolEvent should not carry Asked "
            "variant; Asked flows through ToolConfirmationEvent."
        )

    # I-R4.2 冗余一致性 soft-check (R-8 fix: log.error 降级, 不硬抛)
    if event.tool_source and artifact.tool_source != event.tool_source:
        logger.error(
            "projector: I-R4.2 violation — artifact.tool_source=%r != "
            "event.tool_source=%r; using event.tool_source",
            artifact.tool_source, event.tool_source,
        )

    fr = _function_result_from_outcome(outcome)
    render_style, media_type = _derive_render_style_from_outcome(outcome)
    if render_style and not top.get("render_style"):
        top["render_style"] = render_style
    if media_type and not top.get("media_type"):
        top["media_type"] = media_type

    return ToolEventEnvelopeV1(**top, function_result=fr)


def _project_from_legacy_result(
    event: ToolEvent,
    legacy: ToolResult,
    status_lit: Literal["calling", "called"],
) -> ToolEventEnvelopeV1:
    top = _project_common_top_fields(event, status_lit)
    fr = FunctionResultV1(
        status="ok" if legacy.success else "error",
        message=legacy.message or "",
        data=legacy.data,
    )
    return ToolEventEnvelopeV1(**top, function_result=fr)


def _project_skeleton(
    event: ToolEvent,
    status_lit: Literal["calling", "called"],
) -> ToolEventEnvelopeV1:
    top = _project_common_top_fields(event, status_lit)
    return ToolEventEnvelopeV1(**top, function_result=None)


def _project_unknown_variant_fallback(
    event: ToolEvent,
    variant: str,
    status_lit: Literal["calling", "called"],
) -> ToolEventEnvelopeV1:
    """R2 加新 variant 且 projector 未适配时的 last-resort 降级.

    前端按 reason.type="unknown_variant" 通用 badge 显示. log.warning
    已在调用方发出; 这里只造 envelope 不重复告警.
    """
    top = _project_common_top_fields(event, status_lit)
    fr = FunctionResultV1(
        status="error",
        message=f"[unknown variant: {variant}]",
        reason=DecisionReasonWire(
            type="unknown_variant",
            code=variant,
            message=f"R2 variant {variant!r} not supported by projector",
        ),
    )
    return ToolEventEnvelopeV1(**top, function_result=fr)


# ============================================================
# Main entry
# ============================================================


def project_tool_event_to_envelope_v1(event: ToolEvent) -> ToolEventEnvelopeV1:
    """将 ToolEvent 投影到 CS3 v1 wire envelope.

    优先级：
    1. event.artifact dict (R2 typed via adapter) → 完整映射
    2. event.function_result (legacy ToolResult) → fallback 扁平映射
    3. 都没有 (CALLING 事件) → function_result=None

    Invariant I-R4.3: 本函数是 ToolArtifact → wire envelope 的唯一入口;
    CI AST 扫描锁住.

    Pydantic discriminated union 回放保护 (F2 fix):
    event.artifact 是 dict 而非 typed ToolArtifact, 所以 event log
    反序列化不会在 ToolEvent 层做 union dispatch. Projector 用
    TOOL_ARTIFACT_ADAPTER.validate_python 懒校验 + try/except
    (ValidationError | TypeError | AttributeError), 未知 variant 走
    _project_unknown_variant_fallback, 不会把整条 SSE 流炸掉.

    event_id / created_at 从 event.id / event.created_at 直接注入
    (BaseEventData 基类字段).
    """
    status_lit: Literal["calling", "called"] = (
        "called" if event.status == ToolEventStatus.CALLED else "calling"
    )

    # --- 路径 1: event.artifact dict (R2 typed via adapter) ---
    if event.artifact is not None:
        try:
            artifact = TOOL_ARTIFACT_ADAPTER.validate_python(event.artifact)
            return _project_from_artifact(event, artifact, status_lit)
        except (ValidationError, TypeError, AttributeError) as e:
            variant = _safe_extract_variant(event.artifact)
            logger.warning(
                "projector: artifact validation failed (variant=%r), "
                "falling back to unknown-variant envelope: %s",
                variant, e,
            )
            return _project_unknown_variant_fallback(event, variant, status_lit)

    # --- 路径 2: legacy function_result ---
    if event.function_result is not None:
        return _project_from_legacy_result(event, event.function_result, status_lit)

    # --- 路径 3: CALLING 或空事件 ---
    return _project_skeleton(event, status_lit)
