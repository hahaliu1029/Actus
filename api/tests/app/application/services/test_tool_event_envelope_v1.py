"""R4 projector unit tests. See spec §Projector 单测."""
from __future__ import annotations

from typing import Any

import pytest

from app.application.services.tool_event_envelope_v1 import (
    _derive_render_style_from_outcome,
    _derive_render_style_from_passthrough,
    _function_result_from_outcome,
    _project_common_top_fields,
    _project_from_artifact,
    _project_from_legacy_result,
    _project_skeleton,
    _project_unknown_variant_fallback,
    _safe_extract_variant,
    _wire_reason,
    project_tool_event_to_envelope_v1,
)
from app.domain.models.event import (
    BrowserToolContent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    FileBlock,
    FilePayload,
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalPayload,
    Passthrough,
    TextBlock,
    TOOL_ARTIFACT_ADAPTER,
    ToolArtifact,
    ToolResult,
)
from app.domain.services.tools.tool_source_resolver import ToolSource
from app.interfaces.schemas.event import DecisionReasonWire


class TestSafeExtractVariant:
    def test_non_dict_input_returns_unknown(self) -> None:
        assert _safe_extract_variant(None) == "unknown"
        assert _safe_extract_variant("str") == "unknown"
        assert _safe_extract_variant(42) == "unknown"
        assert _safe_extract_variant([1, 2]) == "unknown"

    def test_missing_outcome_key_returns_unknown(self) -> None:
        assert _safe_extract_variant({}) == "unknown"
        assert _safe_extract_variant({"tool_call_id": "c1"}) == "unknown"

    def test_outcome_not_dict_returns_unknown(self) -> None:
        assert _safe_extract_variant({"outcome": "ok"}) == "unknown"
        assert _safe_extract_variant({"outcome": None}) == "unknown"

    def test_variant_not_string_returns_unknown(self) -> None:
        assert _safe_extract_variant({"outcome": {"variant": 123}}) == "unknown"
        assert _safe_extract_variant({"outcome": {"variant": None}}) == "unknown"
        assert _safe_extract_variant({"outcome": {}}) == "unknown"
        # Empty string is technically a str but indicates a malformed artifact
        # (no meaningful variant) — projector must treat it as unknown so that
        # dispatch chains in Tasks 6-9 fall through to the fallback branch.
        assert _safe_extract_variant({"outcome": {"variant": ""}}) == "unknown"

    def test_valid_dict_returns_variant(self) -> None:
        assert _safe_extract_variant({"outcome": {"variant": "allow_success"}}) == "allow_success"
        assert _safe_extract_variant({"outcome": {"variant": "partial_success"}}) == "partial_success"


class TestWireReason:
    def test_none_returns_none(self) -> None:
        assert _wire_reason(None) is None

    def test_transparent_passthrough_type_code_message(self) -> None:
        reason = DecisionReason(
            type="approval_policy",
            code="user_denied_session",
            message="User denied tool call for session",
        )
        wire = _wire_reason(reason)
        assert isinstance(wire, DecisionReasonWire)
        assert wire.type == "approval_policy"
        assert wire.code == "user_denied_session"
        assert wire.message == "User denied tool call for session"

    def test_known_type_passed_through_untouched(self) -> None:
        """Projector 对已知 type 不做 rewrite, 原样透传."""
        reason = DecisionReason(type="exception", code="x")
        wire = _wire_reason(reason)
        assert wire is not None
        assert wire.type == "exception"

    def test_future_unknown_type_passed_through_untouched(self) -> None:
        """R-1 缓解: 未来 R2 新增的 type 老 server 必须原样透传 wire, 由前端 unknown badge 处理.

        使用 model_construct 绕过 Pydantic Literal 校验模拟 future R2 type.
        DecisionReasonWire.type 是开放 str 接收端 (I-R4.5), 所以 wire 端不抛.
        """
        reason = DecisionReason.model_construct(type="future_v3_type", code="x", message="")
        wire = _wire_reason(reason)
        assert wire is not None
        assert wire.type == "future_v3_type"


class TestProjectCommonTopFields:
    def _make_event(self, **overrides: Any) -> ToolEvent:
        defaults: dict[str, Any] = dict(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
            status=ToolEventStatus.CALLED,
        )
        defaults.update(overrides)
        return ToolEvent(**defaults)

    def test_minimum_event_all_defaults(self) -> None:
        evt = self._make_event()
        top = _project_common_top_fields(evt, "called")
        assert top["envelope_version"] == 1
        assert top["tool_call_id"] == "c1"
        assert top["tool_name"] == "shell"
        assert top["function_name"] == "shell_execute"
        assert top["function_args"] == {"command": "ls"}
        assert top["status"] == "called"
        assert top["activity_description"] == ""
        assert top["display_icon"] is None
        assert top["render_style"] is None
        assert top["media_type"] is None
        assert top["content"] is None
        assert top["tool_source"] is None

    def test_event_with_tool_content_writes_content_dict(self) -> None:
        evt = self._make_event(tool_content=BrowserToolContent(screenshot="data:image/png;base64,xyz"))
        top = _project_common_top_fields(evt, "called")
        assert top["content"] == {"screenshot": "data:image/png;base64,xyz"}

    def test_event_with_tool_source_passed_through(self) -> None:
        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        evt = self._make_event(tool_source=ts)
        top = _project_common_top_fields(evt, "called")
        assert top["tool_source"] == ts

    def test_event_id_and_created_at_copied(self) -> None:
        evt = self._make_event()
        top = _project_common_top_fields(evt, "calling")
        assert top["event_id"] == evt.id
        assert top["created_at"] == evt.created_at


class TestFunctionResultFromOutcome:
    def test_allow_success_maps_to_ok(self) -> None:
        outcome = AllowSuccess(content="ls output", data={"files": ["a", "b"]})
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "ok"
        assert fr.message == "ls output"
        assert fr.data == {"files": ["a", "b"]}
        assert fr.retryable is False
        assert fr.user_action_required is False
        assert fr.reason is None
        assert fr.result_blocks is None

    def test_allow_error_exception_maps_to_error(self) -> None:
        outcome = AllowError(
            content="connection refused",
            reason=DecisionReason(type="exception", code="ECONNREFUSED"),
            retryable=True,
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "error"
        assert fr.message == "connection refused"
        assert fr.retryable is True
        assert fr.reason is not None
        assert fr.reason.type == "exception"
        assert fr.reason.code == "ECONNREFUSED"

    def test_allow_error_timeout_maps_to_timeout(self) -> None:
        outcome = AllowError(
            content="tool took too long",
            reason=DecisionReason(type="timeout", code="step_timeout_600s"),
            retryable=False,
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "timeout"
        assert fr.reason is not None
        assert fr.reason.type == "timeout"

    def test_denied_ast_validator_user_action_required_false(self) -> None:
        """F3: all Denied → user_action_required=False (envelope v1 恒 False)."""
        outcome = Denied(
            content="command 'rm -rf /' blocked by AST",
            reason=DecisionReason(type="ast_validator", code="blocked_rm_rf_root"),
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "denied"
        assert fr.user_action_required is False
        assert fr.reason is not None
        assert fr.reason.type == "ast_validator"

    def test_denied_approval_policy_user_action_required_false(self) -> None:
        """F3 同上."""
        outcome = Denied(
            content="user denied",
            reason=DecisionReason(type="approval_policy", code="user_denied_session"),
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "denied"
        assert fr.user_action_required is False

    def test_denied_smart_approve_user_action_required_false(self) -> None:
        """F3: all 4 DecisionReason.type Denied paths → user_action_required=False."""
        outcome = Denied(
            content="smart_approve denied",
            reason=DecisionReason(type="smart_approve", code="llm_deny"),
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "denied"
        assert fr.user_action_required is False

    def test_denied_risk_enforce_user_action_required_false(self) -> None:
        """F3 同上: 覆盖第 4 个 Denied reason.type."""
        outcome = Denied(
            content="risk_enforce blocked",
            reason=DecisionReason(type="risk_enforce", code="risk_block"),
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "denied"
        assert fr.user_action_required is False

    def test_passthrough_maps_to_passthrough_with_result_blocks(self) -> None:
        outcome = Passthrough(
            content="[file_view: file_view — 1 image(s) loaded]",
            data=MultimodalPayload(blocks=[
                ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,xyz", detail="auto")),
            ]),
        )
        fr = _function_result_from_outcome(outcome)
        assert fr.status == "passthrough"
        assert fr.result_blocks == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz", "detail": "auto"}},
        ]

    def test_asked_triggers_assertion_error(self) -> None:
        """Asked 走独立 ToolConfirmationEvent, projector 不应收到."""
        outcome = Asked(
            content="confirm deletion",
            reason=DecisionReason(type="approval_policy"),
        )
        with pytest.raises(AssertionError, match="Asked variant should have been intercepted"):
            _function_result_from_outcome(outcome)


class TestDeriveRenderStyle:
    def test_empty_blocks_returns_none_none(self) -> None:
        payload = MultimodalPayload(blocks=[])
        assert _derive_render_style_from_passthrough(payload) == (None, None)

    def test_pure_image_returns_image_no_mime(self) -> None:
        payload = MultimodalPayload(blocks=[
            ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x")),
        ])
        assert _derive_render_style_from_passthrough(payload) == ("image", None)

    def test_pdf_file_returns_document_with_pdf_mime(self) -> None:
        payload = MultimodalPayload(blocks=[
            FileBlock(file=FilePayload(filename="x.pdf", file_data="data:application/pdf;base64,x")),
        ])
        assert _derive_render_style_from_passthrough(payload) == ("document", "application/pdf")

    def test_non_pdf_file_returns_document_no_mime(self) -> None:
        """非 pdf mime 只推断 document, 不猜 media_type (留给 B12)."""
        payload = MultimodalPayload(blocks=[
            FileBlock(file=FilePayload(
                filename="x.docx",
                file_data="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,x",
            )),
        ])
        style, mime = _derive_render_style_from_passthrough(payload)
        assert style == "document"
        assert mime is None

    def test_pure_text_returns_text(self) -> None:
        payload = MultimodalPayload(blocks=[
            TextBlock(text="hello"),
        ])
        assert _derive_render_style_from_passthrough(payload) == ("text", None)

    def test_image_file_mixed_file_wins_tiebreaker(self) -> None:
        """混合 image+file → file 优先 (tiebreaker 规则, spec §render_style)."""
        payload = MultimodalPayload(blocks=[
            ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x")),
            FileBlock(file=FilePayload(filename="x.pdf", file_data="data:application/pdf;base64,x")),
        ])
        style, mime = _derive_render_style_from_passthrough(payload)
        assert style == "document"
        assert mime == "application/pdf"

    def test_explicit_media_type_preferred_for_image(self) -> None:
        """B12 P2: payload.media_type 存在时优先于 block 推导。"""
        payload = MultimodalPayload(
            blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x"))],
            media_type="image/png",
        )
        assert _derive_render_style_from_passthrough(payload) == ("image", "image/png")

    def test_image_without_media_type_unchanged(self) -> None:
        """回归：无 media_type（flag-OFF）→ 保持 ('image', None)。"""
        payload = MultimodalPayload(
            blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x"))],
        )
        assert _derive_render_style_from_passthrough(payload) == ("image", None)

    def test_non_passthrough_outcome_returns_none_none(self) -> None:
        outcome = AllowSuccess(content="ok")
        assert _derive_render_style_from_outcome(outcome) == (None, None)


class TestProjectFromArtifact:
    def _make_event_with_artifact(self, outcome, **overrides):
        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        artifact = ToolArtifact(
            tool_call_id="c1",
            tool_name="shell_execute",
            tool_source=ts,
            outcome=outcome,
        )
        return ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
            status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
            **overrides,
        )

    def test_allow_success_produces_envelope_status_ok(self) -> None:
        evt = self._make_event_with_artifact(AllowSuccess(content="ls output"))
        typed_artifact = TOOL_ARTIFACT_ADAPTER.validate_python(evt.artifact)
        env = _project_from_artifact(evt, typed_artifact, "called")

        assert env.envelope_version == 1
        assert env.function_result is not None
        assert env.function_result.status == "ok"
        assert env.function_result.message == "ls output"

    def test_passthrough_image_populates_render_style_image(self) -> None:
        outcome = Passthrough(
            content="[file_view: 1 image]",
            data=MultimodalPayload(blocks=[
                ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,x")),
            ]),
        )
        evt = self._make_event_with_artifact(outcome)
        typed_artifact = TOOL_ARTIFACT_ADAPTER.validate_python(evt.artifact)
        env = _project_from_artifact(evt, typed_artifact, "called")

        assert env.function_result is not None
        assert env.function_result.status == "passthrough"
        assert env.function_result.result_blocks == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x", "detail": "auto"}},
        ]
        assert env.render_style == "image"
        assert env.media_type is None

    def test_asked_triggers_assertion_error(self) -> None:
        """Asked 不应到达 projector 路径."""
        outcome = Asked(content="confirm?", reason=DecisionReason(type="approval_policy"))
        evt = self._make_event_with_artifact(outcome)
        typed_artifact = TOOL_ARTIFACT_ADAPTER.validate_python(evt.artifact)
        with pytest.raises(AssertionError):
            _project_from_artifact(evt, typed_artifact, "called")

    def test_tool_source_mismatch_event_side_logs_error_does_not_raise(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """I-R4.2 R-8 fix: event-side 不一致 log.error 降级, 不硬抛.

        场景: artifact.tool_source 是真相, event.tool_source 被下游篡改.
        """
        import logging

        evt = self._make_event_with_artifact(AllowSuccess(content="ok"))
        evt.tool_source = ToolSource(source="mcp", category="mcp", canonical_name="x")
        typed_artifact = TOOL_ARTIFACT_ADAPTER.validate_python(evt.artifact)

        caplog.set_level(logging.ERROR)
        env = _project_from_artifact(evt, typed_artifact, "called")
        assert env.envelope_version == 1
        assert any("I-R4.2" in r.message for r in caplog.records)

    def test_tool_source_mismatch_artifact_side_logs_error_does_not_raise(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """I-R4.2 反方向: artifact JSON 一侧漂移 (例: R2 新 variant 落库后
        event.tool_source 仍是旧 routing), 经 adapter validate 后仍触发降级.

        通过修改 evt.artifact dict 再 validate, 模拟事件日志回放时 artifact
        JSON 实际漂移的场景, 比 post-validate setattr 更贴近真实执行路径.
        spec §I-R4.2 要求两种不一致方向都降级 log.error, 不硬抛.
        """
        import logging

        evt = self._make_event_with_artifact(AllowSuccess(content="ok"))
        # event.tool_source 保持 native/shell (Task 8 _make_event_with_artifact 默认)
        # 改 artifact dict 的 tool_source 再 validate, 模拟真实 JSON 漂移
        assert evt.artifact is not None
        evt.artifact["tool_source"] = {
            "source": "a2a", "category": "a2a", "canonical_name": "future_skill",
        }
        typed_artifact = TOOL_ARTIFACT_ADAPTER.validate_python(evt.artifact)

        caplog.set_level(logging.ERROR)
        env = _project_from_artifact(evt, typed_artifact, "called")
        assert env.envelope_version == 1
        assert any("I-R4.2" in r.message for r in caplog.records)


class TestProjectFromLegacyResult:
    def test_legacy_success_maps_to_ok(self) -> None:
        legacy = ToolResult(success=True, message="ok", data={"x": 1})
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED, function_result=legacy,
        )
        env = _project_from_legacy_result(evt, legacy, "called")
        assert env.function_result is not None
        assert env.function_result.status == "ok"
        assert env.function_result.message == "ok"
        assert env.function_result.data == {"x": 1}

    def test_legacy_failure_maps_to_error(self) -> None:
        legacy = ToolResult(success=False, message="failed", data=None)
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED, function_result=legacy,
        )
        env = _project_from_legacy_result(evt, legacy, "called")
        assert env.function_result is not None
        assert env.function_result.status == "error"
        assert env.function_result.message == "failed"


class TestProjectSkeleton:
    def test_calling_event_produces_function_result_none(self) -> None:
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLING,
        )
        env = _project_skeleton(evt, "calling")
        assert env.function_result is None
        assert env.status == "calling"


class TestProjectUnknownVariantFallback:
    def test_unknown_variant_produces_error_envelope(self) -> None:
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED,
        )
        env = _project_unknown_variant_fallback(evt, "partial_success", "called")
        assert env.function_result is not None
        assert env.function_result.status == "error"
        assert "partial_success" in env.function_result.message
        assert env.function_result.reason is not None
        assert env.function_result.reason.type == "unknown_variant"
        assert env.function_result.reason.code == "partial_success"


class TestProjectorMainEntry:
    def test_artifact_path_taken_when_artifact_present(self) -> None:
        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="shell_execute", tool_source=ts,
            outcome=AllowSuccess(content="ok"),
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts, artifact=artifact.model_dump(mode="json", by_alias=True),
        )
        env = project_tool_event_to_envelope_v1(evt)
        assert env.function_result is not None
        assert env.function_result.status == "ok"

    def test_legacy_path_taken_when_only_function_result_present(self) -> None:
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED,
            function_result=ToolResult(success=True, message="ok"),
        )
        env = project_tool_event_to_envelope_v1(evt)
        assert env.function_result is not None
        assert env.function_result.status == "ok"

    def test_skeleton_path_taken_when_calling_and_no_result(self) -> None:
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLING,
        )
        env = project_tool_event_to_envelope_v1(evt)
        assert env.function_result is None
        assert env.status == "calling"

    def test_unknown_variant_dict_fallbacks_to_error_envelope(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """F2 关键: 未知 variant 进 projector 走 ValidationError → fallback, 不炸."""
        import logging

        future_artifact = {
            "tool_call_id": "c1",
            "tool_name": "shell_execute",
            "tool_source": {"source": "native", "category": "shell", "canonical_name": "shell_execute"},
            "outcome": {"variant": "partial_success", "content": "部分"},
        }
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED,
            artifact=future_artifact,
        )
        caplog.set_level(logging.WARNING)
        env = project_tool_event_to_envelope_v1(evt)

        assert env.function_result is not None
        assert env.function_result.status == "error"
        assert env.function_result.reason is not None
        assert env.function_result.reason.type == "unknown_variant"
        assert env.function_result.reason.code == "partial_success"
        assert any(
            "unknown" in r.message.lower() or "fallback" in r.message.lower()
            for r in caplog.records
        )

    def test_malformed_artifact_non_dict_falls_back_safely(self) -> None:
        """R-7 扩: artifact 是 dict 但缺关键字段 → fallback 不抛."""
        broken_artifact = {"foo": "bar"}
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED,
            artifact=broken_artifact,
        )
        env = project_tool_event_to_envelope_v1(evt)
        assert env.function_result is not None
        assert env.function_result.status == "error"
        assert env.function_result.reason is not None
        assert env.function_result.reason.type == "unknown_variant"
        assert env.function_result.reason.code == "unknown"

    def test_envelope_version_always_1(self) -> None:
        """所有路径输出 envelope_version=1."""
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLING,
        )
        env = project_tool_event_to_envelope_v1(evt)
        assert env.envelope_version == 1
