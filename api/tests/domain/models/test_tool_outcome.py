"""Unit tests for R2 CS2 ToolOutcome domain models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.tool_result import (
    TOOL_ARTIFACT_ADAPTER,
    TOOL_OUTCOME_ADAPTER,
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    DecisionReasonType,
    Denied,
    FileBlock,
    FilePayload,
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalBlock,
    MultimodalPayload,
    Passthrough,
    TextBlock,
    ToolArtifact,
    ToolOutcome,
)
from app.domain.services.tools.tool_source_resolver import ToolSource


class TestDecisionReason:
    def test_construct_with_all_fields(self):
        r = DecisionReason(
            type="approval_policy",
            code="session_cache_hit",
            message="User previously approved in this session",
        )
        assert r.type == "approval_policy"
        assert r.code == "session_cache_hit"
        assert r.message == "User previously approved in this session"

    def test_default_code_and_message_are_empty_strings(self):
        r = DecisionReason(type="timeout")
        assert r.code == ""
        assert r.message == ""

    def test_all_6_types_accepted(self):
        expected_types: list[DecisionReasonType] = [
            "approval_policy",
            "smart_approve",
            "ast_validator",
            "risk_enforce",
            "exception",
            "timeout",
        ]
        for t in expected_types:
            DecisionReason(type=t)  # must not raise

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            DecisionReason(type="invalid_type")  # type: ignore[arg-type]

    def test_frozen_model(self):
        r = DecisionReason(type="exception")
        with pytest.raises(ValidationError):
            r.type = "timeout"  # type: ignore[misc]

    def test_round_trip_json(self):
        r = DecisionReason(
            type="smart_approve",
            code="llm_eval_deny",
            message="LLM flagged as high risk",
        )
        j = r.model_dump_json()
        r2 = DecisionReason.model_validate_json(j)
        assert r == r2


class TestMultimodalBlocks:
    def test_text_block_wire_format(self):
        b = TextBlock(text="Hello world")
        dumped = b.model_dump(by_alias=True)
        assert dumped == {"type": "text", "text": "Hello world"}

    def test_image_url_block_wire_format_matches_image_py_line_85(self):
        """Must match api/app/infrastructure/external/file_processors/image.py:85:
        {"type": "image_url", "image_url": {"url": "...", "detail": "auto"}}
        """
        b = ImageUrlBlock(
            image_url=ImageUrlPayload(
                url="data:image/png;base64,iVBOR...",
                detail="auto",
            ),
        )
        dumped = b.model_dump(by_alias=True)
        assert dumped == {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBOR...",
                "detail": "auto",
            },
        }

    def test_file_block_wire_format_matches_pdf_py_lines_152_158(self):
        """Must match api/app/infrastructure/external/file_processors/pdf.py:152-158:
        {"type": "file", "file": {"filename": "...", "file_data": "data:application/pdf;base64,..."}}
        """
        b = FileBlock(
            file=FilePayload(
                filename="report.pdf",
                file_data="data:application/pdf;base64,JVBERi0xLjQ...",
            ),
        )
        dumped = b.model_dump(by_alias=True)
        assert dumped == {
            "type": "file",
            "file": {
                "filename": "report.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0xLjQ...",
            },
        }

    def test_multimodal_payload_with_mixed_blocks(self):
        p = MultimodalPayload(blocks=[
            TextBlock(text="Summary"),
            ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,abc")),
        ])
        assert len(p.blocks) == 2
        assert isinstance(p.blocks[0], TextBlock)
        assert isinstance(p.blocks[1], ImageUrlBlock)

    def test_multimodal_payload_empty_blocks_allowed(self):
        p = MultimodalPayload(blocks=[])
        assert p.blocks == []

    def test_block_round_trip_via_discriminator(self):
        original = MultimodalPayload(blocks=[
            TextBlock(text="hi"),
            ImageUrlBlock(image_url=ImageUrlPayload(url="x")),
            FileBlock(file=FilePayload(filename="a", file_data="b")),
        ])
        dumped = original.model_dump(by_alias=True)
        restored = MultimodalPayload.model_validate(dumped)
        assert restored == original


class TestToolOutcomeVariants:
    def test_allow_success_minimal(self):
        o = AllowSuccess(content="tool output text")
        assert o.variant == "allow_success"
        assert o.content == "tool output text"
        assert o.data is None

    def test_allow_success_with_data(self):
        o = AllowSuccess(content="result", data={"key": "value"})
        assert o.data == {"key": "value"}

    def test_allow_error_with_exception_reason(self):
        o = AllowError(
            content="Error: network timeout",
            reason=DecisionReason(type="timeout", code="tcp_connect_timeout"),
            retryable=True,
        )
        assert o.variant == "allow_error"
        assert o.reason.type == "timeout"
        assert o.retryable is True

    def test_allow_error_rejects_non_error_reasons(self):
        """CS2.3: AllowError.reason.type must be exception or timeout."""
        for invalid_type in (
            "approval_policy",
            "smart_approve",
            "ast_validator",
            "risk_enforce",
        ):
            with pytest.raises(ValidationError):
                AllowError(
                    content="x",
                    reason=DecisionReason(type=invalid_type),
                )

    def test_denied_with_policy_reason(self):
        o = Denied(
            content="Tool denied by approval policy",
            reason=DecisionReason(type="approval_policy", code="always_deny_rule"),
        )
        assert o.variant == "denied"

    def test_denied_with_ast_validator_allowed(self):
        """Denied can carry ast_validator reason (shell fail-closed deny)."""
        o = Denied(
            content="AST validator blocked: rm -rf pattern",
            reason=DecisionReason(type="ast_validator", code="dangerous_rm_pattern"),
        )
        assert o.reason.type == "ast_validator"

    def test_denied_rejects_error_reasons(self):
        """CS2.4: Denied.reason.type must not be exception or timeout."""
        for invalid_type in ("exception", "timeout"):
            with pytest.raises(ValidationError):
                Denied(
                    content="x",
                    reason=DecisionReason(type=invalid_type),
                )

    def test_asked_with_risk_enforce_reason(self):
        o = Asked(
            content="Skill requires user confirmation",
            reason=DecisionReason(type="risk_enforce", code="skill_xyz_risk_high"),
        )
        assert o.variant == "asked"

    def test_asked_rejects_non_policy_reasons(self):
        """CS2.2: Asked.reason.type must be approval_policy/smart_approve/risk_enforce.

        - ast_validator: AST is fail-closed deny (→ Denied, not Asked)
        - exception/timeout: runtime failures (→ AllowError, not Asked)
        """
        for invalid_type in ("ast_validator", "exception", "timeout"):
            with pytest.raises(ValidationError):
                Asked(
                    content="x",
                    reason=DecisionReason(type=invalid_type),
                )

    def test_passthrough_with_multimodal_payload(self):
        o = Passthrough(
            content="[file_view: file_view — 2 image(s) loaded]",
            data=MultimodalPayload(blocks=[
                ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,abc")),
                ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,def")),
            ]),
        )
        assert o.variant == "passthrough"
        assert len(o.data.blocks) == 2

    def test_variants_do_not_have_call_context_fields(self):
        """CS2.6: ToolOutcome variants must not contain tool_call_id / tool_name / tool_source."""
        for cls in (AllowSuccess, AllowError, Denied, Asked, Passthrough):
            fields = cls.model_fields
            assert "tool_call_id" not in fields, f"{cls.__name__} should not have tool_call_id"
            assert "tool_name" not in fields, f"{cls.__name__} should not have tool_name"
            assert "tool_source" not in fields, f"{cls.__name__} should not have tool_source"


class TestToolOutcomeDiscriminator:
    def test_adapter_dispatches_on_variant_field(self):
        o1 = AllowSuccess(content="a")
        o2 = AllowError(content="b", reason=DecisionReason(type="exception"))

        restored1 = TOOL_OUTCOME_ADAPTER.validate_python(o1.model_dump())
        restored2 = TOOL_OUTCOME_ADAPTER.validate_python(o2.model_dump())

        assert isinstance(restored1, AllowSuccess)
        assert isinstance(restored2, AllowError)

    def test_adapter_round_trip_json(self):
        original = Denied(
            content="denied",
            reason=DecisionReason(type="smart_approve", code="llm_deny"),
        )
        j = original.model_dump_json()
        restored = TOOL_OUTCOME_ADAPTER.validate_json(j)
        assert restored == original

    def test_all_5_variants_round_trip(self):
        cases: list[ToolOutcome] = [
            AllowSuccess(content="ok"),
            AllowError(content="err", reason=DecisionReason(type="exception")),
            Denied(content="no", reason=DecisionReason(type="approval_policy")),
            Asked(content="?", reason=DecisionReason(type="risk_enforce")),
            Passthrough(content="pdf", data=MultimodalPayload(blocks=[])),
        ]
        for c in cases:
            j = c.model_dump_json()
            restored = TOOL_OUTCOME_ADAPTER.validate_json(j)
            assert restored == c, f"Round-trip failed for {type(c).__name__}"


class TestToolArtifact:
    def _make_tool_source(self) -> ToolSource:
        return ToolSource(
            source="native",
            category="shell",
            canonical_name="shell_execute",
        )

    def test_construct_with_allow_success_outcome(self):
        artifact = ToolArtifact(
            tool_call_id="call_123",
            tool_name="shell_execute",
            tool_source=self._make_tool_source(),
            outcome=AllowSuccess(content="ls output"),
        )
        assert artifact.tool_call_id == "call_123"
        assert artifact.outcome.variant == "allow_success"

    def test_round_trip_via_adapter_preserves_outcome_variant(self):
        """Two-level deserialization: ToolArtifact wraps outcome (ToolOutcome union)."""
        original = ToolArtifact(
            tool_call_id="call_x",
            tool_name="mcp_slack_post",
            tool_source=ToolSource(source="mcp", category="mcp", canonical_name="mcp_slack_post"),
            outcome=AllowError(
                content="MCP connection lost",
                reason=DecisionReason(type="exception", code="ConnectionError"),
                retryable=True,
            ),
        )
        dumped = original.model_dump(mode="json", by_alias=True)
        restored = TOOL_ARTIFACT_ADAPTER.validate_python(dumped)

        assert restored.tool_call_id == "call_x"
        assert isinstance(restored.outcome, AllowError)
        assert restored.outcome.reason.type == "exception"
        assert restored.outcome.retryable is True

    def test_all_5_outcome_variants_work_inside_artifact(self):
        outcomes: list[ToolOutcome] = [
            AllowSuccess(content="a"),
            AllowError(content="b", reason=DecisionReason(type="timeout")),
            Denied(content="c", reason=DecisionReason(type="ast_validator")),
            Asked(content="d", reason=DecisionReason(type="approval_policy")),
            Passthrough(content="e", data=MultimodalPayload(blocks=[])),
        ]
        ts = self._make_tool_source()
        for o in outcomes:
            artifact = ToolArtifact(
                tool_call_id="c1",
                tool_name="x",
                tool_source=ts,
                outcome=o,
            )
            dumped = artifact.model_dump(mode="json", by_alias=True)
            restored = TOOL_ARTIFACT_ADAPTER.validate_python(dumped)
            assert type(restored.outcome) == type(o)

    def test_tool_artifact_rejects_empty_call_id(self):
        """ToolArtifact.tool_call_id must be non-empty (min_length=1)."""
        with pytest.raises(ValidationError):
            ToolArtifact(
                tool_call_id="",
                tool_name="shell_execute",
                tool_source=self._make_tool_source(),
                outcome=AllowSuccess(content="ok"),
            )

    def test_tool_artifact_rejects_empty_tool_name(self):
        """ToolArtifact.tool_name must be non-empty (min_length=1)."""
        with pytest.raises(ValidationError):
            ToolArtifact(
                tool_call_id="call_1",
                tool_name="",
                tool_source=self._make_tool_source(),
                outcome=AllowSuccess(content="ok"),
            )


class TestContractSurfaceRejectsUnknownFields:
    """CS2 contract freeze: every new model must reject unknown fields.

    Without extra='forbid', producer-side typos, schema drift, or
    checkpointer/fixture pollution would silently pass round-trip, laundering
    schema violations. PR-A's core goal is freezing the contract surface —
    silent-drop defeats that goal and the golden matrix protection.
    """

    # Every new model added in PR-A. Hardcoded intentionally: adding a new model
    # to the taxonomy MUST be a deliberate act that extends this list, so that
    # future reviewers see the extra='forbid' requirement in diff.
    ALL_NEW_MODELS = [
        DecisionReason,
        ImageUrlPayload,
        ImageUrlBlock,
        FilePayload,
        FileBlock,
        TextBlock,
        MultimodalPayload,
        AllowSuccess,
        AllowError,
        Denied,
        Asked,
        Passthrough,
        ToolArtifact,
    ]

    def test_all_new_models_configure_extra_forbid(self):
        """Introspection: every model in ALL_NEW_MODELS must have extra='forbid'."""
        missing = [
            cls.__name__
            for cls in self.ALL_NEW_MODELS
            if cls.model_config.get("extra") != "forbid"
        ]
        assert not missing, (
            f"Models missing ConfigDict(extra='forbid'): {missing}. "
            "Unknown fields would be silently dropped, breaking contract freeze."
        )

    def test_decision_reason_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            DecisionReason.model_validate({"type": "exception", "unexpected": 1})

    def test_image_url_payload_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ImageUrlPayload.model_validate({"url": "http://x", "unexpected": 1})

    def test_file_payload_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            FilePayload.model_validate(
                {"filename": "a", "file_data": "b", "unexpected": 1}
            )

    def test_text_block_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            TextBlock.model_validate({"type": "text", "text": "x", "extra_key": 1})

    def test_image_url_block_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ImageUrlBlock.model_validate(
                {"type": "image_url", "image_url": {"url": "x"}, "extra_key": 1}
            )

    def test_file_block_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            FileBlock.model_validate(
                {
                    "type": "file",
                    "file": {"filename": "a", "file_data": "b"},
                    "extra_key": 1,
                }
            )

    def test_multimodal_payload_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MultimodalPayload.model_validate({"blocks": [], "unexpected": 1})

    def test_allow_success_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            AllowSuccess.model_validate(
                {"variant": "allow_success", "content": "ok", "unexpected": 123}
            )

    def test_allow_error_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            AllowError.model_validate(
                {
                    "variant": "allow_error",
                    "content": "e",
                    "reason": {"type": "exception"},
                    "unexpected": 1,
                }
            )

    def test_denied_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            Denied.model_validate(
                {
                    "variant": "denied",
                    "content": "x",
                    "reason": {"type": "ast_validator"},
                    "unexpected": 1,
                }
            )

    def test_asked_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            Asked.model_validate(
                {
                    "variant": "asked",
                    "content": "x",
                    "reason": {"type": "risk_enforce"},
                    "unexpected": 1,
                }
            )

    def test_passthrough_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            Passthrough.model_validate(
                {
                    "variant": "passthrough",
                    "content": "x",
                    "data": {"blocks": []},
                    "unexpected": 1,
                }
            )

    def test_tool_artifact_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ToolArtifact.model_validate(
                {
                    "tool_call_id": "c1",
                    "tool_name": "x",
                    "tool_source": {
                        "source": "native",
                        "category": "shell",
                        "canonical_name": "x",
                    },
                    "outcome": {"variant": "allow_success", "content": "ok"},
                    "unexpected": 456,
                }
            )

    def test_tool_artifact_rejects_unknown_field_nested_in_tool_source(self):
        """Contract freeze must close the loop on nested ToolSource too —
        otherwise producer/checkpointer drift on tool_source fields would be
        silently laundered and the golden matrix護欄 would become a leaky sieve.
        """
        with pytest.raises(ValidationError):
            ToolArtifact.model_validate(
                {
                    "tool_call_id": "c1",
                    "tool_name": "shell_execute",
                    "tool_source": {
                        "source": "native",
                        "category": "shell",
                        "canonical_name": "shell_execute",
                        "unexpected_nested": 1,
                    },
                    "outcome": {"variant": "allow_success", "content": "ok"},
                }
            )

    def test_tool_outcome_adapter_rejects_unknown_field(self):
        """Discriminated union via adapter also enforces extra='forbid'."""
        with pytest.raises(ValidationError):
            TOOL_OUTCOME_ADAPTER.validate_python(
                {"variant": "allow_success", "content": "ok", "unexpected": 1}
            )
