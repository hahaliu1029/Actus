"""R2 CS2.14: ``_format_error_prefix`` helper tests.

Covers the three input shapes (typed ToolArtifact, typed outcome,
dict-form after checkpointer round-trip) + the no-prefix path for
success variants + defensive handling of garbage input.
"""
from __future__ import annotations

from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    DecisionReason,
    Denied,
    MultimodalPayload,
    Passthrough,
    ToolArtifact,
)
from app.domain.services.tools.tool_source_resolver import ToolSource
from app.infrastructure.external.llm._error_prefix import _format_error_prefix


def _ts() -> ToolSource:
    return ToolSource(
        source="native", category="shell", canonical_name="shell_execute"
    )


def _make_artifact(outcome) -> ToolArtifact:
    return ToolArtifact(
        tool_call_id="c1",
        tool_name="x",
        tool_source=_ts(),
        outcome=outcome,
    )


class TestFormatErrorPrefixDictForm:
    """Dict form — mirrors the checkpointer round-trip path."""

    def test_allow_error_produces_tool_failed_prefix(self):
        outcome = AllowError(
            content="err", reason=DecisionReason(type="timeout", code="tcp")
        )
        artifact_dict = _make_artifact(outcome).model_dump(
            mode="json", by_alias=True
        )
        assert (
            _format_error_prefix(artifact_dict) == "[TOOL_FAILED: timeout]"
        )

    def test_allow_error_exception_reason(self):
        outcome = AllowError(
            content="boom",
            reason=DecisionReason(type="exception", code="ValueError"),
        )
        artifact_dict = _make_artifact(outcome).model_dump(
            mode="json", by_alias=True
        )
        assert (
            _format_error_prefix(artifact_dict) == "[TOOL_FAILED: exception]"
        )

    def test_denied_produces_tool_denied_prefix(self):
        outcome = Denied(
            content="no",
            reason=DecisionReason(type="approval_policy"),
        )
        artifact_dict = _make_artifact(outcome).model_dump(
            mode="json", by_alias=True
        )
        assert (
            _format_error_prefix(artifact_dict)
            == "[TOOL_DENIED: approval_policy]"
        )

    def test_denied_ast_validator_reason(self):
        outcome = Denied(
            content="AST blocked",
            reason=DecisionReason(type="ast_validator"),
        )
        artifact_dict = _make_artifact(outcome).model_dump(
            mode="json", by_alias=True
        )
        assert (
            _format_error_prefix(artifact_dict)
            == "[TOOL_DENIED: ast_validator]"
        )

    def test_allow_success_produces_no_prefix(self):
        outcome = AllowSuccess(content="ok")
        artifact_dict = _make_artifact(outcome).model_dump(
            mode="json", by_alias=True
        )
        assert _format_error_prefix(artifact_dict) is None

    def test_passthrough_produces_no_prefix(self):
        outcome = Passthrough(content="pdf", data=MultimodalPayload(blocks=[]))
        artifact_dict = _make_artifact(outcome).model_dump(
            mode="json", by_alias=True
        )
        assert _format_error_prefix(artifact_dict) is None


class TestFormatErrorPrefixPydanticForm:
    """Pydantic object form — in-process, pre-checkpointer."""

    def test_typed_artifact_allow_error(self):
        outcome = AllowError(
            content="x",
            reason=DecisionReason(type="exception", code="TypeError"),
        )
        artifact = _make_artifact(outcome)
        assert _format_error_prefix(artifact) == "[TOOL_FAILED: exception]"

    def test_typed_artifact_denied(self):
        outcome = Denied(
            content="x", reason=DecisionReason(type="risk_enforce")
        )
        artifact = _make_artifact(outcome)
        assert _format_error_prefix(artifact) == "[TOOL_DENIED: risk_enforce]"

    def test_typed_artifact_allow_success(self):
        artifact = _make_artifact(AllowSuccess(content="ok"))
        assert _format_error_prefix(artifact) is None

    def test_typed_outcome_directly(self):
        """Some callers may pass just the outcome without the artifact wrapper."""
        outcome = AllowError(
            content="e", reason=DecisionReason(type="timeout")
        )
        assert _format_error_prefix(outcome) == "[TOOL_FAILED: timeout]"

        denied = Denied(
            content="d", reason=DecisionReason(type="smart_approve")
        )
        assert _format_error_prefix(denied) == "[TOOL_DENIED: smart_approve]"


class TestFormatErrorPrefixDefensive:
    """Malformed / unknown input must never crash the adapter."""

    def test_none_returns_none(self):
        assert _format_error_prefix(None) is None

    def test_bare_string_returns_none(self):
        assert _format_error_prefix("just a string") is None

    def test_empty_dict_returns_none(self):
        assert _format_error_prefix({}) is None

    def test_dict_without_variant_returns_none(self):
        assert _format_error_prefix({"random": "data"}) is None

    def test_dict_with_unknown_variant_returns_none(self):
        assert _format_error_prefix({"variant": "mystery"}) is None

    def test_dict_outcome_without_reason_still_works(self):
        """Fallback reason='unknown' when reason dict missing."""
        payload = {"variant": "allow_error"}
        assert _format_error_prefix(payload) == "[TOOL_FAILED: unknown]"


class TestFormatErrorPrefixMalformedReason:
    """Regression: non-dict ``reason`` must not crash the helper.

    Older artifact shapes / corrupted checkpointer state / hand-crafted
    fixtures can deliver ``reason`` as a string, list, int, or other
    non-dict type. The pre-fix ``(reason or {}).get("type", ...)`` path
    crashed with ``AttributeError`` on any truthy non-dict, which
    propagated out of the helper and killed the LLM serializer before
    the ``or "[TOOL_ERROR]"`` fallback in the adapter could fire.

    These tests lock in fail-soft behavior: reason='unknown' when the
    ``reason`` field is malformed, while still honoring the variant
    (allow_error vs denied) so the LLM still sees the right marker.
    """

    def test_flat_dict_with_string_reason(self):
        """``reason="oops"`` → degrade to 'unknown' reason, keep variant."""
        assert (
            _format_error_prefix({"variant": "allow_error", "reason": "oops"})
            == "[TOOL_FAILED: unknown]"
        )

    def test_nested_dict_with_string_reason(self):
        """``outcome.reason="oops"`` under full artifact wrapper."""
        assert (
            _format_error_prefix(
                {"outcome": {"variant": "denied", "reason": "oops"}}
            )
            == "[TOOL_DENIED: unknown]"
        )

    def test_flat_dict_with_list_reason(self):
        """``reason=[...]`` → degrade to 'unknown'."""
        assert (
            _format_error_prefix(
                {"variant": "allow_error", "reason": [1, 2, 3]}
            )
            == "[TOOL_FAILED: unknown]"
        )

    def test_flat_dict_with_int_reason(self):
        """``reason=42`` → degrade to 'unknown'."""
        assert (
            _format_error_prefix({"variant": "allow_error", "reason": 42})
            == "[TOOL_FAILED: unknown]"
        )

    def test_flat_dict_with_bool_reason(self):
        """``reason=True`` → degrade to 'unknown' (truthy non-dict)."""
        assert (
            _format_error_prefix({"variant": "allow_error", "reason": True})
            == "[TOOL_FAILED: unknown]"
        )

    def test_nested_dict_with_malformed_reason_for_denied(self):
        """Mirror coverage for the Denied path."""
        assert (
            _format_error_prefix(
                {"outcome": {"variant": "denied", "reason": {"no_type": "x"}}}
            )
            == "[TOOL_DENIED: unknown]"
        )

    def test_valid_dict_reason_still_extracts_type(self):
        """Regression guard: the fix must not break the happy path."""
        assert (
            _format_error_prefix(
                {"variant": "allow_error", "reason": {"type": "timeout"}}
            )
            == "[TOOL_FAILED: timeout]"
        )
