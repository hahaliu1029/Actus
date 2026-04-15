"""R2 CS2.14: ``ActusResponsesModel._to_openai_messages`` error prefix tests.

Mirror of ``test_actus_chat_model_error_prefix.py`` — verifies the
Responses API adapter injects the same prefix as the Chat Completions
adapter, so both code paths produce identical wire-format content on
the error branch.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    DecisionReason,
    Denied,
    ToolArtifact,
)
from app.domain.services.tools.tool_source_resolver import ToolSource
from app.infrastructure.external.llm.actus_responses_model import (
    ActusResponsesModel,
)


def _adapter() -> ActusResponsesModel:
    """Skip ``__init__`` — ``_to_openai_messages`` is pure message
    conversion and doesn't touch any OpenAI client."""
    return ActusResponsesModel.__new__(ActusResponsesModel)


def _make_artifact_dict(outcome) -> dict:
    artifact = ToolArtifact(
        tool_call_id="c1",
        tool_name="x",
        tool_source=ToolSource(
            source="native", category="shell", canonical_name="shell_execute"
        ),
        outcome=outcome,
    )
    return artifact.model_dump(mode="json", by_alias=True)


class TestResponsesModelErrorPrefix:
    def test_allow_error_timeout_gets_tool_failed_prefix(self):
        outcome = AllowError(
            content="TCP timeout", reason=DecisionReason(type="timeout")
        )
        msg = ToolMessage(
            content="TCP timeout",
            tool_call_id="c1",
            name="x",
            status="error",
            artifact=_make_artifact_dict(outcome),
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "tool"
        assert api_msgs[0]["tool_call_id"] == "c1"
        assert api_msgs[0]["content"] == "[TOOL_FAILED: timeout] TCP timeout"

    def test_allow_error_exception_reason(self):
        outcome = AllowError(
            content="Something broke",
            reason=DecisionReason(type="exception", code="ValueError"),
        )
        msg = ToolMessage(
            content="Something broke",
            tool_call_id="c2",
            name="x",
            status="error",
            artifact=_make_artifact_dict(outcome),
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert (
            api_msgs[0]["content"] == "[TOOL_FAILED: exception] Something broke"
        )

    def test_denied_gets_tool_denied_prefix(self):
        outcome = Denied(
            content="blocked by policy",
            reason=DecisionReason(type="approval_policy"),
        )
        msg = ToolMessage(
            content="blocked by policy",
            tool_call_id="c3",
            name="x",
            status="error",
            artifact=_make_artifact_dict(outcome),
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert (
            api_msgs[0]["content"]
            == "[TOOL_DENIED: approval_policy] blocked by policy"
        )

    def test_denied_ast_validator_reason(self):
        outcome = Denied(
            content="rm -rf blocked",
            reason=DecisionReason(type="ast_validator"),
        )
        msg = ToolMessage(
            content="rm -rf blocked",
            tool_call_id="c4",
            name="x",
            status="error",
            artifact=_make_artifact_dict(outcome),
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert (
            api_msgs[0]["content"]
            == "[TOOL_DENIED: ast_validator] rm -rf blocked"
        )

    def test_success_status_no_prefix(self):
        msg = ToolMessage(
            content="normal output",
            tool_call_id="c5",
            name="x",
            status="success",
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "normal output"

    def test_success_status_with_artifact_still_no_prefix(self):
        outcome = AllowSuccess(content="ok")
        msg = ToolMessage(
            content="ok",
            tool_call_id="c6",
            name="x",
            status="success",
            artifact=_make_artifact_dict(outcome),
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "ok"

    def test_error_status_without_artifact_falls_back_to_generic_prefix(self):
        """Missing artifact → generic "[TOOL_ERROR]" fallback (matches
        ActusChatModel Task 33 behavior)."""
        msg = ToolMessage(
            content="raw error",
            tool_call_id="c7",
            name="x",
            status="error",
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_ERROR] raw error"

    def test_error_status_with_malformed_artifact_falls_back_to_generic_prefix(self):
        """Malformed artifact → helper returns None → generic fallback."""
        msg = ToolMessage(
            content="weird payload",
            tool_call_id="c8",
            name="x",
            status="error",
            artifact={"random": "garbage"},
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_ERROR] weird payload"

    def test_error_with_empty_content_and_artifact_no_trailing_space(self):
        """Empty content + artifact → no trailing space after prefix."""
        outcome = AllowError(
            content="", reason=DecisionReason(type="timeout")
        )
        msg = ToolMessage(
            content="",
            tool_call_id="c9",
            name="x",
            status="error",
            artifact=_make_artifact_dict(outcome),
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_FAILED: timeout]"

    def test_error_with_empty_content_and_no_artifact_no_trailing_space(self):
        """Empty content + missing artifact → generic fallback alone."""
        msg = ToolMessage(
            content="",
            tool_call_id="c10",
            name="x",
            status="error",
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_ERROR]"

    def test_error_with_malformed_reason_dict_falls_back_gracefully(self):
        """Regression mirror for ActusChatModel: non-dict ``reason`` must
        not crash the Responses adapter — degrades to 'unknown' reason
        while keeping the variant marker."""
        msg = ToolMessage(
            content="real error",
            tool_call_id="c_mal_1",
            name="x",
            status="error",
            artifact={
                "outcome": {
                    "variant": "allow_error",
                    "reason": "oops",
                }
            },
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_FAILED: unknown] real error"

    def test_error_with_malformed_reason_list_falls_back_gracefully(self):
        """Same regression for list-valued ``reason``."""
        msg = ToolMessage(
            content="boom",
            tool_call_id="c_mal_2",
            name="x",
            status="error",
            artifact={
                "variant": "denied",
                "reason": [1, 2, 3],
            },
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_DENIED: unknown] boom"

    def test_error_with_list_content_keeps_list_unchanged(self):
        """List content (multimodal) on error branch is left alone —
        shouldn't occur in practice (Passthrough is status=success), but
        the defensive ``isinstance(str)`` guard prevents wire-format
        corruption if it ever does."""
        msg = ToolMessage(
            content=[{"type": "text", "text": "hi"}],  # type: ignore[arg-type]
            tool_call_id="c11",
            name="x",
            status="error",
            artifact={
                "outcome": {
                    "variant": "allow_error",
                    "reason": {"type": "exception"},
                }
            },
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == [{"type": "text", "text": "hi"}]

    def test_mixed_batch_only_errors_get_prefix(self):
        err_outcome = AllowError(
            content="timeout here", reason=DecisionReason(type="timeout")
        )
        err_msg = ToolMessage(
            content="timeout here",
            tool_call_id="c_err",
            name="x",
            status="error",
            artifact=_make_artifact_dict(err_outcome),
        )
        ok_msg = ToolMessage(
            content="all good",
            tool_call_id="c_ok",
            name="x",
            status="success",
        )

        api_msgs = _adapter()._to_openai_messages(
            [
                HumanMessage(content="run them"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c_err", "name": "x", "args": {}},
                        {"id": "c_ok", "name": "x", "args": {}},
                    ],
                ),
                err_msg,
                ok_msg,
            ]
        )

        tool_entries = [m for m in api_msgs if m.get("role") == "tool"]
        assert len(tool_entries) == 2
        err_entry = next(
            t for t in tool_entries if t["tool_call_id"] == "c_err"
        )
        ok_entry = next(t for t in tool_entries if t["tool_call_id"] == "c_ok")
        assert err_entry["content"] == "[TOOL_FAILED: timeout] timeout here"
        assert ok_entry["content"] == "all good"


# ============================================================
# E2E integration: Layer 3 → adapter
# ============================================================
#
# Mirror of the ChatModel E2E class — locks in the full pipeline shape
# so any future drift in ``_translate_outcome`` is caught at the
# Responses adapter boundary.


import asyncio  # noqa: E402

from app.domain.services.graphs.react_graph import (  # noqa: E402
    _SessionContext,
    _translate_outcome,
)


def _run(coro):
    return asyncio.run(coro)


class TestResponsesModelE2ELayer3ToAdapter:
    """Integration: full ``_translate_outcome → _to_openai_messages`` pipe."""

    def _ctx(self) -> _SessionContext:
        return _SessionContext(session_id="s1", user_id="u1")

    def _mcp_source(self) -> ToolSource:
        return ToolSource(
            source="mcp", category="mcp", canonical_name="mcp_slack_post"
        )

    def _shell_source(self) -> ToolSource:
        return ToolSource(
            source="native", category="shell", canonical_name="shell_execute"
        )

    def test_allow_error_e2e(self):
        tc = {
            "id": "call_e2e_err",
            "name": "mcp_slack_post",
            "args": {"channel": "#foo"},
            "type": "tool_call",
        }
        outcome = AllowError(
            content="MCP timeout", reason=DecisionReason(type="timeout")
        )

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                self._mcp_source(),
                self._ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert msg is not None
        assert msg.status == "error"
        assert msg.artifact is not None

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "[TOOL_FAILED: timeout] MCP timeout"
        assert api_msgs[0]["tool_call_id"] == "call_e2e_err"

    def test_denied_e2e(self):
        tc = {
            "id": "call_e2e_deny",
            "name": "shell_execute",
            "args": {"command": "rm -rf /"},
            "type": "tool_call",
        }
        outcome = Denied(
            content="blocked by AST validator",
            reason=DecisionReason(type="ast_validator"),
        )

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                self._shell_source(),
                self._ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert (
            api_msgs[0]["content"]
            == "[TOOL_DENIED: ast_validator] blocked by AST validator"
        )

    def test_allow_success_e2e_no_prefix(self):
        tc = {
            "id": "call_e2e_ok",
            "name": "shell_execute",
            "args": {"command": "ls"},
            "type": "tool_call",
        }
        outcome = AllowSuccess(content="file1\nfile2")

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                self._shell_source(),
                self._ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        api_msgs = _adapter()._to_openai_messages([msg])

        assert api_msgs[0]["content"] == "file1\nfile2"
        assert "[TOOL_" not in api_msgs[0]["content"]
