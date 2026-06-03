"""Tests for R2 PR-B Commit 2 tool_node helpers.

Covers:
- ``_invoke_wrapper``
- ``_translate_outcome``

NOTE: Project convention — no pytest-asyncio. Async tests wrap coroutines in
``_run()`` which delegates to ``asyncio.run()`` (mirrors the pattern in
tests/domain/services/test_approval_cache.py).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from langchain_core.tools import tool as lc_tool

from langchain_core.messages import HumanMessage, ToolMessage

from app.domain.models.event import ToolEvent
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalPayload,
    Passthrough,
    TextBlock,
)
from app.domain.services.graphs.react_graph import (
    _invoke_wrapper,
    _SessionContext,
    _translate_outcome,
)
from app.domain.services.tools._supervisor_tool_wrapper import (
    SupervisorAwareToolWrapper,
)
from app.domain.services.tools.tool_source_resolver import ToolSource


def _run(coro):
    """Run an async coroutine synchronously — no pytest-asyncio needed."""
    return asyncio.run(coro)


class TestInvokeWrapperContentAndArtifact:
    def test_uses_tool_call_shape_and_reads_typed_artifact(self):
        """Commit 2a: _invoke_wrapper must pass ToolCall dict and read artifact."""

        @lc_tool(response_format="content_and_artifact")
        async def fake_tool(x: str) -> tuple[str, AllowSuccess]:
            """Fake tool for testing wrapper invocation shape."""
            outcome = AllowSuccess(content=f"got {x}", data={"echo": x})
            return outcome.content, outcome

        tc = _make_tool_call("tc-artifact", "fake_tool", {"x": "hello"})
        src = _make_mcp_source()

        outcome = _run(_invoke_wrapper(fake_tool, tc, src))

        assert isinstance(outcome, AllowSuccess)
        assert outcome.content == "got hello"
        assert outcome.data == {"echo": "hello"}


def _make_native_shell_source() -> ToolSource:
    return ToolSource(
        source="native", category="shell", canonical_name="shell_execute"
    )


def _make_mcp_source() -> ToolSource:
    return ToolSource(
        source="mcp", category="mcp", canonical_name="mcp_slack_post"
    )


def _make_tool_call(tc_id: str, name: str, args: dict | None = None) -> dict:
    """Build a langchain-compatible ToolCall dict.

    langchain_core.messages.tool.ToolCall is a TypedDict, so a plain
    dict with the right keys is indistinguishable from a real ToolCall
    and works in runtime checks.
    """
    return {
        "id": tc_id,
        "name": name,
        "args": args or {},
        "type": "tool_call",
    }


class TestInvokeWrapperCommit2:
    """Layer 2 (_invoke_wrapper) — Commit 2a reads ToolMessage.artifact."""

    def test_plain_string_return_is_rejected_as_wrong_shape(self):
        """Misconfigured typed wrapper (declares ``content_and_artifact`` but
        returns a bare string instead of a ToolMessage) must raise the
        ``wrong_ainvoke_shape`` contract error. Setting ``response_format``
        explicitly documents what path this test exercises."""
        tool = AsyncMock()
        tool.response_format = "content_and_artifact"
        tool.ainvoke = AsyncMock(return_value="hello")
        tool.name = "fake"
        tc = _make_tool_call("c1", "fake", {"arg": 1})
        source = _make_mcp_source()

        outcome = _run(_invoke_wrapper(tool, tc, source))

        assert isinstance(outcome, AllowError)
        assert outcome.reason.type == "exception"
        assert outcome.reason.code == "wrong_ainvoke_shape"

    def test_legacy_wrapper_string_return_becomes_allow_success(self):
        """Legacy ``@lc_tool`` without ``response_format`` returns a plain
        value; the dispatcher wraps it as ``AllowSuccess`` so the real
        file_view / shell / memory tools keep working under the dual-path
        dispatcher."""
        tool = AsyncMock()
        tool.response_format = "content"
        tool.ainvoke = AsyncMock(return_value="legacy ok")
        tool.name = "legacy_fake"
        tc = _make_tool_call("c_legacy", "legacy_fake", {"x": 1})
        source = _make_mcp_source()

        outcome = _run(_invoke_wrapper(tool, tc, source))

        assert isinstance(outcome, AllowSuccess)
        assert outcome.content == "legacy ok"

    def test_returns_allow_error_on_exception(self):
        tool = AsyncMock()
        tool.response_format = "content_and_artifact"
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        tool.name = "fake"
        tc = _make_tool_call("c2", "fake")
        source = _make_mcp_source()

        outcome = _run(_invoke_wrapper(tool, tc, source))

        assert isinstance(outcome, AllowError)
        assert outcome.reason.type == "exception"
        assert outcome.reason.code == "RuntimeError"

    def test_tool_source_argument_still_does_not_affect_success_path(self):
        @lc_tool(response_format="content_and_artifact")
        async def fake_tool(x: str) -> tuple[str, AllowSuccess]:
            """Fake tool for source-agnostic wrapper tests."""
            outcome = AllowSuccess(content=f"ok:{x}")
            return outcome.content, outcome

        tc = _make_tool_call("c3", "fake_tool", {"x": "v"})
        for source in (_make_mcp_source(), _make_native_shell_source()):
            outcome = _run(_invoke_wrapper(fake_tool, tc, source))
            assert isinstance(outcome, AllowSuccess)


class _FakeSupervisor:
    def __init__(self) -> None:
        self.inc_calls: list[tuple[str, str]] = []
        self.dec_calls: list[tuple[str, str]] = []

    async def inflight_inc(self, *, session_id: str, kind: str) -> None:
        self.inc_calls.append((session_id, kind))

    async def inflight_dec(self, *, session_id: str, kind: str) -> None:
        self.dec_calls.append((session_id, kind))


class TestInvokeWrapperSessionConfig:
    def test_passes_session_id_to_wrapped_typed_tool(self):
        @lc_tool(response_format="content_and_artifact")
        async def typed_tool(x: str) -> tuple[str, AllowSuccess]:
            """Typed tool for supervisor config propagation."""
            outcome = AllowSuccess(content=f"ok:{x}")
            return outcome.content, outcome

        supervisor = _FakeSupervisor()
        wrapper = SupervisorAwareToolWrapper(
            inner=typed_tool,
            supervisor=supervisor,
        )
        tc = _make_tool_call("c-session", "typed_tool", {"x": "v"})

        outcome = _run(
            _invoke_wrapper(
                wrapper,
                tc,
                _make_mcp_source(),
                session_id="sess-X",
            )
        )

        assert isinstance(outcome, AllowSuccess)
        assert supervisor.inc_calls == [("sess-X", "tool")]
        assert supervisor.dec_calls == [("sess-X", "tool")]

    def test_without_session_id_keeps_wrapped_tool_passthrough_behavior(self):
        @lc_tool(response_format="content_and_artifact")
        async def typed_tool(x: str) -> tuple[str, AllowSuccess]:
            """Typed tool for supervisor passthrough behavior."""
            outcome = AllowSuccess(content=f"ok:{x}")
            return outcome.content, outcome

        supervisor = _FakeSupervisor()
        wrapper = SupervisorAwareToolWrapper(
            inner=typed_tool,
            supervisor=supervisor,
        )
        tc = _make_tool_call("c-no-session", "typed_tool", {"x": "v"})

        outcome = _run(_invoke_wrapper(wrapper, tc, _make_mcp_source()))

        assert isinstance(outcome, AllowSuccess)
        assert supervisor.inc_calls == []
        assert supervisor.dec_calls == []


# ============================================================
# Task 12 — _translate_outcome (Layer 3)
# ============================================================


def _make_session_ctx() -> _SessionContext:
    return _SessionContext(session_id="s1", user_id="u1")


class TestTranslateOutcomeAllVariants:
    """I-4.4 exhaustiveness: all 5 ToolOutcome variants must translate correctly."""

    def test_allow_success_produces_success_tool_message(self):
        tc = _make_tool_call("c1", "shell_execute")
        src = _make_native_shell_source()
        outcome = AllowSuccess(content="ls output")

        msg, deferred, events = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert isinstance(msg, ToolMessage)
        assert msg.status == "success"
        assert msg.content == "ls output"
        assert msg.tool_call_id == "c1"
        assert deferred == []
        assert any(isinstance(e, ToolEvent) for e in events)

    def test_allow_error_produces_error_tool_message_no_prefix(self):
        """Layer 3 must NOT prefix '[TOOL_FAILED]' — that's the adapter's job."""
        tc = _make_tool_call("c2", "mcp_xxx")
        src = _make_mcp_source()
        outcome = AllowError(
            content="MCP timeout",
            reason=DecisionReason(type="timeout", code="mcp_connect_timeout"),
            retryable=True,
        )

        msg, deferred, events = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert msg.status == "error"
        assert msg.content == "MCP timeout"
        assert "TOOL_FAILED" not in msg.content
        assert deferred == []

    def test_denied_produces_error_tool_message(self):
        tc = _make_tool_call("c3", "shell_execute")
        src = _make_native_shell_source()
        outcome = Denied(
            content="AST blocked rm -rf",
            reason=DecisionReason(type="ast_validator", code="dangerous_rm"),
        )

        msg, deferred, events = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert msg.status == "error"
        assert msg.content == "AST blocked rm -rf"
        assert "TOOL_DENIED" not in msg.content  # adapter prefix comes later

    def test_asked_returns_none_tool_message_and_no_confirmation_event(self):
        """Asked is the interrupt path — no ToolMessage and no confirmation event here.

        Confirmation card event is emitted by the dispatcher / risk gate,
        not by `_translate_outcome()`. This keeps severity-scale fields
        (`high/medium/...`) and pattern details sourced from `RiskAssessment`
        instead of overloading `Asked.reason.type`.
        """
        tc = _make_tool_call("c4", "skill_dangerous")
        src = ToolSource(
            source="skill", category="skill", canonical_name="skill_dangerous"
        )
        outcome = Asked(
            content="Skill needs approval",
            reason=DecisionReason(
                type="risk_enforce", code="skill_x_risk_high", message="High risk skill"
            ),
        )

        msg, deferred, events = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert msg is None
        assert deferred == []
        assert events == []

    def test_passthrough_produces_both_tool_message_and_deferred_human_message(self):
        """I-4.4c: Passthrough produces BOTH a ToolMessage and a HumanMessage.

        The HumanMessage is required because OpenAI/LangChain multimodal
        content can only be read from HumanMessage.content — ToolMessage.artifact
        is graph-side side-channel that the LLM cannot see.
        """
        tc = _make_tool_call("c5", "file_view")
        src = ToolSource(source="native", category="file", canonical_name="file_view")
        blocks = [
            ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,abc")),
            ImageUrlBlock(image_url=ImageUrlPayload(url="data:image/png;base64,def")),
        ]
        outcome = Passthrough(
            content="[file_view: file_view — 2 image(s) loaded]",
            data=MultimodalPayload(blocks=blocks),
        )

        msg, deferred, events = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert isinstance(msg, ToolMessage)
        assert msg.status == "success"
        assert len(deferred) == 1
        assert isinstance(deferred[0], HumanMessage)

        human_content = deferred[0].content
        assert isinstance(human_content, list)
        # First block: text summary matching legacy react_graph.py:815-818 format
        assert human_content[0] == {
            "type": "text",
            "text": "[file_view: file_view — 2 image(s) loaded]",
        }
        # Subsequent blocks: by_alias=True dumped image_url blocks with wire-format key "type"
        assert human_content[1]["type"] == "image_url"
        assert "image_url" in human_content[1]
        assert "kind" not in human_content[1]  # alias stripped


class TestTranslateOutcomeEventSemantics:
    """R1/R2 contract: ToolEvent.tool_name holds the canonical CATEGORY
    (browser / search / shell / file / ...), not the literal tool_call name.

    AgentTaskRunner._handle_tool_event (agent_task_runner.py:2078) branches
    on ``event.tool_name`` to enrich browser screenshots / search results.
    Emitting the actual tool name here would silently break the enrichment
    path. The literal tool name lives in ``function_name``.

    This whole class exists because the original Chunk 2 _translate_outcome
    wrote ``tool_name=tool_call["name"]``, which would have caused every
    new node-split path (Layer 1 Denied / Layer 2 AllowSuccess / Layer 2
    AllowError / Layer 2 Passthrough) to drift away from the rest of
    react_graph.py. P2 review caught it; this is the regression guard.
    """

    def _check_category_and_function_name(
        self,
        *,
        outcome,
        tool_call,
        tool_source,
        expected_category: str,
        expected_function_name: str,
    ):
        from app.domain.models.event import ToolEvent

        _, _, events = _run(
            _translate_outcome(
                outcome,
                tool_call,
                tool_source,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )
        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        assert tool_events, f"Expected ≥1 ToolEvent, got {events}"
        e = tool_events[-1]
        assert e.tool_name == expected_category, (
            f"tool_name should be canonical category {expected_category!r}, "
            f"got {e.tool_name!r}"
        )
        assert e.function_name == expected_function_name, (
            f"function_name should be the literal tool name "
            f"{expected_function_name!r}, got {e.function_name!r}"
        )

    def test_allow_success_sets_tool_name_to_category(self):
        self._check_category_and_function_name(
            outcome=AllowSuccess(content="ls output"),
            tool_call=_make_tool_call("c1", "shell_execute"),
            tool_source=_make_native_shell_source(),
            expected_category="shell",
            expected_function_name="shell_execute",
        )

    def test_allow_error_sets_tool_name_to_category(self):
        self._check_category_and_function_name(
            outcome=AllowError(
                content="oops",
                reason=DecisionReason(type="exception", code="boom"),
            ),
            tool_call=_make_tool_call("c2", "mcp_slack_post"),
            tool_source=_make_mcp_source(),
            expected_category="mcp",
            expected_function_name="mcp_slack_post",
        )

    def test_denied_sets_tool_name_to_category(self):
        self._check_category_and_function_name(
            outcome=Denied(
                content="blocked",
                reason=DecisionReason(type="ast_validator"),
            ),
            tool_call=_make_tool_call("c3", "shell_execute"),
            tool_source=_make_native_shell_source(),
            expected_category="shell",
            expected_function_name="shell_execute",
        )

    def test_passthrough_sets_tool_name_to_category(self):
        self._check_category_and_function_name(
            outcome=Passthrough(
                content="summary",
                data=MultimodalPayload(blocks=[]),
            ),
            tool_call=_make_tool_call("c4", "file_view"),
            tool_source=ToolSource(
                source="native", category="file", canonical_name="file_view"
            ),
            expected_category="file",
            expected_function_name="file_view",
        )

    def test_browser_tool_preserves_browser_category_for_enrichment(self):
        """Explicit coverage for the enrichment branch most likely to
        silently fail if the category drifts: browser screenshot injection.
        """
        self._check_category_and_function_name(
            outcome=AllowSuccess(content="clicked"),
            tool_call=_make_tool_call("c5", "browser_click"),
            tool_source=ToolSource(
                source="native",
                category="browser",
                canonical_name="browser_click",
            ),
            expected_category="browser",
            expected_function_name="browser_click",
        )

    def test_function_args_preserved_verbatim(self):
        """function_args must round-trip exactly (audit path)."""
        from app.domain.models.event import ToolEvent

        original_args = {"command": "ls -la", "timeout": 30}
        tc = _make_tool_call("c6", "shell_execute", original_args)

        _, _, events = _run(
            _translate_outcome(
                AllowSuccess(content="ok"),
                tc,
                _make_native_shell_source(),
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        assert tool_events[0].function_args == original_args

    def test_allow_error_emits_failed_function_result(self):
        """AllowError must preserve a failed ToolResult for downstream enrichers."""
        from app.domain.models.event import ToolEvent

        tc = _make_tool_call("c7", "mcp_slack_post", {"channel": "#ops"})

        _, _, events = _run(
            _translate_outcome(
                AllowError(
                    content="remote timeout",
                    reason=DecisionReason(type="timeout", code="mcp_timeout"),
                ),
                tc,
                _make_mcp_source(),
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        assert tool_events
        event = tool_events[-1]
        assert event.function_result is not None
        assert event.function_result.success is False
        assert event.function_result.message == "remote timeout"

    def test_denied_emits_failed_function_result(self):
        """Denied path must not degrade to a result-less ToolEvent."""
        from app.domain.models.event import ToolEvent

        tc = _make_tool_call("c8", "skill_dangerous", {"action": "wipe"})

        _, _, events = _run(
            _translate_outcome(
                Denied(
                    content="用户拒绝了此操作",
                    reason=DecisionReason(type="approval_policy", code="deny"),
                ),
                tc,
                ToolSource(
                    source="skill",
                    category="skill",
                    canonical_name="skill_dangerous",
                ),
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        assert tool_events
        event = tool_events[-1]
        assert event.function_result is not None
        assert event.function_result.success is False
        assert event.function_result.message == "用户拒绝了此操作"


class TestTranslateOutcomeInheritedMechanisms:
    """I-4.4a/b/c: Layer 3 must inherit the 3 existing react_graph mechanisms."""

    def test_applies_truncate_tool_content(self):
        """I-4.4a: truncation applies to final_content."""
        tc = _make_tool_call("c6", "shell_execute")
        src = _make_native_shell_source()
        outcome = AllowSuccess(content="x" * 20000)

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert len(msg.content) <= 8100, (
            f"Expected truncated content ≤ 8100 chars, got {len(msg.content)}"
        )

    def test_applies_guide_injector_on_allow_success(self):
        """I-4.4b: guide_injector fires on AllowSuccess."""
        tc = _make_tool_call("c7", "skill_example")
        src = ToolSource(
            source="skill", category="skill", canonical_name="skill_example"
        )
        outcome = AllowSuccess(content="result text")

        def guide_injector(tool_name: str) -> str | None:
            return "GUIDE: use this tool sparingly"

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=guide_injector,
            )
        )

        assert "GUIDE: use this tool sparingly" in msg.content
        assert "[Skill Guide]" in msg.content

    def test_guide_injector_not_applied_on_allow_error(self):
        """I-4.4b variant: error paths must NOT get guide appended."""
        tc = _make_tool_call("c8", "skill_example")
        src = ToolSource(
            source="skill", category="skill", canonical_name="skill_example"
        )
        outcome = AllowError(
            content="failed", reason=DecisionReason(type="exception")
        )

        def guide_injector(tool_name: str) -> str | None:
            return "SHOULD NOT APPEAR"

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=guide_injector,
            )
        )

        assert "SHOULD NOT APPEAR" not in msg.content
        assert "[Skill Guide]" not in msg.content

    def test_guide_injector_applies_on_passthrough_before_truncation(self):
        """I-4.4b + I-4.4a ordering: guide injected first, then truncated."""
        tc = _make_tool_call("c9", "file_view")
        src = ToolSource(
            source="native", category="file", canonical_name="file_view"
        )
        outcome = Passthrough(
            content="pdf summary",
            data=MultimodalPayload(blocks=[TextBlock(text="hello")]),
        )

        def guide_injector(tool_name: str) -> str | None:
            return "read carefully"

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=guide_injector,
            )
        )

        assert "read carefully" in msg.content


class TestTranslateOutcomeArtifactWireFormat:
    """I-4.3: ToolMessage.artifact wire format must use by_alias=True."""

    def test_artifact_uses_by_alias_true(self):
        tc = _make_tool_call("c10", "file_view")
        src = ToolSource(
            source="native", category="file", canonical_name="file_view"
        )
        outcome = Passthrough(
            content="summary",
            data=MultimodalPayload(
                blocks=[ImageUrlBlock(image_url=ImageUrlPayload(url="x"))]
            ),
        )

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert isinstance(msg.artifact, dict)
        outcome_dict = msg.artifact["outcome"]
        assert outcome_dict["variant"] == "passthrough"
        first_block = outcome_dict["data"]["blocks"][0]
        assert first_block["type"] == "image_url"  # alias, not Python name
        assert "kind" not in first_block

    def test_artifact_contains_tool_source(self):
        tc = _make_tool_call("c11", "mcp_slack_post")
        src = _make_mcp_source()
        outcome = AllowSuccess(content="posted")

        msg, _, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        assert msg.artifact["tool_call_id"] == "c11"
        assert msg.artifact["tool_name"] == "mcp_slack_post"
        assert msg.artifact["tool_source"]["source"] == "mcp"
        assert msg.artifact["tool_source"]["canonical_name"] == "mcp_slack_post"


class TestInterruptHelperDefensiveBranch:
    """Unit test for the module-level defensive pre-check extracted from
    ``interrupt_helper``. Tests the branch where the node is routed to
    without valid ``pending_ask_*`` state.
    """

    def test_missing_pending_id_returns_command_back_to_tool_node(self):
        from langgraph.types import Command

        from app.domain.services.graphs.react_graph import (
            _interrupt_helper_early_return,
        )

        state: dict = {
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": {"tool_name": "x", "tool_source": {}},
        }

        result = _interrupt_helper_early_return(state)  # type: ignore[arg-type]

        assert isinstance(result, Command)
        assert result.goto == "tool_node"
        assert result.update == {}

    def test_missing_pending_artifact_returns_command_back_to_tool_node(self):
        from langgraph.types import Command

        from app.domain.services.graphs.react_graph import (
            _interrupt_helper_early_return,
        )

        state: dict = {
            "pending_ask_tool_call_id": "call_X",
            "pending_ask_artifact": None,
        }

        result = _interrupt_helper_early_return(state)  # type: ignore[arg-type]

        assert isinstance(result, Command)
        assert result.goto == "tool_node"
        assert result.update == {}

    def test_both_missing_returns_command_back_to_tool_node(self):
        from app.domain.services.graphs.react_graph import (
            _interrupt_helper_early_return,
        )

        result = _interrupt_helper_early_return({})  # type: ignore[arg-type]

        assert result is not None
        assert result.goto == "tool_node"

    def test_both_present_returns_none_to_proceed(self):
        from app.domain.services.graphs.react_graph import (
            _interrupt_helper_early_return,
        )

        state: dict = {
            "pending_ask_tool_call_id": "call_X",
            "pending_ask_artifact": {
                "tool_name": "risk_skill_b",
                "tool_source": {
                    "source": "native",
                    "category": "shell",
                    "canonical_name": "risk_skill_b",
                },
            },
        }

        result = _interrupt_helper_early_return(state)  # type: ignore[arg-type]

        assert result is None


class TestBuildReactGraphTopology:
    """R2 CS2 topology: pre_llm → llm → tool_node → (pre_llm | interrupt_helper | END)
    and interrupt_helper → tool_node on resume.

    ``tool_node`` is a closure inside ``build_react_graph`` so we cannot
    import it directly; we check the compiled graph shape instead.
    """

    def test_graph_compiles_with_interrupt_helper_node(self):
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.tools import tool as lc_tool

        from app.domain.services.graphs.react_graph import build_react_graph

        @lc_tool
        async def shell_execute(command: str) -> str:
            """execute shell"""
            return "ok"

        adapter = AsyncMock()
        adapter.bind_tools = MagicMock(return_value=adapter)

        graph = build_react_graph(adapter, [shell_execute])

        node_names = set(graph.get_graph().nodes.keys())
        assert "tool_node" in node_names
        assert "interrupt_helper" in node_names
        assert "pre_llm_node" in node_names
        assert "llm_node" in node_names


class TestTranslateOutcomePassthroughCap:
    """Passthrough block count ≤ _MAX_FILE_VIEW_IMAGES, excess replaced with omitted text."""

    def test_blocks_over_cap_are_omitted_with_marker(self, monkeypatch):
        from app.domain.services.graphs import react_graph

        monkeypatch.setattr(react_graph, "_MAX_FILE_VIEW_IMAGES", 2)

        tc = _make_tool_call("c12", "file_view")
        src = ToolSource(
            source="native", category="file", canonical_name="file_view"
        )
        blocks = [
            ImageUrlBlock(image_url=ImageUrlPayload(url=f"data:image/png;base64,{i}"))
            for i in range(5)
        ]
        outcome = Passthrough(
            content="summary", data=MultimodalPayload(blocks=blocks)
        )

        _, deferred, _ = _run(
            _translate_outcome(
                outcome,
                tc,
                src,
                _make_session_ctx(),
                tool_result_max_chars=8000,
                guide_injector=None,
            )
        )

        human_content = deferred[0].content
        # 1 text header + 2 kept blocks + 1 omitted marker = 4 items
        assert len(human_content) == 4
        assert human_content[-1] == {
            "type": "text",
            "text": "[... 3 more images omitted]",
        }


# ---------------------------------------------------------------------------
# P2#6: _build_resume_error_command must include pending_id in prefix
# ---------------------------------------------------------------------------


class TestBuildResumeErrorCommand:
    """P2#6: _build_resume_error_command adds pending_id to completed_tool_call_prefix."""

    def _make_state(
        self,
        pending_id: str = "tc_abc",
        existing_prefix: list | None = None,
    ) -> dict:
        return {
            "pending_ask_tool_call_id": pending_id,
            "pending_ask_artifact": {"tool_name": "file_write"},
            "pending_ask_outcome": None,
            "pending_ask_tool_args": None,
            "completed_tool_call_prefix": existing_prefix or [],
        }

    def test_adds_pending_id_to_empty_prefix(self):
        from app.domain.services.graphs.react_graph import _build_resume_error_command

        state = self._make_state(pending_id="tc1", existing_prefix=[])
        cmd = _build_resume_error_command(state, Exception("policy conflict"))
        prefix = cmd.update.get("completed_tool_call_prefix", [])
        assert "tc1" in prefix, (
            "_build_resume_error_command must add pending_id to completed_tool_call_prefix"
        )

    def test_appends_to_existing_prefix(self):
        from app.domain.services.graphs.react_graph import _build_resume_error_command

        state = self._make_state(pending_id="tc2", existing_prefix=["tc0", "tc1"])
        cmd = _build_resume_error_command(state, Exception("conflict"))
        prefix = cmd.update.get("completed_tool_call_prefix", [])
        assert prefix == ["tc0", "tc1", "tc2"], (
            "_build_resume_error_command must preserve existing prefix entries"
        )

    def test_error_message_is_included(self):
        from app.domain.services.graphs.react_graph import _build_resume_error_command

        state = self._make_state(pending_id="tc3")
        cmd = _build_resume_error_command(state, Exception("nonce_mismatch"))
        messages = cmd.update.get("messages", [])
        assert len(messages) == 1
        assert "POLICY_CONFLICT" in messages[0].content
        assert "nonce_mismatch" in messages[0].content

    def test_pending_fields_cleared(self):
        from app.domain.services.graphs.react_graph import _build_resume_error_command

        state = self._make_state(pending_id="tc4")
        cmd = _build_resume_error_command(state, Exception("err"))
        assert cmd.update.get("pending_ask_tool_call_id") is None
        assert cmd.update.get("pending_ask_outcome") is None
        assert cmd.update.get("pending_ask_artifact") is None
        assert cmd.update.get("pending_ask_tool_args") is None
