"""R1 CS1 — pin the enrichable vs identity-only category partition.

Runs on every CI build to keep `_handle_tool_event` in sync with the
KNOWN_CATEGORIES contract. If someone adds a new category, this test
fails until they decide which side of the partition it lives on.

R4 CS3 addendum (Task 14+): this file also hosts the projector-migration
enrichment contract tests. Tasks 15/16/17 add 14 contract + 5 regression
tests leveraging `_build_minimal_runner()` below.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.event import ToolEvent, ToolEventStatus
from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.tools.tool_source_resolver import KNOWN_CATEGORIES

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


ENRICHABLE_CATEGORIES: frozenset[str] = frozenset({
    "browser", "search", "shell", "file",
    "mcp", "a2a", "skill", "skill creator",
})
"""Categories where _handle_tool_event triggers UI enrichment."""

IDENTITY_ONLY_CATEGORIES: frozenset[str] = frozenset({
    "message", "memory", "mcp discovery", "skill guide",
    # R2 CS2: sentinel for LLM-hallucinated tool names. Listed here so
    # the partition invariant keeps holding — no _handle_tool_event
    # branch matches "unknown", so the raw error message surfaces
    # instead of triggering shell / browser / file side effects.
    "unknown",
})
"""Categories that exist in the contract but intentionally skip enrichment.

These are not bugs — their tool outputs are plain text that needs no
structured UI representation. R1 documents this as design, not accident.
"""


class TestCategoryPartition:
    def test_partition_covers_all_known_categories(self) -> None:
        """Union of enrichable + identity-only == KNOWN_CATEGORIES."""
        assert (ENRICHABLE_CATEGORIES | IDENTITY_ONLY_CATEGORIES) == KNOWN_CATEGORIES

    def test_partition_is_disjoint(self) -> None:
        """A category is either enrichable or identity-only, not both."""
        assert not (ENRICHABLE_CATEGORIES & IDENTITY_ONLY_CATEGORIES)

    def test_skill_creator_uses_space_not_underscore(self) -> None:
        """Regression: pre-R1 agent_task_runner used 'skill_creation' (underscore).
        R1 canonical is 'skill creator' (space). KNOWN_CATEGORIES must agree."""
        assert "skill creator" in KNOWN_CATEGORIES
        assert "skill_creation" not in KNOWN_CATEGORIES
        assert "skill creator" in ENRICHABLE_CATEGORIES


# ============================================================
# R4 CS3 Task 14: minimal runner fixture helper + smoke tests
# ============================================================


def _build_minimal_runner() -> AgentTaskRunner:
    """最小 AgentTaskRunner 实例, 不初始化 sandbox / event_bridge / skill_tool.

    _handle_tool_event 只依赖 self._sandbox / self._sync_generated_files /
    self._sync_file_to_storage / self._get_browser_screenshot, 这里 AsyncMock 全部.

    Pattern 参考: tests/domain/services/test_agent_task_runner_file_sync.py:17-49
    - object.__new__(AgentTaskRunner) 绕过 __init__ (避免 Docker / sandbox lifecycle)
    - AsyncMock 注入所有 collaborator

    Tasks 15/16/17 会在这个 helper 基础上加 14 contract + 5 regression 测试.
    """
    runner = object.__new__(AgentTaskRunner)
    _sb = AsyncMock()
    _sb.read_shell_output = AsyncMock(return_value=MagicMock(data={}))
    _sb.read_file = AsyncMock(return_value=MagicMock(data={}))
    runner._sandbox_accessor = EagerSandboxAccessor(_sb)
    runner._sync_generated_files = AsyncMock()
    runner._sync_file_to_storage = AsyncMock()
    runner._get_browser_screenshot = AsyncMock(return_value="data:image/png;base64,xyz")
    return runner


class TestFixtureSmoke:
    """证明 _build_minimal_runner() 可跑; 实际 enrichment test 在 Task 15/16/17 填入."""

    async def test_runner_builds_without_error(self) -> None:
        runner = _build_minimal_runner()
        assert runner is not None
        assert callable(runner._handle_tool_event)

    async def test_calling_event_no_op(self) -> None:
        """status=CALLING 不触发 enrichment, _handle_tool_event 立即 return."""
        runner = _build_minimal_runner()
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={},
            status=ToolEventStatus.CALLING,
        )
        await runner._handle_tool_event(evt)
        # CALLING 事件 tool_content 保持 None
        assert evt.tool_content is None


# ============================================================
# R4 CS3 Task 15: search / mcp / a2a enrichment contract tests
# ============================================================


class TestEnrichmentSearch:
    """Search enrichment must consume projector output (envelope.function_result.data)
    as dict with "results" list, using SearchResultItem.model_validate to rebuild.

    P1.3 bug context (Round 2e): current code uses hasattr(data, "results") which
    is always False for dict input — pre-migration search tool never enriches correctly.
    """

    async def test_search_success_fills_results(self) -> None:
        from app.domain.models.event import SearchToolContent
        from app.domain.models.search import SearchResultItem
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="search", canonical_name="search_web")
        outcome = AllowSuccess(
            content="done",
            data={"results": [{"title": "t1", "url": "u1", "snippet": "s1"}]},
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="search_web", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="search", function_name="search_web",
            function_args={"query": "q"}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
            function_result=None,
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, SearchToolContent)
        assert len(evt.tool_content.results) == 1
        assert evt.tool_content.results[0].title == "t1"

    async def test_search_error_produces_empty_results(self) -> None:
        from app.domain.models.event import SearchToolContent
        from app.domain.models.tool_result import AllowError, DecisionReason, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="search", canonical_name="search_web")
        outcome = AllowError(
            content="search failed",
            reason=DecisionReason(type="exception", code="network"),
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="search_web", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="search", function_name="search_web",
            function_args={"query": "q"}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, SearchToolContent)
        assert evt.tool_content.results == []

    async def test_search_end_to_end_through_real_producer(self) -> None:
        """Gate 2.5 P1: real producer chain SearchResults → _wrap_result_outcome
        → ToolOutcome → ToolArtifact → projector → enrichment must preserve results.

        Pre-fix: `_wrap_result_outcome` only kept `dict` for AllowSuccess.data,
        silently dropping SearchResults BaseModel payload. Fix: coerce BaseModel
        via `model_dump(mode="json")`.
        """
        from app.domain.external.search import ToolResult as SearchToolResult  # type: ignore[attr-defined]
        from app.domain.models.event import SearchToolContent
        from app.domain.models.search import SearchResultItem, SearchResults
        from app.domain.models.tool_result import ToolArtifact
        from app.domain.services.tools.langchain_tools import _wrap_result_outcome
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="search", canonical_name="search_web")

        # Simulate Bing search producer: ToolResult(success=True, data=SearchResults(...))
        sr = SearchResults(
            query="foo",
            total_results=1,
            results=[SearchResultItem(url="u1", title="t1", snippet="s1")],
        )
        # Local ToolResult shape: we use a plain namespace-style object mimicking legacy
        class _LegacyToolResult:
            def __init__(self, success: bool, data: object, message: str = "") -> None:
                self.success = success
                self.data = data
                self.message = message

        tool_result = _LegacyToolResult(success=True, data=sr)
        outcome = _wrap_result_outcome(tool_result, default_success_message="done")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="search_web", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="search", function_name="search_web",
            function_args={"query": "foo"}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, SearchToolContent)
        # BEFORE fix: would assert == [] (empty — BaseModel dropped).
        # AFTER fix: 1 result preserved.
        assert len(evt.tool_content.results) == 1
        assert evt.tool_content.results[0].title == "t1"
        assert evt.tool_content.results[0].url == "u1"


class TestEnrichmentMCP:
    async def test_mcp_success_fills_result_from_data(self) -> None:
        from app.domain.models.event import MCPToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="mcp", category="mcp", canonical_name="mcp_foo")
        outcome = AllowSuccess(content="done", data={"payload": "hello mcp"})
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="mcp_foo", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="mcp", function_name="mcp_foo",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, MCPToolContent)
        assert evt.tool_content.result == {"payload": "hello mcp"}

    async def test_mcp_error_fills_result_from_message(self) -> None:
        from app.domain.models.event import MCPToolContent
        from app.domain.models.tool_result import AllowError, DecisionReason, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="mcp", category="mcp", canonical_name="mcp_foo")
        outcome = AllowError(
            content="mcp tool rejected request",
            reason=DecisionReason(type="exception", code="rejected"),
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="mcp_foo", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="mcp", function_name="mcp_foo",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, MCPToolContent)
        assert evt.tool_content.result == "mcp tool rejected request"


class TestEnrichmentA2A:
    async def test_a2a_success_fills_result_from_data(self) -> None:
        from app.domain.models.event import A2AToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="a2a", category="a2a", canonical_name="a2a_agent")
        outcome = AllowSuccess(content="agent ok", data={"agent_output": "hi"})
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="a2a_agent", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="a2a", function_name="a2a_agent",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, A2AToolContent)
        assert evt.tool_content.a2a_result == {"agent_output": "hi"}

    async def test_a2a_error_fills_result_from_message(self) -> None:
        from app.domain.models.event import A2AToolContent
        from app.domain.models.tool_result import AllowError, DecisionReason, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="a2a", category="a2a", canonical_name="a2a_agent")
        outcome = AllowError(
            content="a2a agent timed out",
            reason=DecisionReason(type="timeout", code="step_timeout"),
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="a2a_agent", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="a2a", function_name="a2a_agent",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, A2AToolContent)
        assert evt.tool_content.a2a_result == "a2a agent timed out"


# ============================================================
# R4 CS3 Task 16: shell / file / skill / skill_creator enrichment contract tests
# ============================================================


class TestEnrichmentShell:
    """Shell enrichment is args-driven (session_id, exec_dir); does not depend on function_result."""

    async def test_shell_success_reads_console_and_syncs_exec_dir(self) -> None:
        from app.domain.models.event import ShellToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        outcome = AllowSuccess(content="ls output")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="shell_execute", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={"session_id": "sid1", "exec_dir": "/tmp/work"},
            status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        runner._sandbox_accessor.peek().read_shell_output = AsyncMock(
            return_value=MagicMock(data={"console_records": [{"cmd": "ls"}]})
        )

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, ShellToolContent)
        assert evt.tool_content.console == [{"cmd": "ls"}]
        runner._sync_generated_files.assert_awaited_once_with("/tmp/work")

    async def test_shell_denied_still_reads_console(self) -> None:
        """Shell enrichment runs regardless of outcome — args-driven."""
        from app.domain.models.event import ShellToolContent
        from app.domain.models.tool_result import Denied, DecisionReason, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        outcome = Denied(
            content="blocked",
            reason=DecisionReason(type="ast_validator", code="rm_rf"),
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="shell_execute", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={"session_id": "sid1"},
            status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        runner._sandbox_accessor.peek().read_shell_output = AsyncMock(
            return_value=MagicMock(data={"console_records": []})
        )

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, ShellToolContent)
        assert evt.tool_content.console == []


class TestEnrichmentFile:
    async def test_file_read_success(self) -> None:
        from app.domain.models.event import FileToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="file", canonical_name="file_view")
        outcome = AllowSuccess(content="file viewed")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="file_view", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="file", function_name="file_view",
            function_args={"filepath": "/home/ubuntu/x.txt"},
            status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        runner._sandbox_accessor.peek().read_file = AsyncMock(return_value=MagicMock(data={"content": "hello"}))

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, FileToolContent)
        assert evt.tool_content.content == "hello"
        runner._sync_file_to_storage.assert_awaited_once_with("/home/ubuntu/x.txt")

    async def test_file_no_filepath_placeholder(self) -> None:
        from app.domain.models.event import FileToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="file", canonical_name="file_list")
        outcome = AllowSuccess(content="listed")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="file_list", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="file", function_name="file_list",
            function_args={},  # no filepath
            status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, FileToolContent)
        assert evt.tool_content.content == "(No Content)"


class TestEnrichmentSkill:
    async def test_skill_shell_session_id_reads_console(self) -> None:
        """Native skill with shell_session_id routes to ShellToolContent for UI terminal panel."""
        from app.domain.models.event import ShellToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="skill", canonical_name="skill_xyz")
        outcome = AllowSuccess(
            content="skill done",
            data={"shell_session_id": "ssid42", "exec_dir": "/workspace"},
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="skill_xyz", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="skill", function_name="skill_xyz",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()
        runner._sandbox_accessor.peek().read_shell_output = AsyncMock(
            return_value=MagicMock(data={"console_records": [{"out": "stdout"}]})
        )

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, ShellToolContent)
        assert evt.tool_content.console == [{"out": "stdout"}]
        # Skill exec_dir also triggers sync
        runner._sync_generated_files.assert_awaited_once_with("/workspace")

    async def test_skill_failure_fills_message_placeholder(self) -> None:
        from app.domain.models.event import SkillToolContent
        from app.domain.models.tool_result import AllowError, DecisionReason, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="skill", canonical_name="skill_xyz")
        outcome = AllowError(
            content="skill crashed",
            reason=DecisionReason(type="exception", code="runtime"),
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="skill_xyz", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="skill", function_name="skill_xyz",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, SkillToolContent)
        assert evt.tool_content.skill_result == "skill crashed"


class TestEnrichmentSkillCreator:
    async def test_skill_creator_success_fills_data(self) -> None:
        from app.domain.models.event import SkillToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="skill", category="skill creator", canonical_name="skill_creator")
        outcome = AllowSuccess(content="created", data={"skill_id": "abc"})
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="skill_creator", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="skill creator", function_name="skill_creator",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, SkillToolContent)
        assert evt.tool_content.skill_result == {"skill_id": "abc"}

    async def test_skill_creator_failure_fills_message(self) -> None:
        from app.domain.models.event import SkillToolContent
        from app.domain.models.tool_result import AllowError, DecisionReason, ToolArtifact
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="skill", category="skill creator", canonical_name="skill_creator")
        outcome = AllowError(
            content="create failed",
            reason=DecisionReason(type="exception", code="invalid"),
        )
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="skill_creator", tool_source=ts, outcome=outcome,
        )
        evt = ToolEvent(
            tool_call_id="c1", tool_name="skill creator", function_name="skill_creator",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner = _build_minimal_runner()

        await runner._handle_tool_event(evt)

        assert isinstance(evt.tool_content, SkillToolContent)
        assert evt.tool_content.skill_result == "create failed"


# ============================================================
# R4 CS3 Task 17: tool_content channel binary-equality regression
# ============================================================


class TestToolContentChannelBinaryEquality:
    """Lock the invariant that pre-R4 and R4 tool events produce identical
    `tool_content` after enrichment, for semantically equivalent inputs.

    Ensures the projector migration doesn't drift the frontend-visible
    `tool_content` shape — SSE output must be bit-for-bit compatible with
    the pre-R4 shape.
    """

    async def test_browser_screenshot_content_binary_equal(self) -> None:
        """Browser: screenshot is args-driven + sandbox-sourced. Independent of
        function_result shape."""
        from app.domain.models.event import BrowserToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact, ToolResult
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="browser", canonical_name="browser_view")

        # Pre-R4 event (legacy: function_result only)
        pre_evt = ToolEvent(
            tool_call_id="c1", tool_name="browser", function_name="browser_view",
            function_args={"url": "https://x"}, status=ToolEventStatus.CALLED,
            function_result=ToolResult(success=True, message="visited"),
        )
        # R4 event (artifact, no function_result)
        outcome = AllowSuccess(content="visited")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="browser_view", tool_source=ts, outcome=outcome,
        )
        r4_evt = ToolEvent(
            tool_call_id="c1", tool_name="browser", function_name="browser_view",
            function_args={"url": "https://x"}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner1 = _build_minimal_runner()
        await runner1._handle_tool_event(pre_evt)
        runner2 = _build_minimal_runner()
        await runner2._handle_tool_event(r4_evt)

        assert isinstance(pre_evt.tool_content, BrowserToolContent)
        assert isinstance(r4_evt.tool_content, BrowserToolContent)
        assert pre_evt.tool_content.screenshot == r4_evt.tool_content.screenshot

    async def test_shell_console_records_binary_equal(self) -> None:
        """Shell: console records come from sandbox.read_shell_output. Binary equal
        across pre-R4 and R4 events with same function_args."""
        from app.domain.models.event import ShellToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact, ToolResult
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")

        pre_evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={"session_id": "sid1"}, status=ToolEventStatus.CALLED,
            function_result=ToolResult(success=True, message="ls"),
        )
        outcome = AllowSuccess(content="ls")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="shell_execute", tool_source=ts, outcome=outcome,
        )
        r4_evt = ToolEvent(
            tool_call_id="c1", tool_name="shell", function_name="shell_execute",
            function_args={"session_id": "sid1"}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner1 = _build_minimal_runner()
        runner1._sandbox_accessor.peek().read_shell_output = AsyncMock(
            return_value=MagicMock(data={"console_records": [{"cmd": "ls"}]})
        )
        await runner1._handle_tool_event(pre_evt)

        runner2 = _build_minimal_runner()
        runner2._sandbox_accessor.peek().read_shell_output = AsyncMock(
            return_value=MagicMock(data={"console_records": [{"cmd": "ls"}]})
        )
        await runner2._handle_tool_event(r4_evt)

        assert isinstance(pre_evt.tool_content, ShellToolContent)
        assert isinstance(r4_evt.tool_content, ShellToolContent)
        assert pre_evt.tool_content.console == r4_evt.tool_content.console

    async def test_file_content_binary_equal(self) -> None:
        """File: content comes from sandbox.read_file. Binary equal with same function_args."""
        from app.domain.models.event import FileToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact, ToolResult
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="file", canonical_name="file_view")

        pre_evt = ToolEvent(
            tool_call_id="c1", tool_name="file", function_name="file_view",
            function_args={"filepath": "/x.txt"}, status=ToolEventStatus.CALLED,
            function_result=ToolResult(success=True, message="read"),
        )
        outcome = AllowSuccess(content="read")
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="file_view", tool_source=ts, outcome=outcome,
        )
        r4_evt = ToolEvent(
            tool_call_id="c1", tool_name="file", function_name="file_view",
            function_args={"filepath": "/x.txt"}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner1 = _build_minimal_runner()
        runner1._sandbox_accessor.peek().read_file = AsyncMock(return_value=MagicMock(data={"content": "hello"}))
        await runner1._handle_tool_event(pre_evt)

        runner2 = _build_minimal_runner()
        runner2._sandbox_accessor.peek().read_file = AsyncMock(return_value=MagicMock(data={"content": "hello"}))
        await runner2._handle_tool_event(r4_evt)

        assert isinstance(pre_evt.tool_content, FileToolContent)
        assert isinstance(r4_evt.tool_content, FileToolContent)
        assert pre_evt.tool_content.content == r4_evt.tool_content.content

    async def test_mcp_result_binary_equal(self) -> None:
        """MCP: result comes from fr.data (success case). Binary equal across pre-R4
        (ToolResult.data) and R4 (AllowSuccess.data via projector)."""
        from app.domain.models.event import MCPToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact, ToolResult
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="mcp", category="mcp", canonical_name="mcp_foo")
        shared_data = {"payload": "hello mcp", "nested": {"k": "v"}}

        pre_evt = ToolEvent(
            tool_call_id="c1", tool_name="mcp", function_name="mcp_foo",
            function_args={}, status=ToolEventStatus.CALLED,
            function_result=ToolResult(success=True, message="ok", data=shared_data),
        )
        outcome = AllowSuccess(content="ok", data=shared_data)
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="mcp_foo", tool_source=ts, outcome=outcome,
        )
        r4_evt = ToolEvent(
            tool_call_id="c1", tool_name="mcp", function_name="mcp_foo",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner1 = _build_minimal_runner()
        await runner1._handle_tool_event(pre_evt)
        runner2 = _build_minimal_runner()
        await runner2._handle_tool_event(r4_evt)

        assert isinstance(pre_evt.tool_content, MCPToolContent)
        assert isinstance(r4_evt.tool_content, MCPToolContent)
        assert pre_evt.tool_content.result == r4_evt.tool_content.result

    async def test_skill_result_binary_equal(self) -> None:
        """Skill: result comes from fr.data. Binary equal across pre-R4 and R4 for
        AllowSuccess outcome."""
        from app.domain.models.event import SkillToolContent
        from app.domain.models.tool_result import AllowSuccess, ToolArtifact, ToolResult
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="skill", canonical_name="skill_xyz")
        shared_data = {"key": "value", "count": 42}

        pre_evt = ToolEvent(
            tool_call_id="c1", tool_name="skill", function_name="skill_xyz",
            function_args={}, status=ToolEventStatus.CALLED,
            function_result=ToolResult(success=True, message="done", data=shared_data),
        )
        outcome = AllowSuccess(content="done", data=shared_data)
        artifact = ToolArtifact(
            tool_call_id="c1", tool_name="skill_xyz", tool_source=ts, outcome=outcome,
        )
        r4_evt = ToolEvent(
            tool_call_id="c1", tool_name="skill", function_name="skill_xyz",
            function_args={}, status=ToolEventStatus.CALLED,
            tool_source=ts,
            artifact=artifact.model_dump(mode="json", by_alias=True),
        )

        runner1 = _build_minimal_runner()
        await runner1._handle_tool_event(pre_evt)
        runner2 = _build_minimal_runner()
        await runner2._handle_tool_event(r4_evt)

        assert isinstance(pre_evt.tool_content, SkillToolContent)
        assert isinstance(r4_evt.tool_content, SkillToolContent)
        assert pre_evt.tool_content.skill_result == r4_evt.tool_content.skill_result
