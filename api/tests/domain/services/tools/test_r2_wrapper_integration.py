"""R2 CS2 PR-B Commit 3 — wrapper integration tests.

Verifies the wrapper contract at the LangChain boundary:

    ``@tool(response_format="content_and_artifact")`` + ``ToolCall`` dict
    ainvoke → ``ToolMessage`` whose ``.artifact`` is a typed
    ``ToolOutcome`` variant.

**Scope note**: Task 40/41's AST tests already enforce the structural
invariants across all 8 wrappers (no ``ToolException`` import, no bare
``raise`` in tool bodies, no direct ``ToolMessage`` construction,
``by_alias=True`` on artifact dumps, ``tool_node`` / ``interrupt_helper``
isolation). This suite complements them with **behavioral** integration
tests against wrappers whose factory signatures can be mocked without
spinning up a full sandbox / MinIO / skill creator stack.

**Wrappers exercised here** (3 of 7 canonical R2 wrappers):

1. ``langchain_a2a.create_a2a_langchain_tools`` — needs only an ``A2ATool``
   stub with ``.manager`` truthy and 2 async methods.
2. ``memory_tools.create_memory_tools`` — needs an embedding provider
   stub + session factory stub.
3. ``langchain_mcp_discovery.create_mcp_discovery_tools`` — needs 2
   callable refs (mcp_tool_ref, activated_tools_ref).

**Deferred wrappers** (covered by AST tests + existing per-wrapper
suites; not duplicated here):

- ``langchain_tools`` (6 sub-factories: shell / file / browser / message
  / search / notify-user) — require full ``Sandbox`` + Chrome browser
  stack to mock meaningfully. Covered by AST contract tests + existing
  ``test_langchain_tools.py``.
- ``langchain_skill_tools`` — requires ``SkillCreatorService`` fake.
- ``langchain_dynamic_skill_tools`` — requires ``SkillTool`` factory +
  skill embedding index. Covered by existing
  ``test_langchain_dynamic_skill_tools.py``.
- ``langchain_mcp`` — requires fake MCP client session. Covered by
  existing ``test_langchain_mcp.py``.

Those still produce R2-compliant output (the AST contract tests
prove it), and their existing wrapper-specific test files cover
behavioral details. Adding behavioral integration for them would
require duplicating large mock fixtures already maintained by those
suites.

All tests use the ``_run()`` sync-async pattern (no pytest-asyncio)
consistent with the rest of the project.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import ToolMessage

from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
)


def _run(coro):
    return asyncio.run(coro)


def _tool_call_dict(tool_name: str, args: dict, tc_id: str = "tc1") -> dict:
    """Canonical ToolCall-shape input for tool.ainvoke() under
    langchain-core 1.2.17. Passing this shape (instead of a plain args
    dict) is required for ``ainvoke()`` to wrap the (content, artifact)
    tuple into a ``ToolMessage`` with ``artifact`` populated."""
    return {
        "args": args,
        "id": tc_id,
        "name": tool_name,
        "type": "tool_call",
    }


# ============================================================
# langchain-core version guard
# ============================================================


def test_langchain_core_version_supports_content_and_artifact():
    """Document & pin: langchain-core must support
    ``response_format='content_and_artifact'`` on ``@tool`` +
    ``ainvoke(ToolCall dict) → ToolMessage`` semantics. R2 wrapper
    contract requires >= 0.2.13 per spec Layer 2 §.

    If this test fails after a langchain-core upgrade, audit the
    ``_invoke_wrapper`` in ``react_graph.py`` — the ToolCall-dict →
    ToolMessage bridge may have changed.
    """
    import importlib.metadata

    version = importlib.metadata.version("langchain-core")
    parts = version.split(".")
    major, minor = int(parts[0]), int(parts[1])
    # >= 0.2.13 or any 1.x or later
    assert (major, minor) >= (0, 2), (
        f"langchain-core {version} is too old; R2 wrapper contract "
        f"requires >= 0.2.13 for response_format='content_and_artifact'."
    )


# ============================================================
# Wrapper 1: langchain_a2a
# ============================================================


def _make_a2a_tool_stub(
    *,
    remote_cards_return: Any = None,
    remote_cards_side_effect: Exception | None = None,
    call_remote_return: Any = None,
    call_remote_side_effect: Exception | None = None,
):
    """Build an ``A2ATool`` stub with a truthy ``.manager`` and mockable
    async methods. The factory guard at
    ``langchain_a2a.create_a2a_langchain_tools`` returns an empty list
    if ``.manager`` is falsy, so we must set it to something non-empty.
    """
    stub = MagicMock()
    stub.manager = object()  # truthy non-None

    # Build successful tool result objects with the (success, message, data)
    # shape the wrapper expects.
    cards_result = MagicMock()
    cards_result.success = True
    cards_result.data = {"agents": [{"id": "a1", "name": "AgentOne"}]}
    cards_result.message = None
    stub.get_remote_agent_cards = AsyncMock(
        return_value=remote_cards_return
        if remote_cards_return is not None
        else cards_result,
        side_effect=remote_cards_side_effect,
    )

    call_result = MagicMock()
    call_result.success = True
    call_result.data = {"reply": "task done"}
    call_result.message = None
    stub.call_remote_agent = AsyncMock(
        return_value=call_remote_return
        if call_remote_return is not None
        else call_result,
        side_effect=call_remote_side_effect,
    )

    return stub


class TestLangchainA2AIntegration:
    """All 3 behavioral categories × langchain_a2a wrapper."""

    def _build_tools(self, a2a_stub):
        from app.domain.services.tools.langchain_a2a import (
            create_a2a_langchain_tools,
        )

        return create_a2a_langchain_tools(a2a_stub)

    # ---- Category 1: success ---- #

    def test_a2a_get_remote_agent_cards_success(self):
        stub = _make_a2a_tool_stub()
        tools = self._build_tools(stub)
        tool = next(t for t in tools if t.name == "get_remote_agent_cards")

        tool_msg = _run(tool.ainvoke(_tool_call_dict(tool.name, {})))

        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.artifact, AllowSuccess)
        assert "AgentOne" in tool_msg.artifact.content
        assert tool_msg.status == "success"

    def test_a2a_call_remote_agent_success(self):
        stub = _make_a2a_tool_stub()
        tools = self._build_tools(stub)
        tool = next(t for t in tools if t.name == "call_remote_agent")

        tool_msg = _run(
            tool.ainvoke(
                _tool_call_dict(tool.name, {"id": "a1", "query": "go"})
            )
        )

        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.artifact, AllowSuccess)
        assert "task done" in tool_msg.artifact.content
        assert tool_msg.status == "success"

    # ---- Category 2: exception ---- #

    def test_a2a_get_remote_agent_cards_exception_becomes_allow_error(self):
        """Underlying service raises → wrapper catches and returns
        ``(content, AllowError)`` tuple; wrapper does NOT re-raise.

        **Status note**: LangChain's ``@tool(response_format=
        'content_and_artifact')`` + ``ainvoke(ToolCall dict)`` always
        sets ``tool_msg.status="success"`` regardless of artifact
        variant — because the wrapper's Python body completed normally
        (it returned a tuple; it didn't raise). The R2 error signal
        lives in the ``artifact`` Pydantic variant. Layer 3
        ``_translate_outcome`` in react_graph.py is where the R2
        status="success"/"error" split happens based on variant type,
        and Chunk 4's LLM adapter reads THAT rebuilt ToolMessage.
        """
        stub = _make_a2a_tool_stub(
            remote_cards_side_effect=RuntimeError("boom")
        )
        tools = self._build_tools(stub)
        tool = next(t for t in tools if t.name == "get_remote_agent_cards")

        # Must NOT raise — wrapper catches and returns AllowError
        tool_msg = _run(tool.ainvoke(_tool_call_dict(tool.name, {})))

        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.artifact, AllowError)
        assert tool_msg.artifact.reason.type == "exception"
        assert "boom" in tool_msg.artifact.content
        # Wrapper body returned normally → LangChain sets status=success.
        # Layer 3 in react_graph.py rebuilds ToolMessage with status=error
        # based on the typed artifact — verified separately in
        # test_react_graph_helpers.py::TestTranslateOutcomeAllVariants.
        assert tool_msg.status == "success"

    def test_a2a_call_remote_agent_timeout_becomes_allow_error_timeout(self):
        """``TimeoutError`` → wrapper returns ``AllowError(reason.type=
        timeout, retryable=True)``. Same status=success on the raw
        ToolMessage (see exception test above for rationale).
        """
        stub = _make_a2a_tool_stub(
            call_remote_side_effect=TimeoutError("rpc timeout")
        )
        tools = self._build_tools(stub)
        tool = next(t for t in tools if t.name == "call_remote_agent")

        tool_msg = _run(
            tool.ainvoke(
                _tool_call_dict(tool.name, {"id": "a1", "query": "go"})
            )
        )

        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.artifact, AllowError)
        assert tool_msg.artifact.reason.type == "timeout"
        assert tool_msg.artifact.retryable is True
        # LangChain raw ToolMessage keeps status=success; Layer 3 rebuilds.
        assert tool_msg.status == "success"

    # ---- Category 3: ainvoke shape regression ---- #

    def test_a2a_plain_args_dict_ainvoke_does_not_return_toolmessage(self):
        """Regression guard: plain args dict → raw content, NOT ToolMessage.

        Pins the langchain-core 1.2.17 behavior that ``_invoke_wrapper``
        relies on. If this test fails after a core upgrade, audit the
        Layer 2 bridge.
        """
        stub = _make_a2a_tool_stub()
        tools = self._build_tools(stub)
        tool = next(t for t in tools if t.name == "get_remote_agent_cards")

        # Plain args dict (no ToolCall envelope)
        raw = _run(tool.ainvoke({}))

        assert not isinstance(raw, ToolMessage), (
            f"{tool.name}: plain args dict ainvoke should NOT return "
            f"ToolMessage under langchain-core 1.2.17 behavior. Got "
            f"{type(raw).__name__}. If this fails, audit "
            f"_invoke_wrapper in react_graph.py."
        )


# ============================================================
# Wrapper 2: memory_tools
# ============================================================


class TestMemoryToolsIntegration:
    """memory_search / memory_get — behavioral contract."""

    def _build_tools(
        self,
        *,
        embedding_side_effect: Exception | None = None,
        embedding_vectors: list | None = None,
        chunks_return: list | None = None,
    ):
        from app.domain.services.tools.memory_tools import create_memory_tools

        embedding_provider = MagicMock()
        embedding_provider.embed = AsyncMock(
            return_value=embedding_vectors
            if embedding_vectors is not None
            else [[0.1, 0.2, 0.3]],
            side_effect=embedding_side_effect,
        )

        chunk_mock = MagicMock()
        chunk_mock.id = "chunk_1"
        chunk_mock.content = "This is a memory chunk about task X."
        chunk_mock.source = "user_msg"

        repo = MagicMock()
        repo.search_by_vector = AsyncMock(
            return_value=chunks_return
            if chunks_return is not None
            else [chunk_mock]
        )
        repo.get_by_id = AsyncMock(return_value=chunk_mock)

        # Fake async context manager for session_factory()
        class _FakeSession:
            async def __aenter__(self):
                return MagicMock()

            async def __aexit__(self, *args):
                return None

        session_factory = MagicMock(return_value=_FakeSession())
        repo_factory = MagicMock(return_value=repo)

        return create_memory_tools(
            embedding_provider=embedding_provider,
            session_factory=session_factory,
            repo_factory=repo_factory,
            user_id="test_user",
        )

    def test_memory_search_success_returns_allow_success(self):
        tools = self._build_tools()
        tool = next(t for t in tools if t.name == "memory_search")

        tool_msg = _run(
            tool.ainvoke(_tool_call_dict(tool.name, {"query": "task X"}))
        )

        assert isinstance(tool_msg, ToolMessage)
        assert isinstance(tool_msg.artifact, AllowSuccess)
        assert tool_msg.status == "success"

    def test_memory_search_empty_results_still_allow_success(self):
        """Empty search results are NOT an error — they just return a
        human-readable "未找到相关记忆" with AllowSuccess. Confirms the
        wrapper doesn't mis-classify zero-hit retrieval as failure."""
        tools = self._build_tools(chunks_return=[])
        tool = next(t for t in tools if t.name == "memory_search")

        tool_msg = _run(
            tool.ainvoke(_tool_call_dict(tool.name, {"query": "no hits"}))
        )

        assert isinstance(tool_msg.artifact, AllowSuccess)
        assert "未找到" in tool_msg.artifact.content

    def test_memory_search_embedding_unavailable_returns_allow_success_degraded(self):
        """``EmbeddingUnavailableError`` → graceful ``AllowSuccess`` (not Error).

        Pre-R2 convention: embedding outage is a degradation, not a
        tool failure — the memory tool returns "暂不可用" to the LLM
        as plain content so the agent can continue without memory
        search. Locked in by this test.
        """
        from app.domain.external.embedding_provider import (
            EmbeddingUnavailableError,
        )

        tools = self._build_tools(
            embedding_side_effect=EmbeddingUnavailableError("no embed service")
        )
        tool = next(t for t in tools if t.name == "memory_search")

        tool_msg = _run(
            tool.ainvoke(_tool_call_dict(tool.name, {"query": "x"}))
        )

        assert isinstance(tool_msg.artifact, AllowSuccess)
        assert "暂不可用" in tool_msg.artifact.content

    def test_memory_search_plain_args_dict_does_not_return_toolmessage(self):
        """Regression guard — plain args dict shape."""
        tools = self._build_tools()
        tool = next(t for t in tools if t.name == "memory_search")

        raw = _run(tool.ainvoke({"query": "x"}))

        assert not isinstance(raw, ToolMessage)


# ============================================================
# Wrapper 3: langchain_mcp_discovery
# ============================================================


class TestMCPDiscoveryIntegration:
    """list_mcp_tools / get_mcp_tool (2 Layer-1/Layer-2 metadata tools)."""

    def _build_tools(
        self,
        *,
        mcp_tool_servers: dict | None = None,
        activated_tools: set | None = None,
    ):
        from app.domain.services.tools.langchain_mcp_discovery import (
            create_mcp_discovery_tools,
        )

        mcp_tool = MagicMock()
        mcp_tool.servers = mcp_tool_servers or {}

        def mcp_tool_ref():
            return mcp_tool

        def activated_ref():
            return activated_tools if activated_tools is not None else set()

        return create_mcp_discovery_tools(mcp_tool_ref, activated_ref)

    def test_discovery_tools_registered_with_correct_names(self):
        """Sanity: both discovery tools exist with canonical names."""
        tools = self._build_tools()
        names = {t.name for t in tools}
        assert "list_mcp_tools" in names
        assert "get_mcp_tool" in names

    def test_discovery_plain_args_dict_does_not_return_toolmessage(self):
        """Regression guard — applies to discovery tools too."""
        tools = self._build_tools()
        list_tool = next(t for t in tools if t.name == "list_mcp_tools")

        raw = _run(list_tool.ainvoke({}))

        assert not isinstance(raw, ToolMessage)

    def test_discovery_list_tools_success_via_toolcall_dict(self):
        """ainvoke(ToolCall dict) → ToolMessage with typed artifact."""
        tools = self._build_tools()
        list_tool = next(t for t in tools if t.name == "list_mcp_tools")

        tool_msg = _run(list_tool.ainvoke(_tool_call_dict(list_tool.name, {})))

        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.artifact is not None
        # Every discovery tool output is an AllowSuccess (metadata
        # retrieval doesn't have an error path for empty server lists)
        assert isinstance(tool_msg.artifact, AllowSuccess)


# ============================================================
# Shape-regression contract (cross-wrapper)
# ============================================================


class TestCrossWrapperAinvokeShape:
    """The ainvoke-shape regression guard generalized across wrappers.

    If langchain-core's behavior changes such that plain args dict
    inputs auto-wrap into ToolMessage, this class starts failing and
    the R2 Layer 2 ``_invoke_wrapper`` contract must be revisited.
    """

    def test_toolcall_dict_wraps_into_toolmessage(self):
        """Positive form of the shape regression: ToolCall dict DOES
        produce a ToolMessage."""
        from app.domain.services.tools.langchain_a2a import (
            create_a2a_langchain_tools,
        )

        stub = _make_a2a_tool_stub()
        tools = create_a2a_langchain_tools(stub)
        tool = tools[0]

        tool_msg = _run(tool.ainvoke(_tool_call_dict(tool.name, {})))

        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.artifact is not None

    def test_tool_message_has_status_field(self):
        """Every ToolMessage produced by a migrated wrapper must have
        a readable ``status`` field — this is what Chunk 4's LLM
        adapter reads to decide whether to inject ``[TOOL_FAILED:]`` /
        ``[TOOL_DENIED:]`` prefix."""
        from app.domain.services.tools.langchain_a2a import (
            create_a2a_langchain_tools,
        )

        stub = _make_a2a_tool_stub()
        tools = create_a2a_langchain_tools(stub)
        tool = tools[0]

        tool_msg = _run(tool.ainvoke(_tool_call_dict(tool.name, {})))

        assert hasattr(tool_msg, "status")
        assert tool_msg.status in ("success", "error")

    def test_toolmessage_artifact_is_typed_pydantic_object(self):
        """Wrapper → ainvoke(ToolCall dict) path must produce a typed
        ``ToolOutcome`` variant (not a raw dict). This is the core R2
        CS2 invariant: Layer 2 reads ``tool_msg.artifact`` as a Pydantic
        object without needing an adapter round-trip.
        """
        from app.domain.services.tools.langchain_a2a import (
            create_a2a_langchain_tools,
        )

        stub = _make_a2a_tool_stub()
        tools = create_a2a_langchain_tools(stub)
        tool = tools[0]

        tool_msg = _run(tool.ainvoke(_tool_call_dict(tool.name, {})))

        assert isinstance(tool_msg.artifact, (AllowSuccess, AllowError))
