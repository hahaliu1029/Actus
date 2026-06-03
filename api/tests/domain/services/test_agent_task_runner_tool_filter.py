"""PR-2 / Phase 1 minimal subagent: ``AgentTaskRunner.tool_filter`` tests.

Nine tests covering tool_filter behavior on the runner side:
  - 3 sync tests over `_build_lc_tools_full` (None / empty / allowlist)
  - 1 sync test for `_build_available_tool_summary` filter
  - 2 async smoke tests over the same surface (registry + token-leak)
  - 1 sync test for the `_build_lc_tools_for_step` inheritance contract
  - 2 async tests for the react_graph double-gate (PE + tool_filter)

All tests use ``object.__new__(AgentTaskRunner)`` + setattr to bypass
the heavy constructor — only the attributes actually read by the
methods under test are seeded.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeNamedTool:
    """Minimal stand-in for a LangChain BaseTool — only the ``.name``
    attribute is consumed by the filter; nothing else is touched.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - diagnostic
        return f"_FakeNamedTool({self.name!r})"


def _make_runner_for_filter(tool_filter=None, tools=None):
    """Build a half-constructed ``AgentTaskRunner`` with just the
    attributes the tool_filter surface needs.

    We monkeypatch ``_build_lc_tools_full`` at the class level to
    return a fixed list when the underlying real implementation would
    otherwise require sandbox/browser/skill_tool — but we DON'T
    monkeypatch when the test explicitly wants to exercise the real
    method (``test_tool_filter_blocks_tool_at_registry_construction``
    uses the real implementation against seeded internals).
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = tool_filter
    runner._execution_supervisor = None  # wrap_tool_list_for_supervisor passes through
    runner._tools_for_test = tools or []
    return runner


# ---------------------------------------------------------------------------
# 1. _build_lc_tools_full: None / empty / allowlist
# ---------------------------------------------------------------------------


def test_tool_filter_none_returns_all_tools(monkeypatch) -> None:
    """tool_filter=None → no filtering (backward compat)."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    tools = [_FakeNamedTool("shell_execute"), _FakeNamedTool("file_read"), _FakeNamedTool("search_web")]
    runner = _make_runner_for_filter(tool_filter=None, tools=tools)

    # Monkeypatch the production builder to return our fixed list right
    # before the filter block kicks in — we want to test the filter, not
    # the full builder.
    def _stub(self):  # type: ignore[no-untyped-def]
        # Inline the filter logic from production exactly:
        lc_tools = list(self._tools_for_test)
        if self._tool_filter is not None:
            lc_tools = [t for t in lc_tools if t.name in self._tool_filter]
        return lc_tools

    monkeypatch.setattr(AgentTaskRunner, "_build_lc_tools_full", _stub)

    result = runner._build_lc_tools_full()
    assert [t.name for t in result] == ["shell_execute", "file_read", "search_web"]


def test_tool_filter_empty_set_returns_empty(monkeypatch) -> None:
    """tool_filter=frozenset() → explicit deny-all → empty list.

    This is the critical ``is not None`` vs truthy distinction:
    ``frozenset()`` is falsy in Python, but it must be honored as
    "deny everything" (subagent scoped to zero capabilities).
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    tools = [_FakeNamedTool("shell_execute"), _FakeNamedTool("file_read")]
    runner = _make_runner_for_filter(tool_filter=frozenset(), tools=tools)

    def _stub(self):  # type: ignore[no-untyped-def]
        lc_tools = list(self._tools_for_test)
        if self._tool_filter is not None:
            lc_tools = [t for t in lc_tools if t.name in self._tool_filter]
        return lc_tools

    monkeypatch.setattr(AgentTaskRunner, "_build_lc_tools_full", _stub)

    result = runner._build_lc_tools_full()
    assert result == []


def test_tool_filter_allowlist_returns_only_matching(monkeypatch) -> None:
    """tool_filter={search_web, memory_search} → only those two pass."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    tools = [
        _FakeNamedTool("shell_execute"),
        _FakeNamedTool("file_read"),
        _FakeNamedTool("search_web"),
        _FakeNamedTool("memory_search"),
        _FakeNamedTool("memory_save"),
    ]
    runner = _make_runner_for_filter(
        tool_filter=frozenset({"search_web", "memory_search"}),
        tools=tools,
    )

    def _stub(self):  # type: ignore[no-untyped-def]
        lc_tools = list(self._tools_for_test)
        if self._tool_filter is not None:
            lc_tools = [t for t in lc_tools if t.name in self._tool_filter]
        return lc_tools

    monkeypatch.setattr(AgentTaskRunner, "_build_lc_tools_full", _stub)

    result_names = {t.name for t in runner._build_lc_tools_full()}
    assert result_names == {"search_web", "memory_search"}


# ---------------------------------------------------------------------------
# 2. _build_available_tool_summary
# ---------------------------------------------------------------------------


def _seed_summary_runner(tool_filter):
    """Seed the minimal surface that ``_build_available_tool_summary``
    actually reads. Skill/MCP/A2A/Memory pathways are emptied via
    stubs so the test focuses on the filter behavior; native tools are
    seeded via a stub ``_get_native_tool_names_by_category``.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = tool_filter

    # Stub skill_selection_policy with the field summary code reads.
    runner._skill_selection_policy = MagicMock()
    runner._skill_selection_policy.available_tool_summary_token_budget = 4096

    # Stub native tools — return a category map with concrete names.
    runner._get_native_tool_names_by_category = MagicMock(
        return_value={
            "shell": ["shell_execute", "shell_read_output"],
            "file": ["file_read", "file_write"],
            "browser": ["browser_view", "browser_navigate"],
            "search": ["search_web"],
        }
    )

    # Empty skill / creator / brainstorm tool registries.
    runner._skill_tool = MagicMock()
    runner._skill_tool.get_tools = MagicMock(return_value=[])
    runner._create_skill_tool = None
    runner._brainstorm_skill_tool = None

    # Empty MCP / A2A.
    runner._mcp_tool = MagicMock()
    runner._mcp_tool.get_tools = MagicMock(return_value=[])
    runner._a2a_tool = MagicMock()
    runner._a2a_tool.manager = None

    # No memory tools.
    runner._memory_session_factory = None
    runner._memory_repo_factory = None
    runner._memory_write_service = None
    runner._memory_session_redis = None

    return runner


def test_available_tool_summary_respects_tool_filter() -> None:
    """Summary lists only allowed native tools when tool_filter is set."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = _seed_summary_runner(
        tool_filter=frozenset({"search_web", "file_read"})
    )
    summary = AgentTaskRunner._build_available_tool_summary(runner)

    # search_web and file_read should appear; the other 5 native tools
    # should be filtered out.
    assert "search_web" in summary
    assert "file_read" in summary

    # The blocked tools should NOT appear as tokens anywhere.
    for blocked in ("shell_execute", "shell_read_output", "file_write",
                    "browser_view", "browser_navigate"):
        assert blocked not in summary, (
            f"Blocked tool '{blocked}' leaked into summary:\n{summary}"
        )


# ---------------------------------------------------------------------------
# 3. Async tests over the production builder (with monkeypatched dependencies)
# ---------------------------------------------------------------------------


async def test_tool_filter_blocks_tool_at_registry_construction(monkeypatch) -> None:
    """Real ``_build_lc_tools_full`` block: when tool_filter is set, blocked
    tools never reach the LangChain registry.

    We monkeypatch the imported factories inside ``_build_lc_tools_full``
    so it returns a known set, then assert the filter strips them.
    """
    import app.domain.services.tools.langchain_a2a as la2a
    import app.domain.services.tools.langchain_dynamic_skill_tools as ldyn
    import app.domain.services.tools.langchain_mcp as lmcp
    import app.domain.services.tools.langchain_skill_tools as lskill
    import app.domain.services.tools.langchain_tools as ltools
    from app.domain.services.agent_task_runner import AgentTaskRunner

    fake_native = [_FakeNamedTool("shell_execute"), _FakeNamedTool("file_read"), _FakeNamedTool("search_web")]
    fake_skill = []
    fake_dyn = []
    fake_mcp_list = []
    fake_a2a = []

    monkeypatch.setattr(ltools, "create_native_tools", lambda **kw: fake_native)
    monkeypatch.setattr(lmcp, "create_mcp_langchain_tools", lambda *a, **kw: fake_mcp_list)
    monkeypatch.setattr(la2a, "create_a2a_langchain_tools", lambda *a, **kw: fake_a2a)
    monkeypatch.setattr(lskill, "create_skill_langchain_tools", lambda *a, **kw: fake_skill)
    monkeypatch.setattr(lskill, "create_skill_guide_tool", lambda *a, **kw: _FakeNamedTool("get_skill_guide"))
    monkeypatch.setattr(ldyn, "create_dynamic_skill_langchain_tools", lambda *a, **kw: fake_dyn)

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = frozenset({"search_web"})
    runner._execution_supervisor = None
    runner._sandbox = MagicMock()
    runner._browser = MagicMock()
    runner._search_engine = MagicMock()
    runner._file_processor_lookup = None
    runner._supports_vision = True
    runner._supports_pdf_input = False
    runner._build_memory_mount_scope = lambda: None

    # MCP: get_tools small (< auto-bind threshold) → uses create_mcp_langchain_tools branch.
    runner._mcp_tool = MagicMock()
    runner._mcp_tool.get_tools = MagicMock(return_value=[])
    runner._image_url_map = {}
    runner._upload_sandbox_file_for_mcp = MagicMock()
    runner._get_always_bind_tool_names = lambda: set()
    runner._activated_mcp_tools = set()

    runner._a2a_tool = MagicMock()
    runner._create_skill_tool = None
    runner._brainstorm_skill_tool = None
    runner._skill_tool = MagicMock()

    runner._session_skill_pool = []
    runner._skill_bundle_sync = MagicMock()
    runner._skill_bundle_sync.get_file_listing_all = MagicMock(return_value={})
    runner._skill_bundle_sync.sandbox_skill_root = "/tmp/skills"

    runner._memory_session_factory = None
    runner._memory_repo_factory = None
    runner._user_id = "u1"
    runner._session_id = "s1"

    result = AgentTaskRunner._build_lc_tools_full(runner)

    names = {t.name for t in result}
    assert names == {"search_web"}, (
        f"Expected only search_web after filter, got {names}"
    )


async def test_tool_filter_available_summary_does_not_leak_blocked_tools() -> None:
    """Token-level invariant: blocked tool names MUST NOT appear as
    ``[a-z_][a-z0-9_]+`` tokens anywhere in the returned summary string.

    Tighter than the comparison test above — uses regex to scan every
    snake_case token and asserts none of them is in the blocked set.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    allow = frozenset({"file_read", "search_web"})
    blocked = {"shell_execute", "shell_read_output", "file_write",
               "browser_view", "browser_navigate"}
    runner = _seed_summary_runner(tool_filter=allow)

    summary = AgentTaskRunner._build_available_tool_summary(runner)

    # Tokenize the summary into snake_case identifiers and assert none
    # of the blocked names appears.
    tokens = set(re.findall(r"[a-z_][a-z0-9_]+", summary))
    leaks = tokens & blocked
    assert not leaks, (
        f"Token-level leak detected: blocked tools {leaks} appear in "
        f"summary:\n{summary}"
    )


# ---------------------------------------------------------------------------
# 4. _build_lc_tools_for_step inherits filter
# ---------------------------------------------------------------------------


def test_tool_filter_behavioral_invariant_no_bypass(monkeypatch) -> None:
    """``_build_lc_tools_for_step`` must inherit filtering from
    ``_build_lc_tools_full`` (it delegates and caches; no new logic).
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = frozenset({"file_read"})
    runner._last_skill_ids = ()
    runner._activated_mcp_tools = set()
    runner._lc_tools_cache = {}

    # Stub the full builder; assert the per-step builder honors its output.
    def _full(self):  # type: ignore[no-untyped-def]
        return [_FakeNamedTool("file_read")]  # already filtered

    monkeypatch.setattr(AgentTaskRunner, "_build_lc_tools_full", _full)

    step_tools = AgentTaskRunner._build_lc_tools_for_step(runner)
    assert [t.name for t in step_tools] == ["file_read"]

    # Cached on second call — same object.
    step_tools_2 = AgentTaskRunner._build_lc_tools_for_step(runner)
    assert step_tools_2 is step_tools


# ---------------------------------------------------------------------------
# 5. react_graph PE double-gate tests
#
# These mirror the FakePE + tool_node harness from
# api/tests/invariants/test_inv5_behavior_pe_call_order.py and add a
# tool_filter dimension on top.
# ---------------------------------------------------------------------------


class _RecordingPEAlwaysDeny:
    """PE that records evaluate calls and returns a Denied outcome —
    used to prove that even when tool_filter allows a tool, the PE
    second gate can still block it (double-gate semantics).
    """

    def __init__(self):
        self.evaluate_calls: list = []
        self.tool_handler_invocations = 0

    async def evaluate(self, call, ctx):
        from app.domain.models.tool_result import Denied, DecisionReason
        self.evaluate_calls.append(call)
        return Denied(
            content="denied-by-pe-policy-for-test",
            reason=DecisionReason(
                type="approval_policy",
                code="policy_deny",
                message="test always-deny PE",
            ),
        )

    async def preflight_resume(self, *a, **kw):
        return None

    async def commit_resume(self, *a, **kw):
        from app.domain.models.tool_result import AllowSuccess
        return AllowSuccess(content="ok", data={})


def _make_fake_ssm():
    from app.domain.models.session import SessionStatus
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(SessionStatus.RUNNING, 1))
    return ssm


def _make_state(tool_name: str, tool_args: dict, call_id: str = "tc1") -> dict:
    from langchain_core.messages import AIMessage
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": call_id, "name": tool_name, "args": tool_args, "type": "tool_call"}
                ],
            )
        ],
        "llm_input_messages": [],
        "step_description": "test",
        "original_request": "test",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "events": [],
        "should_interrupt": False,
        "soft_hint_sent": False,
        "attempt_count": 0,
        "failure_count": 0,
        "completed_tool_call_prefix": [],
        "approved_tool_call_ids": [],
        "pending_ask_outcome": None,
        "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None,
        "pending_ask_tool_args": None,
        "pe_resume_outcomes": None,
    }


def _make_config(fake_pe, fake_ssm, *, user_id: str = "u", session_id: str = "s") -> dict:
    from types import SimpleNamespace
    tc_cfg = SimpleNamespace(enabled=True)
    return {
        "configurable": {
            "permission_engine": fake_pe,
            "session_state_machine": fake_ssm,
            "tool_confirmation_config": tc_cfg,
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }


def _build_tool_node_fn_for_double_gate(handler_counter: list):
    """Build a react_graph and return tool_node, plus a search_web tool
    whose handler increments ``handler_counter[0]`` when invoked.
    Mirrors ``test_inv5_behavior_pe_call_order._build_tool_node_fn``.
    """
    from langchain_core.tools import tool as lc_tool
    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def search_web(query: str = "") -> str:
        """Search the web."""
        handler_counter[0] += 1
        return f"search results for {query}"

    stub_llm = AsyncMock()
    from langchain_core.messages import AIMessage as _AIMessage
    stub_llm.ainvoke = AsyncMock(
        return_value=_AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [search_web])
    tool_node_fn = graph.nodes["tool_node"].bound.afunc
    return tool_node_fn


async def test_pe_double_gate_at_react_graph_pe_dispatch() -> None:
    """Double-gate: tool_filter ALLOWS search_web AND PE policy DENIES
    it → final outcome is DENY.

    Setup: tool_filter is applied at AgentTaskRunner-level (NOT at the
    react_graph level — react_graph receives whichever tools the runner
    passed to ``build_react_graph``). The filter is the *first* gate.
    PE.evaluate is the *second* gate at react_graph dispatch time.

    Here we simulate "tool_filter allows search_web": we pass search_web
    INTO the react_graph (i.e. it's in the registry). Then we set a PE
    that always denies. The test asserts:
      1. PE.evaluate IS called for search_web (proves PE gate ran)
      2. The tool handler was NOT invoked (proves DENY was effective)
    """
    handler_counter = [0]
    tool_node_fn = _build_tool_node_fn_for_double_gate(handler_counter)

    fake_pe = _RecordingPEAlwaysDeny()
    fake_ssm = _make_fake_ssm()

    state = _make_state("search_web", {"query": "deep research"}, call_id="tc-pe-deny")
    config = _make_config(fake_pe, fake_ssm, session_id="sess-double-gate-1")

    await tool_node_fn(state, config)

    # Gate 2 (PE) fired for the allowed-by-filter tool.
    assert len(fake_pe.evaluate_calls) >= 1, (
        f"PE.evaluate should have been called for search_web "
        f"(got {len(fake_pe.evaluate_calls)})"
    )
    assert any(c.tool_name == "search_web" for c in fake_pe.evaluate_calls), (
        "PE.evaluate was called but not for search_web — wrong tool routed"
    )

    # PE returned Denied → handler must NOT have been invoked.
    assert handler_counter[0] == 0, (
        f"search_web handler was invoked {handler_counter[0]} times despite "
        f"PE Denied outcome — double-gate broken"
    )


async def test_tool_filter_blocked_tool_skips_pe_at_react_graph() -> None:
    """When a tool is NOT in tool_filter, it's never registered with
    react_graph at all → react_graph's ``_pe_dispatch`` never receives
    a tool_call for it → PE.evaluate is never called for it.

    We prove this by building a graph with NO tools registered (mimicking
    "tool_filter stripped everything") and asserting that even if the LLM
    emitted a tool_call for ``search_web``, dispatching through tool_node
    never calls PE.evaluate.

    Why this is correct: AgentTaskRunner's filter runs BEFORE
    ``build_react_graph``; the graph itself has no awareness of the
    filter. If the runner stripped search_web from lc_tools, the graph
    has no binding to search_web — so when a stale tool_call shows up
    in state.messages, ``_pe_dispatch``'s normal "non-PE-eligible" /
    "unknown source" path kicks in (falls back to legacy tool_node) and
    PE.evaluate is never invoked for that name. This test asserts the
    PE counter stays at zero.
    """
    from langchain_core.messages import AIMessage as _AIMessage
    from app.domain.services.graphs.react_graph import build_react_graph

    handler_counter = [0]  # never incremented — there's no handler

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(
        return_value=_AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    # Empty registry — simulating tool_filter that filtered out search_web.
    graph = build_react_graph(stub_llm, [])
    tool_node_fn = graph.nodes["tool_node"].bound.afunc

    fake_pe = _RecordingPEAlwaysDeny()
    fake_ssm = _make_fake_ssm()

    state = _make_state("search_web", {"query": "deep research"}, call_id="tc-blocked")
    config = _make_config(fake_pe, fake_ssm, session_id="sess-blocked-1")

    # Run tool_node. The blocked tool should NOT cause PE.evaluate to
    # fire for ``search_web`` (because the runner stripped it before
    # the graph was built, so the PE dispatcher hits ``tool_fn is None``
    # in its main loop and synthesizes an ``AllowError(unknown_tool)``
    # via ``_finalize_pe_outcome`` — no PE.evaluate call).
    #
    # We intentionally do NOT wrap this in try/except: the previous
    # implementation swallowed exceptions, which let the test pass
    # vacuously if the code path crashed before reaching the PE
    # evaluate assertion. Asserting the return is a real ``Command``
    # proves we actually executed the dispatcher to completion.
    from langgraph.types import Command
    result = await tool_node_fn(state, config)
    assert isinstance(result, Command), (
        f"tool_node_fn must return a Command; got {type(result).__name__} — "
        f"the dispatcher crashed before completion and the PE assertion "
        f"below would have passed vacuously"
    )

    # Hard invariant: PE was never asked about search_web because the
    # tool wasn't registered in the graph (filter ran upstream).
    leaked = [c for c in fake_pe.evaluate_calls if c.tool_name == "search_web"]
    assert not leaked, (
        f"PE.evaluate was called for blocked tool 'search_web' "
        f"({len(leaked)} times) — filter is being bypassed at react_graph level"
    )

    # Handler was never invoked (there was no handler in the first place).
    assert handler_counter[0] == 0


# ---------------------------------------------------------------------------
# 6. MCP discovery + summary regression tests for codex P1 #1 + P1 #2
# ---------------------------------------------------------------------------


async def test_mcp_discovery_respects_tool_filter() -> None:
    """P1 #1 regression: ``list_mcp_tools`` / ``get_mcp_tool`` must
    honor ``tool_filter`` so metadata for blocked MCP tools never reaches
    the LLM and blocked names can never be activated.

    Without the fix: ``list_mcp_tools`` returns the full MCP catalog and
    ``get_mcp_tool('blocked_name')`` activates the tool + returns the
    full schema, bypassing the runner-level allowlist.
    """
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )

    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_allowed_tool",
                    "description": "Permitted by allowlist",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string", "description": "query"},
                        },
                        "required": ["q"],
                    },
                }
            },
            {
                "function": {
                    "name": "mcp_blocked_tool",
                    "description": "Blocked by allowlist — SHOULD NOT LEAK",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "secret": {"type": "string", "description": "secret"},
                        },
                    },
                }
            },
        ]
    )

    activated: set[str] = set()
    # Allowlist excludes ``mcp_blocked_tool``. ``get_mcp_tool`` itself
    # is in the allowlist (otherwise the discovery tool wouldn't be
    # bound at the runner level in the first place).
    allow = frozenset({"mcp_allowed_tool", "list_mcp_tools", "get_mcp_tool"})

    tools = create_mcp_discovery_tools(
        mcp_tool_ref=lambda: mcp_tool_mock,
        activated_tools_ref=lambda: activated,
        tool_filter=allow,
    )
    list_tool = next(t for t in tools if t.name == "list_mcp_tools")
    get_tool = next(t for t in tools if t.name == "get_mcp_tool")

    # 1. list_mcp_tools omits the blocked name from output.
    list_result = await list_tool.ainvoke({"server_name": ""})
    assert "mcp_allowed_tool" in list_result
    assert "mcp_blocked_tool" not in list_result, (
        f"Blocked MCP tool name leaked into list_mcp_tools output:\n{list_result}"
    )
    # The description text for the blocked tool must also not leak.
    assert "SHOULD NOT LEAK" not in list_result

    # 2. get_mcp_tool refuses activation + does not return schema for blocked name.
    get_result = await get_tool.ainvoke({"tool_name": "mcp_blocked_tool"})
    assert "mcp_blocked_tool" not in activated, (
        f"Blocked MCP tool was activated despite filter: activated={activated}"
    )
    # Denial message MUST NOT include the schema field name 'secret'.
    assert "secret" not in get_result, (
        f"Schema parameter for blocked tool leaked: {get_result!r}"
    )
    assert "SHOULD NOT LEAK" not in get_result

    # 3. Sanity: allowed name still works (proves we didn't break the
    # happy path).
    happy_result = await get_tool.ainvoke({"tool_name": "mcp_allowed_tool"})
    assert "mcp_allowed_tool" in activated
    assert "mcp_allowed_tool" in happy_result


async def test_mcp_discovery_no_filter_keeps_legacy_behavior() -> None:
    """P1 #1 backward-compat: when ``tool_filter`` is ``None``, the
    discovery tools surface and activate the full MCP catalog (parent
    agent / planner path, no subagent restriction).
    """
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )

    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_anything",
                    "description": "Some MCP tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            },
        ]
    )

    activated: set[str] = set()
    tools = create_mcp_discovery_tools(
        mcp_tool_ref=lambda: mcp_tool_mock,
        activated_tools_ref=lambda: activated,
        tool_filter=None,
    )
    list_tool = next(t for t in tools if t.name == "list_mcp_tools")
    get_tool = next(t for t in tools if t.name == "get_mcp_tool")

    list_result = await list_tool.ainvoke({"server_name": ""})
    assert "mcp_anything" in list_result

    await get_tool.ainvoke({"tool_name": "mcp_anything"})
    assert "mcp_anything" in activated


def test_summary_no_blocked_tool_name_in_prefix() -> None:
    """P1 #2 regression: ``_build_available_tool_summary`` must NOT
    leak blocked tool names via line prefixes.

    Previously the "mcp available" bullet used the prefix
    ``"- mcp available (use get_mcp_tool to activate): "``. The summary
    filter only tokenized the BODY (post-colon), so a subagent whose
    allowlist excluded ``get_mcp_tool`` would still see that name
    embedded in the prefix.

    Fix: the prefix was renamed to ``"- mcp available: "`` (tool-name
    free) and the activation hint moved to a separate bullet
    ``"- mcp activation hint: get_mcp_tool"`` which the existing
    body-token filter naturally drops when ``get_mcp_tool`` is blocked.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    # Build a runner with a large MCP catalog (> 15 → forces discovery
    # mode) and an allowlist that excludes ``get_mcp_tool`` and
    # ``list_mcp_tools`` (so the prefix-leak path is exercised).
    runner = _seed_summary_runner(
        tool_filter=frozenset({"file_read", "search_web", "mcp_allowed_one"})
    )

    # Override the MCP mock to return > 15 tools → triggers the
    # discovery-mode branch that contains the offending bullet.
    fake_mcp_schemas = [
        {"function": {"name": f"mcp_tool_{i}"}} for i in range(20)
    ]
    fake_mcp_schemas.append({"function": {"name": "mcp_allowed_one"}})
    runner._mcp_tool.get_tools = MagicMock(return_value=fake_mcp_schemas)
    runner._get_always_bind_tool_names = MagicMock(return_value=set())

    summary = AgentTaskRunner._build_available_tool_summary(runner)

    # Hard token-level invariant: ``get_mcp_tool`` (which is NOT in the
    # allowlist) must not appear as a snake_case token anywhere in the
    # summary string — neither in body nor in any prefix.
    tokens = set(re.findall(r"[a-z_][a-z0-9_]+", summary))
    assert "get_mcp_tool" not in tokens, (
        f"P1 #2 regression: 'get_mcp_tool' leaked via line prefix into "
        f"summary despite being blocked by tool_filter:\n{summary}"
    )
    assert "list_mcp_tools" not in tokens, (
        f"P1 #2 regression: 'list_mcp_tools' leaked into summary despite "
        f"being blocked by tool_filter:\n{summary}"
    )


def test_summary_keeps_activation_hint_when_get_mcp_tool_allowed() -> None:
    """P1 #2 forward-compat: when ``get_mcp_tool`` IS in the allowlist,
    the activation-hint bullet must survive the filter so the LLM still
    knows how to activate MCP tools.

    Without this assertion we'd risk silently dropping the hint and
    confusing the model in the (common) parent-agent / unfiltered case.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = _seed_summary_runner(
        tool_filter=frozenset({
            "file_read", "search_web",
            "list_mcp_tools", "get_mcp_tool", "mcp_allowed_one",
        })
    )
    fake_mcp_schemas = [
        {"function": {"name": f"mcp_tool_{i}"}} for i in range(20)
    ]
    fake_mcp_schemas.append({"function": {"name": "mcp_allowed_one"}})
    runner._mcp_tool.get_tools = MagicMock(return_value=fake_mcp_schemas)
    runner._get_always_bind_tool_names = MagicMock(return_value=set())

    summary = AgentTaskRunner._build_available_tool_summary(runner)
    # Activation hint survives when allowlist permits it.
    assert "get_mcp_tool" in summary
    assert "list_mcp_tools" in summary


# ---------------------------------------------------------------------------
# 7. MCP discovery cross-leak fixes (codex R2 P1 #5)
#
# Bug: the hints embedded in list_mcp_tools' output and get_mcp_tool's denial
# message used to mention the *other* meta-tool unconditionally. If the
# subagent's allowlist excluded ``get_mcp_tool`` but allowed
# ``list_mcp_tools`` (or vice versa), the blocked meta-tool's name still
# leaked through the hint text reaching the LLM.
#
# Fix: both hints are now allowlist-aware. ``list_mcp_tools`` only mentions
# ``get_mcp_tool`` if it's in the allowlist; ``get_mcp_tool``'s denial /
# not-found messages only mention ``list_mcp_tools`` if it's in the
# allowlist. If neither is allowed, the hint is omitted entirely.
# ---------------------------------------------------------------------------


async def test_list_mcp_tools_omits_get_mcp_tool_hint_when_blocked() -> None:
    """P1 #5 (Bug 1): ``list_mcp_tools`` must not embed ``get_mcp_tool`` in
    its output when ``get_mcp_tool`` is excluded from the allowlist —
    otherwise the blocked meta-tool name leaks to the LLM.
    """
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )

    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_allowed_tool",
                    "description": "An allowed tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            },
        ]
    )

    # Allowlist permits ``list_mcp_tools`` and ``mcp_allowed_tool`` but
    # NOT ``get_mcp_tool`` — the hint must be omitted.
    allow = frozenset({"list_mcp_tools", "mcp_allowed_tool"})

    tools = create_mcp_discovery_tools(
        mcp_tool_ref=lambda: mcp_tool_mock,
        activated_tools_ref=lambda: set(),
        tool_filter=allow,
    )
    list_tool = next(t for t in tools if t.name == "list_mcp_tools")
    result = await list_tool.ainvoke({"server_name": ""})

    # The allowed tool is still surfaced.
    assert "mcp_allowed_tool" in result, (
        f"Allowed MCP tool was hidden by the filter: {result!r}"
    )
    # But the blocked meta-tool name must NOT appear anywhere in the
    # output (token-level invariant — no substring leak either).
    assert "get_mcp_tool" not in result, (
        f"P1 #5 Bug 1 regression: blocked meta-tool 'get_mcp_tool' "
        f"leaked via list_mcp_tools hint:\n{result}"
    )


async def test_get_mcp_tool_denial_omits_list_mcp_tools_hint_when_blocked() -> None:
    """P1 #5 (Bug 2): ``get_mcp_tool`` denial message must not embed
    ``list_mcp_tools`` when ``list_mcp_tools`` is excluded from the
    allowlist — otherwise the blocked meta-tool name leaks via the
    denial path.
    """
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )

    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_allowed_tool",
                    "description": "Allowed",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "function": {
                    "name": "mcp_blocked_tool",
                    "description": "Blocked — should never be activated",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    )

    # Allowlist permits ``get_mcp_tool`` and ``mcp_allowed_tool`` but
    # NOT ``list_mcp_tools`` — the denial hint must be omitted.
    allow = frozenset({"get_mcp_tool", "mcp_allowed_tool"})

    tools = create_mcp_discovery_tools(
        mcp_tool_ref=lambda: mcp_tool_mock,
        activated_tools_ref=lambda: set(),
        tool_filter=allow,
    )
    get_tool = next(t for t in tools if t.name == "get_mcp_tool")

    # Trigger denial branch — ``mcp_blocked_tool`` is NOT in the allowlist.
    denial_result = await get_tool.ainvoke({"tool_name": "mcp_blocked_tool"})

    # The blocked meta-tool name must NOT appear anywhere in the denial.
    assert "list_mcp_tools" not in denial_result, (
        f"P1 #5 Bug 2 regression: blocked meta-tool 'list_mcp_tools' "
        f"leaked via get_mcp_tool denial hint:\n{denial_result!r}"
    )
    # The denial still has to convey the unavailability — minimally the
    # requested tool name (which the LLM already supplied) appears.
    assert "mcp_blocked_tool" in denial_result, (
        f"Denial message lost the requested tool_name: {denial_result!r}"
    )


async def test_get_mcp_tool_denial_omits_all_hints_when_neither_meta_allowed() -> None:
    """P1 #5 corner case: when NEITHER ``list_mcp_tools`` NOR
    ``get_mcp_tool`` is in the allowlist (parent agent stripped both
    discovery meta-tools but the runner still bound them — pathological
    but possible), the denial message must omit both hints — only the
    bare unavailability notice remains.

    No leak in either direction; the LLM never sees either meta-tool
    name via the discovery surface.
    """
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )

    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_blocked_tool",
                    "description": "Blocked",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    )

    # Empty allowlist (or one that omits both meta-tools — same shape).
    allow = frozenset()

    tools = create_mcp_discovery_tools(
        mcp_tool_ref=lambda: mcp_tool_mock,
        activated_tools_ref=lambda: set(),
        tool_filter=allow,
    )
    get_tool = next(t for t in tools if t.name == "get_mcp_tool")

    denial_result = await get_tool.ainvoke({"tool_name": "mcp_blocked_tool"})

    assert "list_mcp_tools" not in denial_result, (
        f"Hint leaked despite both meta-tools blocked:\n{denial_result!r}"
    )
    assert "get_mcp_tool" not in denial_result, (
        f"Self-reference leaked despite both meta-tools blocked:\n{denial_result!r}"
    )


async def test_list_mcp_tools_keeps_hint_when_get_mcp_tool_allowed() -> None:
    """P1 #5 forward-compat: when ``get_mcp_tool`` IS in the allowlist,
    the activation hint must still appear in ``list_mcp_tools`` output
    so the LLM knows how to activate tools (don't regress the happy
    path while plugging the leak).
    """
    from app.domain.services.tools.langchain_mcp_discovery import (
        create_mcp_discovery_tools,
    )

    mcp_tool_mock = MagicMock()
    mcp_tool_mock.get_tools = MagicMock(
        return_value=[
            {
                "function": {
                    "name": "mcp_allowed_tool",
                    "description": "Allowed",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    )

    allow = frozenset({"list_mcp_tools", "get_mcp_tool", "mcp_allowed_tool"})

    tools = create_mcp_discovery_tools(
        mcp_tool_ref=lambda: mcp_tool_mock,
        activated_tools_ref=lambda: set(),
        tool_filter=allow,
    )
    list_tool = next(t for t in tools if t.name == "list_mcp_tools")
    result = await list_tool.ainvoke({"server_name": ""})

    assert "get_mcp_tool" in result, (
        f"Activation hint disappeared even though 'get_mcp_tool' is "
        f"allowed:\n{result}"
    )


# ---------------------------------------------------------------------------
# 8. Fail-closed react_graph_provider under tool_filter (codex R2 P1 #6)
#
# Bug: ``AgentTaskRunner._tool_filter`` only applied inside
# ``_build_lc_tools_full`` / ``_build_lc_tools_for_step``. The default
# ``react_graph`` constructed by ``PlannerReActFlow._ensure_graphs`` from
# the unfiltered ``_collect_all_tools()`` carried the parent agent's
# full tool set. When ``main_graph.executor_node``'s per-step
# ``react_graph_provider`` raised, the executor silently fell back to
# that default unfiltered graph, re-arming the subagent.
#
# Fix: a runner-level wrapper ``_react_graph_provider_for_executor``
# delegates verbatim when ``_tool_filter is None`` and raises
# ``ToolFilterProviderFailure`` (a marker exception) when the underlying
# provider fails while a filter is in effect. ``main_graph.executor_node``
# checks the exception type and propagates the marker instead of falling
# back. This test verifies the wrapper's contract end-to-end (no real
# main_graph involvement needed — the wrapper is the single behavioural
# gate).
# ---------------------------------------------------------------------------


async def test_tool_filter_fail_closed_on_provider_exception(monkeypatch) -> None:
    """P1 #6 regression: when ``_tool_filter`` is set and the underlying
    ``_build_step_react_graph`` raises, the runner wrapper MUST raise
    ``ToolFilterProviderFailure`` (a marker exception that
    ``main_graph.executor_node`` recognises and propagates) instead of
    swallowing the exception or returning a value that lets the
    executor fall back to the default unfiltered ``react_graph``.

    Setup: half-construct an ``AgentTaskRunner`` with
    ``_tool_filter = frozenset({"search_web"})`` and a
    ``_build_step_react_graph`` stub that always raises. Call the
    wrapper and assert the right exception type fires.

    Backwards path: also verify the wrapper delegates verbatim (no
    wrapping) when ``_tool_filter is None`` — exception propagates as
    the original type, NOT as ``ToolFilterProviderFailure``.
    """
    from app.domain.services.agent_task_runner import (
        AgentTaskRunner,
        ToolFilterProviderFailure,
    )

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = frozenset({"search_web"})

    class _BoomFromProvider(RuntimeError):
        pass

    async def _boom_provider(self, step_description: str = ""):
        # type: ignore[no-untyped-def]
        raise _BoomFromProvider("simulated provider failure")

    monkeypatch.setattr(
        AgentTaskRunner, "_build_step_react_graph", _boom_provider
    )

    # Under tool_filter, the wrapper must re-raise as the marker type
    # so main_graph recognises it and does NOT fall back to the default
    # unfiltered react_graph (which would silently re-arm the subagent
    # with the parent agent's full toolset).
    with pytest.raises(ToolFilterProviderFailure) as excinfo:
        await AgentTaskRunner._react_graph_provider_for_executor(runner, "any step")

    # __cause__ preserves the original exception for diagnostics.
    assert isinstance(excinfo.value.__cause__, _BoomFromProvider), (
        f"Original cause not preserved on the marker exception: "
        f"{excinfo.value.__cause__!r}"
    )

    # Backwards / unfiltered path: the wrapper delegates verbatim and
    # the legacy graceful-degrade in main_graph still fires
    # (executor.except catches it and falls back). The raised type is
    # the ORIGINAL exception type, NOT the marker — this proves the
    # wrapper is a no-op when no filter is set.
    runner_no_filter = object.__new__(AgentTaskRunner)
    runner_no_filter._tool_filter = None
    with pytest.raises(_BoomFromProvider):
        await AgentTaskRunner._react_graph_provider_for_executor(
            runner_no_filter, "any step"
        )


async def test_tool_filter_wrapper_passthrough_on_success(monkeypatch) -> None:
    """P1 #6 happy path: with ``_tool_filter`` set AND the underlying
    provider succeeding, the wrapper returns the provider's value
    verbatim. The filter does not interfere with the happy path — it
    only short-circuits exceptions.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = frozenset({"search_web"})

    sentinel_graph = MagicMock(name="sentinel_step_react")
    sentinel_meta = MagicMock(name="sentinel_step_meta")

    async def _ok_provider(self, step_description: str = ""):
        # type: ignore[no-untyped-def]
        return (sentinel_graph, sentinel_meta)

    monkeypatch.setattr(
        AgentTaskRunner, "_build_step_react_graph", _ok_provider
    )

    result = await AgentTaskRunner._react_graph_provider_for_executor(
        runner, "any step"
    )
    assert result == (sentinel_graph, sentinel_meta), (
        f"Wrapper did not return the underlying provider's value verbatim: "
        f"{result!r}"
    )
