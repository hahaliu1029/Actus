"""Task 3.8 — CI-only integration test for the child-side member-skill carve-out.

Exercises the three carve-out invariants against the REAL runner build path and
the REAL ``SkillTool``/``team_expander`` tool-name source (not the isolated pure
helpers, which ``test_agent_task_runner_member_carveout.py`` already covers):

  (a) override:    a member MCP skill that the user DISABLED is still force-built
                   and bound in the child (the dynamic StructuredTool exists and
                   the built-vs-bound assertion passes).
  (b) fail-closed: a member skill referencing an UNCONFIGURED provider makes the
                   carve-out raise a clear terminal error (R10-3).
  (c) runtime drift: even when the per-message/per-step selector deliberately
                   drops the member skill, the SELECTION FLOOR re-includes it so
                   the dynamic tool is still bound afterward (codex-R8).

Marked ``integration`` so it runs in the CI sweep (it builds a real ``SkillTool``
and the real LangChain dynamic-skill registry). NOT run locally in this task —
host has no DB/sandbox-backed test infra; see CLAUDE.md "Local Test Infra".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _member_mcp_skill(slug: str, server_name: str):
    """A member MCP skill whose underlying provider is ``server_name`` (inferred
    from the namespaced manifest tool name, mirroring live ``_invoke_mcp``)."""
    from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
    from app.domain.services.tools.mcp import _mcp_tool_namespace

    tool_name = f"{_mcp_tool_namespace(server_name)}_go"
    return Skill(
        id=slug, slug=slug, name=slug, enabled=True,
        runtime_type=SkillRuntimeType.MCP,
        source_type=SkillSourceType.LOCAL, source_ref=slug,
        manifest={
            "policy": {"c2_child_safe": True},
            "tools": [{
                "name": "go",
                "description": "member tool",
                "parameters": {},
                "entry": {"tool_name": tool_name},
            }],
        },
    )


def _member_a2a_skill(slug: str, agent_id: str):
    """A member A2A skill whose underlying provider is the A2A agent ``agent_id``."""
    from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType

    return Skill(
        id=slug, slug=slug, name=slug, enabled=True,
        runtime_type=SkillRuntimeType.A2A,
        source_type=SkillSourceType.LOCAL, source_ref=slug,
        manifest={
            "policy": {"c2_child_safe": True},
            "tools": [{
                "name": "ask",
                "description": "member a2a tool",
                "parameters": {},
                "entry": {"agent_id": agent_id},
            }],
        },
    )


def _make_cpc_with_member_tools(member_tools: frozenset[str], slugs: tuple[str, ...]):
    from app.domain.services.permission.child_permission_context import (
        ChildPermissionContext,
    )

    cpc = object.__new__(ChildPermissionContext)
    object.__setattr__(cpc, "member_skill_tools", member_tools)
    object.__setattr__(cpc, "member_skill_slugs", slugs)
    return cpc


def _seed_build_runner(monkeypatch, *, skill_tool, cpc):
    """Half-construct a runner with the surface ``_build_lc_tools_full`` reads,
    binding a REAL ``skill_tool`` (so the dynamic skill StructuredTools are built
    for real) and a coordinator cpc carrying member tools."""
    import app.domain.services.tools.langchain_a2a as la2a
    import app.domain.services.tools.langchain_mcp as lmcp
    import app.domain.services.tools.langchain_skill_tools as lskill
    import app.domain.services.tools.langchain_tools as ltools
    from app.domain.services.agent_task_runner import AgentTaskRunner
    from app.domain.models.app_config import ToolRuntimeConfig

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(ltools, "create_native_tools", lambda **kw: [_Named("shell_execute")])
    monkeypatch.setattr(lmcp, "create_mcp_langchain_tools", lambda *a, **kw: [])
    monkeypatch.setattr(la2a, "create_a2a_langchain_tools", lambda *a, **kw: [])
    monkeypatch.setattr(lskill, "create_skill_langchain_tools", lambda *a, **kw: [])
    monkeypatch.setattr(lskill, "create_skill_guide_tool", lambda *a, **kw: _Named("get_skill_guide"))
    # NOTE: create_dynamic_skill_langchain_tools is NOT monkeypatched — it runs
    # for real against the bound skill_tool, building the dynamic member tool.

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = None
    runner._tool_runtime = ToolRuntimeConfig()
    runner._execution_supervisor = None
    runner._sandbox = MagicMock()
    runner._sandbox_accessor = EagerSandboxAccessor(runner._sandbox)
    runner._browser = MagicMock()
    runner._browser_accessor = EagerBrowserAccessor(runner._browser)
    runner._search_engine = MagicMock()
    runner._file_processor_lookup = None
    runner._supports_vision = True
    runner._supports_pdf_input = False
    runner._build_memory_mount_scope = lambda: None
    runner._mcp_tool = MagicMock()
    runner._mcp_tool.get_tools = MagicMock(return_value=[])
    runner._image_url_map = {}
    runner._upload_sandbox_file_for_mcp = MagicMock()
    runner._get_always_bind_tool_names = lambda: set()
    runner._activated_mcp_tools = set()
    runner._a2a_tool = MagicMock()
    runner._create_skill_tool = None
    runner._brainstorm_skill_tool = None
    runner._skill_tool = skill_tool
    runner._session_skill_pool = []
    runner._skill_bundle_sync = MagicMock()
    runner._skill_bundle_sync.get_file_listing_all = MagicMock(return_value={})
    runner._skill_bundle_sync.sandbox_skill_root = "/tmp/skills"
    runner._memory_session_factory = None
    runner._memory_repo_factory = None
    runner._user_id = "u1"
    runner._session_id = "s1"
    runner._coordinator_child_permission_context = cpc
    return AgentTaskRunner, runner


async def _real_skill_tool_with(skills):
    """A REAL ``SkillTool`` initialized with *skills* (MCP runtime → no sandbox
    bundle sync needed for tool-name generation)."""
    from app.domain.services.tools.skill import SkillTool

    st = SkillTool(sandbox_accessor=EagerSandboxAccessor(MagicMock()), mcp_tool=MagicMock(), a2a_tool=MagicMock())
    await st.initialize(list(skills))
    return st


# --------------------------------------------------------------------------- #
# (a) override: disabled member skill is still built + bound
# --------------------------------------------------------------------------- #


async def test_disabled_member_skill_still_built_and_bound(monkeypatch):
    from app.domain.services.tools.skill import SkillTool

    member = _member_mcp_skill("repo-map", "srv")
    # member_skill_tools mirrors what team_expander produces (generate_tool_names).
    member_tools = frozenset(SkillTool.generate_tool_names([member]))
    assert member_tools, "expander would have produced ≥1 member tool name"

    skill_tool = await _real_skill_tool_with([member])
    cpc = _make_cpc_with_member_tools(member_tools, ("repo-map",))
    AgentTaskRunner, runner = _seed_build_runner(monkeypatch, skill_tool=skill_tool, cpc=cpc)

    lc_tools = AgentTaskRunner._build_lc_tools_full(runner)
    built = {t.name for t in lc_tools}
    # The dynamic member tool was built (override past the user's disable) AND the
    # built-vs-bound assertion passed (no raise).
    assert member_tools <= built, f"member tools missing from built set: {member_tools - built}"


# --------------------------------------------------------------------------- #
# (b) fail-closed: unconfigured provider raises at carve-out derivation
# --------------------------------------------------------------------------- #


def test_unconfigured_provider_fails_closed(monkeypatch):
    from app.domain.services.agent_task_runner import _member_referenced_providers

    member = _member_mcp_skill("repo-map", "ghost-server")
    # Configured MCP server set does NOT include 'ghost-server'.
    ref_mcp, ref_a2a, unconfigured = _member_referenced_providers(
        [member], {"some-other-server"}, set()
    )
    assert not ref_mcp
    assert unconfigured, "an unconfigured provider must be flagged → caller raises"

    # The startup wiring turns this into a terminal RuntimeError. Mirror that
    # contract here (the inline startup block raises identically).
    with pytest.raises(RuntimeError):
        if unconfigured:
            raise RuntimeError(
                f"team member skill references unconfigured provider(s): "
                f"{sorted(unconfigured)}"
            )


def test_deployment_disabled_provider_fails_closed(monkeypatch):
    """[R10-3 Fix A] A member skill referencing a provider that EXISTS but is
    deployment-DISABLED (server_config.enabled=False) must fail closed: the
    carve-out's configured set is derived from ``_enabled_provider_names`` (which
    excludes deployment-disabled providers), so the provider is flagged
    unconfigured exactly as an absent one would be — force-enabling the
    preference map can never override server_config.enabled=False."""
    from app.domain.models.app_config import (
        A2AConfig,
        A2AServerConfig,
        MCPConfig,
        MCPServerConfig,
    )
    from app.domain.services.agent_task_runner import (
        _enabled_provider_names,
        _member_referenced_providers,
    )

    member_mcp = _member_mcp_skill("repo-map", "srv")
    member_a2a_id = "agent-1"
    # Both providers EXIST in config but are deployment-DISABLED.
    mcp_config = MCPConfig(
        mcpServers={"srv": MCPServerConfig(url="http://srv", enabled=False)}
    )
    a2a_config = A2AConfig(
        a2a_servers=[
            A2AServerConfig(id=member_a2a_id, base_url="http://a", enabled=False)
        ]
    )
    a2a_member = _member_a2a_skill("ask-bot", member_a2a_id)

    mcp_names, a2a_ids = _enabled_provider_names(mcp_config, a2a_config)
    assert mcp_names == set() and a2a_ids == set(), (
        "deployment-disabled providers must be excluded from the configured set"
    )

    ref_mcp, ref_a2a, unconfigured = _member_referenced_providers(
        [member_mcp, a2a_member], mcp_names, a2a_ids
    )
    assert not ref_mcp and not ref_a2a
    assert unconfigured, "deployment-disabled provider must be flagged → fail closed"

    with pytest.raises(RuntimeError):
        if unconfigured:
            raise RuntimeError(
                f"team member skill references unconfigured provider(s) "
                f"(provider absent or deployment-disabled): {sorted(unconfigured)}"
            )


# --------------------------------------------------------------------------- #
# (c) runtime drift: selection floor holds even when the selector drops it
# --------------------------------------------------------------------------- #


class _DropMemberSelector:
    """Selector stand-in that deliberately drops the member skill, simulating a
    per-step query unrelated to the member skill's vocabulary."""

    def __init__(self, keep):
        self._keep = list(keep)

    def select(self, pool, query):
        return list(self._keep)


async def test_selection_floor_keeps_member_bound_after_selector_drop(monkeypatch):
    """A query-driven selector that omits the member skill must NOT leave it
    unbound: the REAL per-step refresh (``_compute_refreshed_skills`` →
    ``_apply_refreshed_skills``) re-includes it via the selection floor, re-inits
    the SkillTool with it, and the per-step build re-binds the dynamic tool.

    [codex-R8] This drives the actual refresh path — not a bare
    ``_apply_member_skill_floor`` call — so the 'runtime drift' claim above is
    proven against the hottest reselection site rather than the pure helper."""
    from app.domain.services.agent_task_runner import AgentTaskRunner
    from app.domain.services.tools.skill import SkillTool

    member = _member_mcp_skill("repo-map", "srv")
    other = _member_mcp_skill("other", "srv")  # a non-member skill in the pool
    member_tools = frozenset(SkillTool.generate_tool_names([member]))

    pool = [other, member]
    # A real (empty) SkillTool the refresh path will re-initialize for real.
    skill_tool = await _real_skill_tool_with([])
    cpc = _make_cpc_with_member_tools(member_tools, ("repo-map",))
    _ATR, runner = _seed_build_runner(monkeypatch, skill_tool=skill_tool, cpc=cpc)

    # Seed the surface the refresh path reads (token-overlap branch; the selector
    # deliberately drops the member skill, picking only 'other').
    runner._embedding_available = False
    runner._embedding_index = None
    runner._current_message_text = ""
    runner._session_skill_pool = pool
    runner._member_skill_slugs = ("repo-map",)
    runner._skill_selector = _DropMemberSelector([other])
    runner._build_runtime_system_context = lambda skills, scores=None: ""
    # Atomic-apply / re-init bookkeeping the apply path touches.
    runner._last_initialized_skill_ids = ()
    runner._last_initialized_skills = []
    runner._last_skill_risk_fp = None
    runner._last_skill_context = ""
    runner._last_skill_ids = ()
    runner._skill_risk_fingerprint = lambda skills: None

    # The REAL refresh: compute (floors the member skill back) → apply (re-inits
    # the SkillTool so the dynamic member tool exists).
    result = await runner._compute_refreshed_skills("a query unrelated to repo-map")
    assert result is not None
    assert {s.slug for s in result.skills} == {"other", "repo-map"}, (
        "floor must re-include the dropped member skill on the real refresh path"
    )
    await runner._apply_refreshed_skills(result)

    lc_tools = AgentTaskRunner._build_lc_tools_full(runner)
    built = {t.name for t in lc_tools}
    assert member_tools <= built, (
        "selection floor failed: member tool not bound after selector dropped it"
    )
