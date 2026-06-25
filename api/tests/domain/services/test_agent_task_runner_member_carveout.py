"""Task 3.8 — child-side member-skill carve-out pure helpers.

[S4 §12] These exercise the four module-level pure helpers that implement the
team-member skill carve-out (pool override, selection floor, provider 4th-hop
derivation, built-vs-bound assertion). They are pure/sandbox-free by design so
the #1-risk override logic is unit-tested in isolation.
"""

import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    MCPConfig,
    MCPServerConfig,
)
from app.domain.services.agent_task_runner import (
    _apply_member_skill_floor,
    _assert_member_tools_built,
    _enabled_provider_names,
    _force_include_member_skills,
    _member_referenced_providers,
)
from app.domain.services.tools.mcp import _mcp_tool_namespace

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _skill(slug):
    # [codex-R1-F3] Skill requires source_type + source_ref (no defaults).
    return Skill(id=slug, slug=slug, name=slug, runtime_type=SkillRuntimeType.MCP,
                 enabled=True, source_type=SkillSourceType.LOCAL, source_ref=slug,
                 manifest={"tools": [{"name": "go"}]})


def test_force_include_adds_disabled_member_skill():
    enabled_all = [_skill("repo-map"), _skill("other")]
    filtered_pool = [_skill("other")]   # repo-map disabled in user prefs
    out = _force_include_member_skills(filtered_pool, enabled_all, member_slugs=("repo-map",))
    assert {s.slug for s in out} == {"other", "repo-map"}


def test_force_include_no_member_slugs_is_identity():
    pool = [_skill("other")]
    assert _force_include_member_skills(pool, [_skill("other")], member_slugs=()) is pool


def test_selection_floor_unions_member_skill_when_selector_dropped_it():
    # [codex-R8] the runtime silent-absence guard: a query-driven selection that
    # omits the member skill is floored back in.
    member = _skill("repo-map")
    pool = [_skill("other"), member]
    selected = [_skill("other")]                       # selector picked only 'other'
    out = _apply_member_skill_floor(selected, pool, member_slugs=("repo-map",))
    assert {s.slug for s in out} == {"other", "repo-map"}


def test_selection_floor_identity_no_member_slugs():
    selected = [_skill("other")]
    assert _apply_member_skill_floor(selected, [_skill("other")], ()) is selected


def test_selection_floor_no_duplicate_when_already_selected():
    member = _skill("repo-map")
    out = _apply_member_skill_floor([member], [member], ("repo-map",))
    assert [s.slug for s in out] == ["repo-map"]       # no duplicate


# ---- per-step REFRESH path floor (codex-R8: the hottest reselection site) -- #
#
# These drive the REAL ``AgentTaskRunner._compute_refreshed_skills`` (not just
# the pure helper) to prove the floor is wired into the per-step refresh — the
# branch that, without the floor, would drop a non-matching member skill and
# crash the per-step built-vs-bound assertion (fail-closed, but defeating the
# whole purpose of the carve-out on the path that runs every executor step).


class _DropMemberSelector:
    """Token-overlap selector stand-in that deliberately drops the member skill,
    simulating a query unrelated to the member skill's vocabulary."""

    def __init__(self, keep):
        self._keep = list(keep)

    def select(self, pool, query):  # signature mirrors the real skill selector
        return list(self._keep)


def _bare_refresh_runner(*, session_pool, member_slugs, selector_keep):
    """Half-construct an ``AgentTaskRunner`` seeded with exactly the surface that
    the token-overlap branch of ``_compute_refreshed_skills`` reads — no
    embedding index, no SkillTool, no context builder dependencies."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._embedding_available = False          # forces the token-overlap branch
    runner._embedding_index = None
    runner._current_message_text = ""
    runner._session_skill_pool = session_pool
    runner._member_skill_slugs = member_slugs
    runner._skill_selector = _DropMemberSelector(selector_keep)
    # Stub the heavy context builder so this stays a pure unit (avoids
    # _build_available_tool_summary's dependency on _skill_tool/_mcp_tool/etc.).
    runner._build_runtime_system_context = lambda skills, scores=None: ""
    return runner


async def test_refresh_path_floors_member_skill_when_selector_drops_it():
    """[codex-R8] The real per-step refresh re-includes a member skill that the
    selector dropped — proving the floor lives on the hottest reselection path,
    not only on the pure helper."""
    member = _skill("repo-map")
    other = _skill("other")
    runner = _bare_refresh_runner(
        session_pool=[other, member],
        member_slugs=(member.slug,),
        selector_keep=[other],                   # selector drops the member skill
    )

    result = await runner._compute_refreshed_skills("a query unrelated to repo-map")

    assert result is not None
    assert member.id in result.skill_ids, "floor must union the dropped member skill back"
    assert member.slug in {s.slug for s in result.skills}
    assert other.id in result.skill_ids        # the selected skill is still present


async def test_refresh_path_identity_when_no_member_slugs():
    """INV-0: with no team (empty member slugs) the per-step refresh returns the
    selector's choice verbatim — the member skill is NOT force-added."""
    member = _skill("repo-map")
    other = _skill("other")
    runner = _bare_refresh_runner(
        session_pool=[other, member],
        member_slugs=(),                         # no team
        selector_keep=[other],
    )

    result = await runner._compute_refreshed_skills("a query unrelated to repo-map")

    assert result is not None
    assert [s.slug for s in result.skills] == ["other"]   # floor identity (no add)
    assert result.skill_ids == (other.id,)


def test_assert_member_tools_built_raises_on_missing():
    import pytest
    with pytest.raises(Exception):
        _assert_member_tools_built(
            required={"skill_repo_map_go"}, built_names={"skill_other_go"},
        )


def test_assert_member_tools_built_ok_when_all_present():
    _assert_member_tools_built(required={"skill_a_x"}, built_names={"skill_a_x", "z"})  # no raise


# ---- provider 4th-hop derivation (R10-3 / codex-R1-F2) ------------------- #

def _mcp_skill(slug, tool_name):
    return Skill(id=slug, slug=slug, name=slug, runtime_type=SkillRuntimeType.MCP,
                 source_type=SkillSourceType.LOCAL, source_ref=slug,
                 manifest={"tools": [{"name": "go", "entry": {"tool_name": tool_name}}]})


def _a2a_skill(slug, agent_id):
    return Skill(id=slug, slug=slug, name=slug, runtime_type=SkillRuntimeType.A2A,
                 source_type=SkillSourceType.LOCAL, source_ref=slug,
                 manifest={"tools": [{"name": "ask", "entry": {"agent_id": agent_id}}]})


def test_referenced_mcp_server_derived_from_namespace_prefix():
    # MCP server identity is inferred by matching entry.tool_name against each
    # configured server's namespace prefix (skill.py:437-445 + mcp.py:411-420).
    sk = _mcp_skill("m", _mcp_tool_namespace("srv") + "_go")
    ref_mcp, ref_a2a, unconfigured = _member_referenced_providers([sk], {"srv"}, set())
    assert ref_mcp == {"srv"}
    assert not unconfigured


def test_a2a_agent_id_is_direct():
    sk = _a2a_skill("a", "agent-1")
    ref_mcp, ref_a2a, unconfigured = _member_referenced_providers([sk], set(), {"agent-1"})
    assert ref_a2a == {"agent-1"}
    assert not unconfigured


def test_unconfigured_provider_is_flagged_fail_closed():
    sk = _mcp_skill("m", "ghost_go")            # no configured server matches
    ref_mcp, _, unconfigured = _member_referenced_providers([sk], {"srv"}, set())
    assert not ref_mcp
    assert unconfigured  # caller raises → child fails closed


def test_referenced_mcp_falls_back_to_manifest_tool_name():
    # [codex-R2-F4] entry.tool_name absent → fall back to manifest tool name
    # (mirrors live _invoke_mcp skill.py:437-445), else a live-executable skill
    # is wrongly flagged unconfigured.
    ns = _mcp_tool_namespace("srv")
    sk = Skill(id="m", slug="m", name="m", runtime_type=SkillRuntimeType.MCP,
               source_type=SkillSourceType.LOCAL, source_ref="m",
               manifest={"tools": [{"name": f"{ns}_go", "entry": {}}]})
    ref_mcp, _, unconfigured = _member_referenced_providers([sk], {"srv"}, set())
    assert ref_mcp == {"srv"}
    assert not unconfigured


def test_referenced_mcp_longest_prefix_wins_deterministically():
    # [codex-R10-3 Fix B] Belt-and-suspenders determinism probe of the PURE helper
    # with a SYNTHETIC candidate set `{foo, foo_bar}`. In production the live MCP
    # namespace-collision guard (mcp.py:_validate_no_tool_namespace_collisions)
    # would REJECT `foo` + `foo_bar` as ENABLED servers — `mcp_foo_bar`.startswith
    # ("mcp_foo_") is a prefix overlap — so this pair can never both be enabled at
    # runtime. The helper must STILL be deterministic regardless of set iteration
    # order: a tool under the `foo_bar` namespace resolves to `foo_bar`, NOT the
    # shorter `foo` prefix (longest-prefix wins, not first-in-set).
    tool_name = _mcp_tool_namespace("foo_bar") + "_go"
    sk = _mcp_skill("m", tool_name)
    ref_mcp, _, unconfigured = _member_referenced_providers(
        [sk], {"foo", "foo_bar"}, set()
    )
    assert ref_mcp == {"foo_bar"}      # longest-prefix beats the shorter `foo`
    assert not unconfigured


# ---- _enabled_provider_names: configured == deployment-ENABLED only ------- #


def test_enabled_provider_names_excludes_deployment_disabled():
    # [§12/R10-3 Fix A] A deployment-disabled provider is NOT configured for the
    # carve-out: force-enabling the preference map can never override
    # server_config.enabled=False, so such a provider must be EXCLUDED from the
    # configured set → a member skill referencing it fails closed.
    mcp_config = MCPConfig(
        mcpServers={
            "on": MCPServerConfig(url="http://on", enabled=True),
            "off": MCPServerConfig(url="http://off", enabled=False),
        }
    )
    a2a_config = A2AConfig(
        a2a_servers=[
            A2AServerConfig(id="a-on", base_url="http://a-on", enabled=True),
            A2AServerConfig(id="a-off", base_url="http://a-off", enabled=False),
        ]
    )
    mcp_names, a2a_ids = _enabled_provider_names(mcp_config, a2a_config)
    assert mcp_names == {"on"}          # the deployment-disabled `off` is excluded
    assert a2a_ids == {"a-on"}          # the deployment-disabled `a-off` is excluded


def test_enabled_provider_names_none_configs_return_empty():
    mcp_names, a2a_ids = _enabled_provider_names(None, None)
    assert mcp_names == set()
    assert a2a_ids == set()


# ---- flag-OFF / non-coordinator built-vs-bound safety (INV-0) ------------- #


class _FakeNamedTool:
    """Minimal stand-in for a LangChain BaseTool — only ``.name`` is read by the
    per-step built-vs-bound assertion."""

    def __init__(self, name: str) -> None:
        self.name = name


def _build_lc_tools_runner_no_cpc(monkeypatch):
    """Half-construct an ``AgentTaskRunner`` seeded with exactly the surface the
    real ``_build_lc_tools_full`` reads, with NO
    ``_coordinator_child_permission_context`` attribute (flag-OFF / non-coordinator
    path). Mirrors test_agent_task_runner_tool_filter::
    test_tool_filter_blocks_tool_at_registry_construction."""
    from unittest.mock import MagicMock

    import app.domain.services.tools.langchain_a2a as la2a
    import app.domain.services.tools.langchain_dynamic_skill_tools as ldyn
    import app.domain.services.tools.langchain_mcp as lmcp
    import app.domain.services.tools.langchain_skill_tools as lskill
    import app.domain.services.tools.langchain_tools as ltools
    from app.domain.services.agent_task_runner import AgentTaskRunner

    fake_native = [_FakeNamedTool("shell_execute"), _FakeNamedTool("file_read")]
    monkeypatch.setattr(ltools, "create_native_tools", lambda **kw: fake_native)
    monkeypatch.setattr(lmcp, "create_mcp_langchain_tools", lambda *a, **kw: [])
    monkeypatch.setattr(la2a, "create_a2a_langchain_tools", lambda *a, **kw: [])
    monkeypatch.setattr(lskill, "create_skill_langchain_tools", lambda *a, **kw: [])
    monkeypatch.setattr(
        lskill, "create_skill_guide_tool",
        lambda *a, **kw: _FakeNamedTool("get_skill_guide"),
    )
    monkeypatch.setattr(
        ldyn, "create_dynamic_skill_langchain_tools", lambda *a, **kw: [],
    )

    runner = object.__new__(AgentTaskRunner)
    runner._tool_filter = None
    runner._execution_supervisor = None
    runner._sandbox = MagicMock()
    runner._browser = MagicMock()
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
    runner._skill_tool = MagicMock()
    runner._session_skill_pool = []
    runner._skill_bundle_sync = MagicMock()
    runner._skill_bundle_sync.get_file_listing_all = MagicMock(return_value={})
    runner._skill_bundle_sync.sandbox_skill_root = "/tmp/skills"
    runner._memory_session_factory = None
    runner._memory_repo_factory = None
    runner._user_id = "u1"
    runner._session_id = "s1"
    return AgentTaskRunner, runner


def test_build_lc_tools_no_cpc_does_not_raise(monkeypatch):
    """INV-0 / flag-OFF: a runner with NO _coordinator_child_permission_context
    builds lc_tools WITHOUT firing the built-vs-bound assertion (the
    non-coordinator path must never NameError or raise)."""
    AgentTaskRunner, runner = _build_lc_tools_runner_no_cpc(monkeypatch)
    # No _coordinator_child_permission_context attribute at all.
    assert not hasattr(runner, "_coordinator_child_permission_context")
    result = AgentTaskRunner._build_lc_tools_full(runner)
    assert {t.name for t in result} == {"shell_execute", "file_read", "get_skill_guide"}


def test_build_lc_tools_empty_member_tools_does_not_raise(monkeypatch):
    """INV-0: a coordinator child whose cpc carries an EMPTY member_skill_tools
    set still builds lc_tools without the assertion firing."""
    from app.domain.services.permission.child_permission_context import (
        ChildPermissionContext,
    )

    AgentTaskRunner, runner = _build_lc_tools_runner_no_cpc(monkeypatch)
    runner._coordinator_child_permission_context = object.__new__(
        ChildPermissionContext
    )
    # frozenset() default ⇒ assertion short-circuits.
    object.__setattr__(
        runner._coordinator_child_permission_context,
        "member_skill_tools",
        frozenset(),
    )
    result = AgentTaskRunner._build_lc_tools_full(runner)
    assert {t.name for t in result} == {"shell_execute", "file_read", "get_skill_guide"}


def test_build_lc_tools_missing_member_tool_raises(monkeypatch):
    """The loud assertion: a cpc demanding a member tool that is NOT among the
    built tools raises a clear terminal error (NOT silent absence)."""
    import pytest

    from app.domain.services.permission.child_permission_context import (
        ChildPermissionContext,
    )

    AgentTaskRunner, runner = _build_lc_tools_runner_no_cpc(monkeypatch)
    runner._coordinator_child_permission_context = object.__new__(
        ChildPermissionContext
    )
    object.__setattr__(
        runner._coordinator_child_permission_context,
        "member_skill_tools",
        frozenset({"skill_repo_map_go"}),  # never built ⇒ must raise
    )
    with pytest.raises(RuntimeError, match="were not built"):
        AgentTaskRunner._build_lc_tools_full(runner)
