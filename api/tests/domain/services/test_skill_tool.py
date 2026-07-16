from __future__ import annotations

import asyncio
import copy

import pytest

from app.application.services.sandbox_accessors import EagerSandboxAccessor
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.tool_result import AllowError, AllowSuccess, Asked, ToolResult
from app.domain.services.tools.skill import SkillTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.available_dirs = {
            "/home/ubuntu/workspace",
            "/home/ubuntu/workspace/.skills/pptx--1234abcd",
            "/tmp/custom-skill-dir",
        }

    async def exec_command(self, session_id: str, exec_dir: str, command: str) -> ToolResult:
        self.calls.append((session_id, exec_dir, command))
        return ToolResult(success=True, message="ok", data={"session_id": session_id})

    async def read_shell_output(self, session_id: str, console: bool = False) -> ToolResult:
        return ToolResult(success=True, data={"session_id": session_id, "output": "native-ok"})

    async def check_file_exists(self, filepath: str) -> ToolResult:
        return ToolResult(success=True, data={"exists": filepath in self.available_dirs})


class _FakeMCPTool:
    def __init__(self) -> None:
        self.called: tuple[str, dict] | None = None

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        self.called = (tool_name, kwargs)
        return ToolResult(success=True, data="mcp-ok")


class _FakeA2ATool:
    def __init__(self) -> None:
        self.called: tuple[str, str] | None = None

    async def call_remote_agent(self, id: str, query: str) -> ToolResult:
        self.called = (id, query)
        return ToolResult(success=True, data="a2a-ok")


class _FakeBundleSyncManager:
    def __init__(
        self,
        *,
        ready_dir: str | None = None,
        error: str | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.ready_dir = ready_dir
        self.error = error
        self.gate = gate
        self.calls: list[str] = []

    async def ensure_ready_for_invoke(
        self,
        skill_id: str,
        *,
        skill: Skill | None = None,
    ) -> tuple[str | None, str | None]:
        self.calls.append(skill_id)
        if self.gate is not None:
            await self.gate.wait()
        return self.ready_dir, self.error



def _build_native_skill(
    *,
    skill_id: str = "demo-native--1234abcd",
    slug: str = "demo-native",
    entry: dict | None = None,
) -> Skill:
    return Skill(
        id=skill_id,
        slug=slug,
        name="Demo Native",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/repo",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": "Demo Native",
            "runtime_type": "native",
            "skill_md": "# Demo Native\nUse this skill when user asks for demo shell action.",
            "bundle_file_count": 1,
            "last_sync_at": "v1",
            "tools": [
                {
                    "name": "run_demo",
                    "description": "run",
                    "parameters": {"target": {"type": "string"}},
                    "required": ["target"],
                    "entry": entry
                    if entry is not None
                    else {
                        "exec_dir": "/home/ubuntu/workspace",
                        "command": "echo demo",
                    },
                }
            ],
        },
        installed_by="admin-1",
    )


def _build_skill_with_tools(
    *,
    slug: str,
    tool_names: list[str],
    skill_id: str | None = None,
) -> Skill:
    """Factory for tests that need skills with a custom set of tool names.

    Unlike ``_build_native_skill`` (hardcoded single "run_demo" tool), this
    builder constructs a manifest with one entry per name in ``tool_names``,
    each with a minimal parameters schema. Used by the atomicity tests to
    verify behavior across multi-tool / multi-skill scenarios.
    """
    return Skill(
        id=skill_id or f"{slug}--test",
        slug=slug,
        name=f"Demo {slug}",
        source_type=SkillSourceType.GITHUB,
        source_ref=f"github:test/{slug}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": f"Demo {slug}",
            "runtime_type": "native",
            "skill_md": f"# {slug}\nTest skill.",
            "bundle_file_count": 0,
            "last_sync_at": "v1",
            "tools": [
                {
                    "name": name,
                    "description": f"Tool {name}",
                    "parameters": {"query": {"type": "string"}},
                    "required": ["query"],
                    "entry": {
                        "exec_dir": "/home/ubuntu/workspace",
                        "command": f"echo {name}",
                    },
                }
                for name in tool_names
            ],
        },
        installed_by="admin-1",
    )


def _snapshot_skill_tool(st: SkillTool) -> dict:
    """Capture SkillTool's 5 mutable state fields for atomicity assertions.

    Used by the atomicity tests to verify that a failing initialize() call
    leaves the internal state exactly equal to its pre-call value. Uses
    ``copy.deepcopy`` on nested structures so that post-call mutation of
    the live state cannot retroactively taint the snapshot.
    """
    return {
        "skills": list(st._skills),
        "tools": copy.deepcopy(st._tools),
        "bindings": copy.deepcopy(st._tool_bindings),
        "name_index": dict(st._tool_name_index),
        "tools_cache": list(st._tools_cache) if st._tools_cache is not None else None,
    }


def _assert_skill_tool_matches(st: SkillTool, snap: dict) -> None:
    """Assert SkillTool's 5 mutable fields equal a prior snapshot."""
    assert [s.id for s in st._skills] == [s.id for s in snap["skills"]]
    assert st._tools == snap["tools"]
    assert st._tool_bindings == snap["bindings"]
    assert st._tool_name_index == snap["name_index"]
    assert st._tools_cache == snap["tools_cache"]


def _make_bare_skill_tool() -> SkillTool:
    """Construct a minimal SkillTool for atomicity tests.

    Returns a SkillTool with fresh fake sandbox/mcp/a2a tools. Does NOT call
    initialize() — callers control initialization order to exercise atomicity.
    """
    return SkillTool(
        sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
    )


async def test_native_skill_executes_entry_command() -> None:
    sandbox = _FakeSandbox()
    skill_tool = SkillTool(sandbox_accessor=EagerSandboxAccessor(sandbox), mcp_tool=_FakeMCPTool(), a2a_tool=_FakeA2ATool())

    skill = _build_native_skill()

    await skill_tool.initialize([skill])
    tools = skill_tool.get_tools()
    assert tools[0]["function"]["name"] == "skill_demo_native_run_demo"
    assert "Skill guide:" in tools[0]["function"]["description"]

    result = await skill_tool.invoke("skill_demo_native_run_demo", target="hello")

    assert isinstance(result, AllowSuccess)
    assert sandbox.calls
    _, _, command = sandbox.calls[0]
    assert "echo demo" in command


async def test_native_defaults_exec_dir_to_skill_directory_when_missing() -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill(
        skill_id="pptx--1234abcd",
        slug="pptx",
        entry={"command": "python scripts/run.py"},
    )
    skill.manifest["bundle_file_count"] = 0

    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
    )

    await skill_tool.initialize([skill])
    function_name = skill_tool.get_tools()[0]["function"]["name"]
    result = await skill_tool.invoke(function_name, topic="deck")

    assert isinstance(result, AllowSuccess)
    assert sandbox.calls
    _, exec_dir, _ = sandbox.calls[0]
    assert exec_dir == "/home/ubuntu/workspace/.skills/pptx--1234abcd"


async def test_native_waits_for_sync_before_execute() -> None:
    sandbox = _FakeSandbox()
    gate = asyncio.Event()
    sync_manager = _FakeBundleSyncManager(
        ready_dir="/home/ubuntu/workspace/.skills/pptx--1234abcd",
        gate=gate,
    )
    skill = _build_native_skill(
        skill_id="pptx--1234abcd",
        slug="pptx",
        entry={"command": "python scripts/run.py"},
    )

    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
        bundle_sync_manager=sync_manager,
    )

    await skill_tool.initialize([skill])
    function_name = skill_tool.get_tools()[0]["function"]["name"]

    invoke_task = asyncio.create_task(skill_tool.invoke(function_name, topic="deck"))
    await asyncio.sleep(0.01)
    assert not sandbox.calls

    gate.set()
    result = await invoke_task

    assert isinstance(result, AllowSuccess)
    assert sync_manager.calls == ["pptx--1234abcd"]
    assert sandbox.calls


async def test_native_returns_error_when_sync_failed() -> None:
    sandbox = _FakeSandbox()
    sync_manager = _FakeBundleSyncManager(error="Skill[pptx--1234abcd] bundle同步失败")
    skill = _build_native_skill(
        skill_id="pptx--1234abcd",
        slug="pptx",
        entry={"command": "python scripts/run.py"},
    )

    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
        bundle_sync_manager=sync_manager,
    )

    await skill_tool.initialize([skill])
    function_name = skill_tool.get_tools()[0]["function"]["name"]
    result = await skill_tool.invoke(function_name, topic="deck")

    assert isinstance(result, AllowError)
    assert "同步失败" in result.content
    assert not sandbox.calls


async def test_native_explicit_exec_dir_not_overridden() -> None:
    sandbox = _FakeSandbox()
    sync_manager = _FakeBundleSyncManager(
        ready_dir="/home/ubuntu/workspace/.skills/pptx--1234abcd",
    )
    skill = _build_native_skill(
        skill_id="pptx--1234abcd",
        slug="pptx",
        entry={"command": "python scripts/run.py", "exec_dir": "/tmp/custom-skill-dir"},
    )

    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
        bundle_sync_manager=sync_manager,
    )

    await skill_tool.initialize([skill])
    function_name = skill_tool.get_tools()[0]["function"]["name"]
    result = await skill_tool.invoke(function_name, topic="deck")

    assert isinstance(result, AllowSuccess)
    assert sandbox.calls
    _, exec_dir, _ = sandbox.calls[0]
    assert exec_dir == "/tmp/custom-skill-dir"


async def test_mcp_skill_delegates_to_mcp_tool() -> None:
    mcp_tool = _FakeMCPTool()
    skill_tool = SkillTool(sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()), mcp_tool=mcp_tool, a2a_tool=_FakeA2ATool())

    skill = Skill(
        slug="demo-mcp",
        name="Demo MCP",
        source_type=SkillSourceType.LOCAL,
        source_ref="mcp:demo",
        runtime_type=SkillRuntimeType.MCP,
        manifest={
            "name": "Demo MCP",
            "runtime_type": "mcp",
            "tools": [
                {
                    "name": "route",
                    "description": "route",
                    "parameters": {"query": {"type": "string"}},
                    "required": ["query"],
                    "entry": {
                        "tool_name": "mcp_demo_route",
                    },
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_mcp_route", query="q")

    assert isinstance(result, AllowSuccess)
    assert mcp_tool.called == ("mcp_demo_route", {"query": "q"})


async def test_a2a_skill_delegates_to_a2a_tool() -> None:
    a2a_tool = _FakeA2ATool()
    skill_tool = SkillTool(sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()), mcp_tool=_FakeMCPTool(), a2a_tool=a2a_tool)

    skill = Skill(
        slug="demo-a2a",
        name="Demo A2A",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/a2a",
        runtime_type=SkillRuntimeType.A2A,
        manifest={
            "name": "Demo A2A",
            "runtime_type": "a2a",
            "tools": [
                {
                    "name": "delegate",
                    "description": "delegate",
                    "parameters": {"query": {"type": "string"}},
                    "required": ["query"],
                    "entry": {
                        "agent_id": "agent-1",
                    },
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_a2a_delegate", query="hello")

    assert isinstance(result, AllowSuccess)
    assert a2a_tool.called == ("agent-1", "hello")


async def test_skill_invoke_high_risk_manifest_executes_directly() -> None:
    # R3: _evaluate_risk_enforce has been removed from SkillTool.invoke().
    # Risk enforcement is now handled by the Stage P branch in react_graph.tool_node.
    # SkillTool.invoke() always executes the tool regardless of risk_level.
    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
        risk_mode="enforce_confirmation",
    )
    skill = _build_native_skill()
    skill.manifest["tools"][0]["policy"] = {"risk_level": "high"}

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="hello")

    # invoke() now falls through to execution; Asked is no longer returned here
    assert isinstance(result, AllowSuccess)


async def test_skill_risk_enforce_off_returns_success() -> None:
    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
        risk_mode="off",
    )
    skill = _build_native_skill()

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="hello")

    assert isinstance(result, AllowSuccess)


async def test_skill_risk_enforce_low_risk_returns_success() -> None:
    skill_tool = SkillTool(
        sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()),
        mcp_tool=_FakeMCPTool(),
        a2a_tool=_FakeA2ATool(),
        risk_mode="enforce_confirmation",
    )
    skill = _build_native_skill()
    skill.manifest["tools"][0]["policy"] = {"risk_level": "low"}

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="hello")

    assert isinstance(result, AllowSuccess)


async def test_skill_tool_normalizes_and_shortens_function_name() -> None:
    skill_tool = SkillTool(sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()), mcp_tool=_FakeMCPTool(), a2a_tool=_FakeA2ATool())

    skill = Skill(
        slug="pptx-skill",
        name="PPTX",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/pptx",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": "PPTX",
            "runtime_type": "native",
            "tools": [
                {
                    "name": "Render Slide/Deck (Very Long Name For External Skill Ecosystem)",
                    "description": "render",
                    "parameters": {},
                    "required": [],
                    "entry": {"exec_dir": "/home/ubuntu/workspace", "command": "echo pptx"},
                }
            ],
        },
        installed_by="admin-1",
    )

    await skill_tool.initialize([skill])
    function_name = skill_tool.get_tools()[0]["function"]["name"]
    assert len(function_name) <= 64
    assert "/" not in function_name
    assert "-" not in function_name


# ----- #27: SkillTool.initialize() atomicity — success-path (I3) ----- #


async def test_initialize_atomicity_replaces_empty_state() -> None:
    """I3: initialize on empty SkillTool populates all 5 fields."""
    skill_tool = _make_bare_skill_tool()
    s1 = _build_skill_with_tools(slug="sa", tool_names=["t1", "t2"])

    await skill_tool.initialize([s1])

    assert len(skill_tool._tools) == 2
    assert "skill_sa_t1" in skill_tool._tool_bindings
    assert "skill_sa_t2" in skill_tool._tool_bindings
    assert skill_tool._tool_name_index["skill_sa_t1"] == 1
    assert skill_tool._tool_name_index["skill_sa_t2"] == 1
    assert skill_tool._tools_cache == skill_tool._tools


async def test_initialize_atomicity_replaces_populated_state() -> None:
    """I3: initialize with new skills replaces old state wholesale."""
    skill_tool = _make_bare_skill_tool()
    s1 = _build_skill_with_tools(slug="sa", tool_names=["t1"])
    await skill_tool.initialize([s1])

    s2 = _build_skill_with_tools(slug="sb", tool_names=["t2"])
    await skill_tool.initialize([s2])

    # sa binding is gone
    assert "skill_sa_t1" not in skill_tool._tool_bindings
    # sb binding is present
    assert "skill_sb_t2" in skill_tool._tool_bindings
    # name index does not carry sa's counter
    assert "skill_sa_t1" not in skill_tool._tool_name_index
    assert skill_tool._tool_name_index["skill_sb_t2"] == 1


async def test_initialize_atomicity_to_empty_clears_state() -> None:
    """I3: initialize([]) clears all 5 fields."""
    skill_tool = _make_bare_skill_tool()
    s1 = _build_skill_with_tools(slug="sa", tool_names=["t1"])
    await skill_tool.initialize([s1])

    await skill_tool.initialize([])

    assert skill_tool._skills == []
    assert skill_tool._tools == []
    assert skill_tool._tool_bindings == {}
    assert skill_tool._tool_name_index == {}
    assert skill_tool._tools_cache == []


# ----- #27: SkillTool.initialize() atomicity — failure-path (I1) ----- #


async def test_initialize_atomicity_preserves_state_on_tool_description_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I1: exception from _build_tool_description mid-loop preserves state.

    _build_tool_description has a stable signature (skill, manifest_tool)
    across #27, so this test gives a clean RED against current code.
    """
    skill_tool = _make_bare_skill_tool()
    s_good = _build_skill_with_tools(slug="good", tool_names=["t1"])
    await skill_tool.initialize([s_good])
    pre_call_snap = _snapshot_skill_tool(skill_tool)

    def raising_description(self, skill, manifest_tool):
        raise RuntimeError("simulated description-build failure")

    monkeypatch.setattr(
        SkillTool, "_build_tool_description", raising_description
    )

    s_bad = _build_skill_with_tools(slug="bad", tool_names=["tbad"])

    with pytest.raises(RuntimeError, match="simulated description-build failure"):
        await skill_tool.initialize([s_good, s_bad])

    _assert_skill_tool_matches(skill_tool, pre_call_snap)
    # Previously-valid tool is still invocable
    assert skill_tool.has_tool("skill_good_t1") is True


async def test_initialize_atomicity_preserves_state_on_is_model_invocable_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I1: exception from _is_model_invocable mid-loop preserves state.

    Simulates a real-world failure mode: a manifest with unexpected policy
    shape that triggers AttributeError inside _is_model_invocable.
    _is_model_invocable has a stable signature across #27.
    """
    skill_tool = _make_bare_skill_tool()
    s_good = _build_skill_with_tools(slug="good", tool_names=["t1"])
    await skill_tool.initialize([s_good])
    pre_call_snap = _snapshot_skill_tool(skill_tool)

    def raising_invocable(self, skill, manifest_tool):
        raise AttributeError("simulated is_model_invocable failure")

    monkeypatch.setattr(SkillTool, "_is_model_invocable", raising_invocable)

    s_bad = _build_skill_with_tools(slug="bad", tool_names=["tbad"])

    with pytest.raises(AttributeError, match="simulated is_model_invocable failure"):
        await skill_tool.initialize([s_good, s_bad])

    _assert_skill_tool_matches(skill_tool, pre_call_snap)


async def test_initialize_atomicity_preserves_state_on_build_function_name_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I1: exception from _build_function_name mid-loop preserves pre-call state.

    Added in Task 5 (not Task 4) because _build_function_name signature was
    changed in Task 5's refactor — a test with the new signature would have
    TypeError'd against the old call site.
    """
    skill_tool = _make_bare_skill_tool()
    s_good = _build_skill_with_tools(slug="good", tool_names=["t1"])
    await skill_tool.initialize([s_good])
    pre_call_snap = _snapshot_skill_tool(skill_tool)

    def raising_build(self, skill_slug, tool_name, name_index):
        raise RuntimeError("simulated name-build failure")

    monkeypatch.setattr(SkillTool, "_build_function_name", raising_build)

    s_bad = _build_skill_with_tools(slug="bad", tool_names=["tbad"])

    with pytest.raises(RuntimeError, match="simulated name-build failure"):
        await skill_tool.initialize([s_good, s_bad])

    _assert_skill_tool_matches(skill_tool, pre_call_snap)
    assert skill_tool.has_tool("skill_good_t1") is True


async def test_initialize_atomicity_no_intermediate_state_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I2 static verification: a helper called mid-loop cannot observe
    an intermediate view of self._* fields.

    This is not a concurrency test (asyncio body has no await), but a
    structural guardrail: if a future refactor reintroduces in-place
    mutation, this test catches it by recording what get_tools() would
    see at the moment _build_function_name runs inside the loop.
    """
    skill_tool = _make_bare_skill_tool()
    s_old = _build_skill_with_tools(slug="old", tool_names=["t_old"])
    await skill_tool.initialize([s_old])

    observed: list[dict] = []
    original = SkillTool._build_function_name

    def observing_build(self, skill_slug, tool_name, name_index):
        observed.append(
            {
                "tools_len": len(self._tools),
                "bindings_len": len(self._tool_bindings),
                "tools_cache_len": len(self._tools_cache or []),
            }
        )
        return original(self, skill_slug, tool_name, name_index)

    monkeypatch.setattr(SkillTool, "_build_function_name", observing_build)

    s_new = _build_skill_with_tools(slug="new", tool_names=["t_new"])
    await skill_tool.initialize([s_new])

    # Mid-loop observation MUST still show the OLD state (not a half-
    # built new state). Under build-in-locals + batch assign, this holds
    # trivially because self._* is only reassigned after the loop ends.
    assert len(observed) == 1
    assert observed[0]["tools_len"] == 1  # old: 1 tool [t_old]
    assert observed[0]["bindings_len"] == 1
    assert observed[0]["tools_cache_len"] == 1


async def test_cleanup_clears_all_five_fields() -> None:
    """cleanup() symmetrically clears all 5 fields covered by the
    atomicity contract (#27). Pre-#27 cleanup missed _tool_name_index."""
    skill_tool = _make_bare_skill_tool()
    s1 = _build_skill_with_tools(slug="sa", tool_names=["t1", "t2"])
    await skill_tool.initialize([s1])
    assert len(skill_tool._tools) == 2  # sanity

    await skill_tool.cleanup()

    assert skill_tool._skills == []
    assert skill_tool._tools == []
    assert skill_tool._tool_bindings == {}
    assert skill_tool._tool_name_index == {}
    assert skill_tool._tools_cache == []
