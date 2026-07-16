"""Task 29: SkillTool._invoke_native now delegates to N1 AST validator.

Primary check is ``shell_ast_validator.validate()``; the legacy
``_contains_blocked_command`` regex path remains as a belt-and-suspenders
secondary layer and will be removed in R5+CS4.

These tests verify:
1. AST validator blocks fs_destructive / system_admin commands and
   returns a ToolResult with the 5-line ``[AST 拦截]`` template.
2. Legacy regex still fires on patterns the AST does not flag (e.g. a
   random ``blocked`` substring) — proves the belt-and-suspenders
   wiring is intact.
3. Benign commands pass both layers and reach ``sandbox.exec_command``.
4. Structural: the module imports the validator symbols, proving the
   integration is in place.
"""
from __future__ import annotations

import inspect

import pytest

from app.application.services.sandbox_accessors import EagerSandboxAccessor
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.tool_result import AllowError, AllowSuccess, ToolResult
from app.domain.services.tools.skill import SkillTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Fakes (mirror the patterns in tests/domain/services/test_skill_tool.py)
# ---------------------------------------------------------------------------


class _FakeSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.available_dirs = {
            "/home/ubuntu/workspace",
            "/home/ubuntu/workspace/.skills/demo-native--1234abcd",
        }

    async def exec_command(self, session_id: str, exec_dir: str, command: str) -> ToolResult:
        self.calls.append((session_id, exec_dir, command))
        return ToolResult(success=True, message="ok", data={"session_id": session_id})

    async def read_shell_output(self, session_id: str, console: bool = False) -> ToolResult:
        return ToolResult(success=True, data={"session_id": session_id, "output": "native-ok"})

    async def check_file_exists(self, filepath: str) -> ToolResult:
        return ToolResult(success=True, data={"exists": filepath in self.available_dirs})


class _FakeMCPTool:
    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        return ToolResult(success=True, data="mcp-ok")


class _FakeA2ATool:
    async def call_remote_agent(self, id: str, query: str) -> ToolResult:
        return ToolResult(success=True, data="a2a-ok")


# ---------------------------------------------------------------------------
# Skill factory — adapted to real Skill schema (source_type / source_ref
# required; no version/trust_origin/scan_report kwargs from plan template).
# ---------------------------------------------------------------------------


def _make_skill_with_command(command: str, *, slug: str = "demo-native") -> Skill:
    return Skill(
        id=f"{slug}--1234abcd",
        slug=slug,
        name="Demo Native",
        source_type=SkillSourceType.GITHUB,
        source_ref="github:owner/repo",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": "Demo Native",
            "runtime_type": "native",
            "bundle_file_count": 1,
            "last_sync_at": "v1",
            "tools": [
                {
                    "name": "run_demo",
                    "description": "run",
                    "parameters": {"target": {"type": "string"}},
                    "required": ["target"],
                    "entry": {
                        "exec_dir": "/home/ubuntu/workspace",
                        "command": command,
                    },
                }
            ],
        },
        installed_by="admin-1",
    )


def _make_skill_tool(**overrides) -> SkillTool:
    defaults: dict = {
        "sandbox_accessor": EagerSandboxAccessor(_FakeSandbox()),
        "mcp_tool": _FakeMCPTool(),
        "a2a_tool": _FakeA2ATool(),
    }
    defaults.update(overrides)
    return SkillTool(**defaults)


# ---------------------------------------------------------------------------
# Structural test (runs even before Step 3 wiring — expected to FAIL in RED).
# ---------------------------------------------------------------------------


def test_skill_module_imports_validator() -> None:
    """_invoke_native must reference shell_ast_validator (primary check).

    Uses ``inspect.getsource`` rather than ``import ... as _`` so the
    check survives module-level caching and is resilient to where the
    import lives (top-level vs. deferred inside the method body).
    """
    src = inspect.getsource(SkillTool._invoke_native)
    assert "shell_ast_validator" in src, (
        "_invoke_native must invoke shell_ast_validator.validate() as the "
        "primary safety gate (N1 Task 29)."
    )
    assert "validate(" in src
    assert "to_legacy_tool_result" in src


# ---------------------------------------------------------------------------
# Integration: AST validator blocks dangerous command.
# ---------------------------------------------------------------------------


async def test_ast_validator_blocks_fs_destructive_rm_rf() -> None:
    """`rm -rf /` must be denied by AST validator (fs_destructive or
    cwd_boundary, per validator precedence) BEFORE sandbox.exec_command
    is reached. The ToolResult surfaces the ``[AST 拦截]`` template."""
    sandbox = _FakeSandbox()
    skill_tool = _make_skill_tool(sandbox_accessor=EagerSandboxAccessor(sandbox))
    skill = _make_skill_with_command("rm -rf /")

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="x")

    assert isinstance(result, AllowError)
    assert "[AST 拦截]" in result.content
    # No exec_command call — denial happens before sandbox dispatch.
    assert sandbox.calls == []


async def test_ast_validator_blocks_system_admin_command() -> None:
    """`mount /dev/sda1 /mnt` must be denied (system_admin category)."""
    sandbox = _FakeSandbox()
    skill_tool = _make_skill_tool(sandbox_accessor=EagerSandboxAccessor(sandbox))
    skill = _make_skill_with_command("mount /dev/sda1 /mnt")

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="x")

    assert isinstance(result, AllowError)
    assert "[AST 拦截]" in result.content
    assert (
        "系统级操作" in result.content
        or "system_admin" in result.content.lower()
        or "mount" in result.content
    )
    assert sandbox.calls == []


# ---------------------------------------------------------------------------
# Integration: legacy regex still acts as secondary layer.
# ---------------------------------------------------------------------------


async def test_legacy_regex_fires_on_patterns_ast_misses() -> None:
    """Configure an AST-benign command (`echo blocked`) with legacy
    blocklist that matches the substring. AST allows it; legacy denies.
    Belt-and-suspenders message must differ from the AST template."""
    sandbox = _FakeSandbox()
    skill_tool = _make_skill_tool(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        blocked_command_patterns=["blocked"],
    )
    skill = _make_skill_with_command("echo blocked")

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="x")

    assert isinstance(result, AllowError)
    # Legacy denial path — distinct from AST template.
    assert "legacy blocklist" in result.content
    assert "[AST 拦截]" not in result.content
    assert sandbox.calls == []


# ---------------------------------------------------------------------------
# Integration: benign command flows through both layers to the sandbox.
# ---------------------------------------------------------------------------


async def test_benign_command_passes_both_layers() -> None:
    sandbox = _FakeSandbox()
    skill_tool = _make_skill_tool(sandbox_accessor=EagerSandboxAccessor(sandbox))
    skill = _make_skill_with_command("echo hello")

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="x")

    assert isinstance(result, AllowSuccess)
    assert sandbox.calls, "sandbox.exec_command should be called for benign command"
    _, _, command = sandbox.calls[0]
    assert "echo hello" in command


# ---------------------------------------------------------------------------
# C5b §8.5 — decision rerouted through the command-policy evaluator.
# ---------------------------------------------------------------------------


def test_invoke_native_source_reroutes_to_evaluator() -> None:
    """_invoke_native's decision must flow through the evaluator AND the old
    `if not ast_result.allowed` predicate must be GONE. Source smoke (mirrors
    test_skill_module_imports_validator); the authoritative AST lock is in Task 6.
    The `not in` check is what makes this RED until Step 4 actually reroutes the
    decision — adding the evaluator import alone (Step 3) is not enough. [codex planR4 P2]"""
    src = inspect.getsource(SkillTool._invoke_native)
    assert "build_command_policy" in src
    assert "evaluate_command" in src
    assert "if not ast_result.allowed" not in src  # old predicate replaced (RED until Step 4)


async def test_invoke_native_emits_no_policy_snapshot(monkeypatch) -> None:
    """§8.5 / §10: _invoke_native reroutes the DECISION but emits no policy snapshot
    (no Seam-B sink here). Prove the compiler is never invoked from _invoke_native."""
    from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler

    called: list = []
    monkeypatch.setattr(
        SandboxPolicyCompiler, "compile_tool_call",
        lambda self, inp: called.append(inp),
    )
    sandbox = _FakeSandbox()
    skill_tool = _make_skill_tool(sandbox_accessor=EagerSandboxAccessor(sandbox))
    skill = _make_skill_with_command("rm -rf /")

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="x")

    assert isinstance(result, AllowError)
    assert "[AST 拦截]" in result.content
    assert called == []  # no snapshot compiled from _invoke_native


async def test_invoke_native_denial_diagnosis_byte_identical() -> None:
    """INV-6 (§8.10): the native denial content is byte-identical to the validator's
    to_legacy_tool_result output — only the decision predicate changed. The invoke()
    wrapper maps ToolResult.message -> AllowError.content verbatim (skill.py:303-305)."""
    from app.domain.services.safety.shell_ast_validator import (
        to_legacy_tool_result,
        validate,
    )

    command = "rm -rf /"
    exec_dir = "/home/ubuntu/workspace"  # _make_skill_with_command's entry.exec_dir
    expected = to_legacy_tool_result(
        validate(command, effective_cwd=exec_dir), original_command=command
    )
    sandbox = _FakeSandbox()
    skill_tool = _make_skill_tool(sandbox_accessor=EagerSandboxAccessor(sandbox))
    skill = _make_skill_with_command(command)

    await skill_tool.initialize([skill])
    result = await skill_tool.invoke("skill_demo_native_run_demo", target="x")

    assert isinstance(result, AllowError)
    assert result.content == expected.message
