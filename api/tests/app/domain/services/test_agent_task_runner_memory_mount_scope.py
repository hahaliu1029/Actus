"""AgentTaskRunner 运行期 memory_mount_scope 接线回归（codex fix P0 round-2）。

M3 codex review round 1 的 fix 只接到了 PlannerReActFlow._collect_native_tools，
**漏了 AgentTaskRunner._build_lc_tools_full** —— step graph 每步 rebuild tool set
时绑到 LLM 的 file tools 是裸 create_native_tools(...) 不带 scope，结果 planner
阶段守护拦了、实际 tool call 路径仍透传。

本套测试钉死两条：
1. ``AgentTaskRunner._build_memory_mount_scope()`` 在条件满足时返回正确 scope
2. **源代码层 regression 检查**：``_build_lc_tools_full`` 和
   ``_get_native_tool_names_by_category`` 两处 ``create_native_tools`` 调用都
   显式写 ``memory_mount_scope=`` 关键字——未来 refactor 漏传会 fail。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner


class TestBuildMemoryMountScopeMethod:
    """``AgentTaskRunner._build_memory_mount_scope`` 行为——通过 fake self
    实例直接调，绕开 runner 完整初始化的重依赖链（sandbox / browser /
    mcp / a2a / skill tool 等）。"""

    def _fake_self(self, user_id: str | None) -> SimpleNamespace:
        return SimpleNamespace(_user_id=user_id)

    def test_returns_none_when_user_id_missing(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            memory_root_container="/tmp/memory",
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=True,
        )
        monkeypatch.setattr("core.config.get_settings", lambda: settings)

        fake = self._fake_self(None)
        result = AgentTaskRunner._build_memory_mount_scope(fake)  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_when_feature_gate_off(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            memory_root_container="/tmp/memory",
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=False,
        )
        monkeypatch.setattr("core.config.get_settings", lambda: settings)

        fake = self._fake_self("alice")
        assert AgentTaskRunner._build_memory_mount_scope(fake) is None  # type: ignore[arg-type]

    def test_returns_scope_when_enabled_with_user_id(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        settings = SimpleNamespace(
            memory_root_container=str(tmp_path),
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=True,
        )
        monkeypatch.setattr("core.config.get_settings", lambda: settings)

        fake = self._fake_self("alice")
        scope = AgentTaskRunner._build_memory_mount_scope(fake)  # type: ignore[arg-type]
        assert scope is not None
        assert scope.user_id == "alice"
        assert scope.sandbox_target == "/workspace/.memory"
        assert scope.container_root == tmp_path

    def test_returns_none_when_settings_import_fails(
        self, monkeypatch
    ) -> None:
        """get_settings 抛异常（测试绕 lifespan）→ factory 返 None，
        不让 tool 绑定 crash。"""
        def _boom():
            raise RuntimeError("no settings in test env")

        monkeypatch.setattr("core.config.get_settings", _boom)
        fake = self._fake_self("alice")
        assert AgentTaskRunner._build_memory_mount_scope(fake) is None  # type: ignore[arg-type]


class TestCreateNativeToolsCallSitesPassScope:
    """源代码层 regression：run-time tool binding 路径必须透传 scope。

    codex round-2 P0 的根因是"改了 planner_react 漏了 agent_task_runner"，
    AST-level 断言防止同类 regression —— 任何 ``create_native_tools(...)``
    调用 **在 agent_task_runner.py 内** 都必须含 ``memory_mount_scope=``
    关键字参数。未来新增 caller 会一起被抓。
    """

    @pytest.fixture(scope="class")
    def runner_ast(self) -> ast.Module:
        source_path = Path(inspect.getfile(AgentTaskRunner))
        return ast.parse(source_path.read_text(encoding="utf-8"))

    def _create_native_tools_calls(self, tree: ast.Module) -> list[ast.Call]:
        calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # 兼容 ``create_native_tools(...)`` 直接调用 以及
            # ``xxx.create_native_tools(...)`` attr 调用
            func = node.func
            if isinstance(func, ast.Name) and func.id == "create_native_tools":
                calls.append(node)
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "create_native_tools"
            ):
                calls.append(node)
        return calls

    def test_at_least_one_create_native_tools_call(
        self, runner_ast: ast.Module
    ) -> None:
        """Sanity：runner 必须调 create_native_tools（否则 test 失效）。"""
        calls = self._create_native_tools_calls(runner_ast)
        assert len(calls) >= 2, (
            f"expected ≥2 create_native_tools calls "
            f"(_get_native_tool_names_by_category + _build_lc_tools_full), "
            f"found {len(calls)}"
        )

    def test_every_create_native_tools_call_passes_scope(
        self, runner_ast: ast.Module
    ) -> None:
        """**核心断言**：每一处 create_native_tools 必须传 memory_mount_scope=。

        防止 refactor 或新增 tool binding site 时漏接 scope——codex round-1
        漏的就是这个。本测试作为 AST-level 护栏，比"跑一遍 step graph"
        轻得多但同样能抓住。
        """
        calls = self._create_native_tools_calls(runner_ast)
        assert calls, "sanity: expected ≥1 call"
        missing: list[int] = []
        for call in calls:
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg}
            if "memory_mount_scope" not in kwarg_names:
                missing.append(call.lineno)
        assert not missing, (
            f"create_native_tools calls missing memory_mount_scope at "
            f"agent_task_runner.py lines {missing}. 所有运行期 tool binding "
            f"都必须带客户端 symlink 守卫（codex fix P0 round-2）"
        )
