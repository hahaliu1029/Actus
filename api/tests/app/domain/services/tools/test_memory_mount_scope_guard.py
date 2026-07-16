"""客户端侧 symlink 守护（M3 codex fix P0 / design §724 Case C (iii)）。

覆盖两层：
1. ``MemoryMountScope.map_to_host_path`` 的路径映射纯逻辑（边界值 / `..` 拒绝）
2. ``_make_file_tools`` 在 scope 非空时对 symlink 的预检行为——关键不变式：
   **symlink 命中时绝不发 sandbox HTTP 请求**，直接客户端返 AllowError。

不走 pytest sandbox marker——这是纯 in-process 单测，和 ``tests/sandbox/``
的 kernel 级对抗测试正交：后者测 "kernel :ro 行为"，本套测 "客户端守护"。
双层覆盖才满足 design §704 的三层纵深防御。
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.application.services.sandbox_accessors import EagerSandboxAccessor
from app.domain.services.tools.langchain_tools import _make_file_tools
from app.domain.services.tools.memory_mount_scope import (
    MemoryMountScope,
    build_memory_mount_scope_from_settings,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ─── MemoryMountScope.map_to_host_path ──────────────────────────────────────


class TestMapToHostPath:
    def test_exact_target_match_maps_to_user_root(self, tmp_path: Path) -> None:
        """filepath 正好等于 sandbox_target → 映射到 container_root/user_id。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        mapped = scope.map_to_host_path("/workspace/.memory")
        assert mapped == tmp_path / "alice"

    def test_under_target_maps_with_relative(self, tmp_path: Path) -> None:
        """filepath 在 target 下 → 拼 user_id + relative。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        mapped = scope.map_to_host_path("/workspace/.memory/user/abc.md")
        assert mapped == tmp_path / "alice" / "user" / "abc.md"

    def test_outside_target_returns_none(self, tmp_path: Path) -> None:
        """filepath 不在 target 前缀下 → None（跳过 guard）。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.map_to_host_path("/etc/passwd") is None
        assert scope.map_to_host_path("/workspace/other") is None

    def test_similar_prefix_not_matched(self, tmp_path: Path) -> None:
        """``/workspace/.memory_evil`` 不能被 ``/workspace/.memory`` 前缀
        误配——严格要求后续是 ``/`` 或结尾。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.map_to_host_path("/workspace/.memory_evil/x") is None
        assert scope.map_to_host_path("/workspace/.memoryx") is None

    def test_dotdot_in_relative_rejected(self, tmp_path: Path) -> None:
        """relative 里含 ``..`` → None，防止 ``/workspace/.memory/../../etc/passwd``
        伪造路径绕过 api 容器 bind source 边界。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.map_to_host_path("/workspace/.memory/../etc/passwd") is None
        assert scope.map_to_host_path("/workspace/.memory/user/../../etc") is None

    def test_trailing_slash_tolerated(self, tmp_path: Path) -> None:
        """target 可带或不带尾斜杠，语义一致。"""
        scope_with = MemoryMountScope(
            container_root=tmp_path,
            user_id="u",
            sandbox_target="/workspace/.memory/",
        )
        scope_without = MemoryMountScope(
            container_root=tmp_path,
            user_id="u",
            sandbox_target="/workspace/.memory",
        )
        assert scope_with.map_to_host_path(
            "/workspace/.memory/x.md"
        ) == scope_without.map_to_host_path("/workspace/.memory/x.md")


# ─── _make_file_tools 客户端守护 ──────────────────────────────────────────────


def _tool_by_name(tools, name: str):
    """从 _make_file_tools 返回的 list 里按 name 取工具。"""
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name} not found in {[t.name for t in tools]}")


class TestFileToolsGuardDisabledWhenNoScope:
    """scope=None → 守护关闭，工具行为与旧版一致。"""

    async def test_file_read_transparently_delegates(self) -> None:
        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(return_value=_ok_result("hello"))

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=None)
        file_read = _tool_by_name(tools, "file_read")

        result = await file_read.ainvoke({"filepath": "/workspace/.memory/evil.md"})

        # 未启用守护 → 无论 path 是什么都透传给 sandbox.read_file
        sandbox.read_file.assert_awaited_once()
        # 结果从 sandbox 正常返回
        assert "hello" in _extract_content(result)


class TestAnyAncestorIsSymlink:
    """``MemoryMountScope.any_ancestor_is_symlink`` 逐段 lstat 检查
    （codex fix P1 round-2：单查叶子会被目录 symlink 旁路绕过）。"""

    def test_leaf_is_symlink_detected(self, tmp_path: Path) -> None:
        """叶子 symlink：旧守卫能抓的场景，新实现向下兼容。"""
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        victim = tmp_path / "outside.txt"
        victim.write_text("x")
        try:
            (user_dir / "evil.md").symlink_to(victim)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.any_ancestor_is_symlink("/workspace/.memory/evil.md") is True

    def test_category_dir_symlink_detected_even_if_leaf_is_regular(
        self, tmp_path: Path
    ) -> None:
        """**核心旁路回归**：攻击者把 ``{user_id}/user`` 做成 symlink 指向
        ``/etc``；叶子 ``secret.txt`` 自身不是 symlink（它是 target 里的真实
        文件），但父目录是 → 读透穿到 /etc/secret.txt。旧守卫漏（codex
        复现过），新守卫必须抓住。"""
        # 构造：host 侧有真实敏感目录 + 有内容
        real_etc = tmp_path / "real_etc"
        real_etc.mkdir()
        (real_etc / "secret.txt").write_text("sensitive", encoding="utf-8")

        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        # user_id/user 做成 symlink 指向真实目录
        try:
            (user_dir / "user").symlink_to(real_etc)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        # 叶子本身：is_symlink() 返 False（secret.txt 是 target 里真实文件）
        assert (user_dir / "user" / "secret.txt").is_symlink() is False
        # 祖先检查：必须返 True —— 这是本次 fix 要钉死的行为
        assert scope.any_ancestor_is_symlink(
            "/workspace/.memory/user/secret.txt"
        ) is True

    def test_user_id_dir_itself_symlink_detected(self, tmp_path: Path) -> None:
        """user_id 根目录自身是 symlink → scope 整个都不可信，必须拒绝。"""
        real_target = tmp_path / "attacker_dir"
        real_target.mkdir()
        (real_target / "whatever.md").write_text("x", encoding="utf-8")

        try:
            (tmp_path / "alice").symlink_to(real_target)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.any_ancestor_is_symlink(
            "/workspace/.memory/whatever.md"
        ) is True

    def test_no_symlink_anywhere_returns_false(self, tmp_path: Path) -> None:
        """全部规范目录 + 文件 → 返 False，守卫不误伤。"""
        user_dir = tmp_path / "alice" / "user"
        user_dir.mkdir(parents=True)
        (user_dir / "abc.md").write_text("normal", encoding="utf-8")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.any_ancestor_is_symlink(
            "/workspace/.memory/user/abc.md"
        ) is False

    def test_outside_scope_returns_false_regardless(
        self, tmp_path: Path
    ) -> None:
        """path 不在 memory scope 内 → 跳过 guard 返 False。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        assert scope.any_ancestor_is_symlink("/etc/passwd") is False


class TestBuildFactory:
    """``build_memory_mount_scope_from_settings`` 共享 factory —— 避免
    planner_react / agent_task_runner 两条路径构造逻辑漂移
    （codex fix P0 round-2）。"""

    def test_returns_none_when_user_id_missing(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        settings = SimpleNamespace(
            memory_root_container=str(tmp_path),
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=True,
        )
        assert build_memory_mount_scope_from_settings(None, settings) is None
        assert build_memory_mount_scope_from_settings("", settings) is None

    def test_returns_none_when_feature_gate_off(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        settings = SimpleNamespace(
            memory_root_container=str(tmp_path),
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=False,
        )
        assert build_memory_mount_scope_from_settings("alice", settings) is None

    def test_returns_scope_when_all_conditions_met(
        self, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        settings = SimpleNamespace(
            memory_root_container=str(tmp_path),
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=True,
        )
        scope = build_memory_mount_scope_from_settings("alice", settings)
        assert scope is not None
        assert scope.user_id == "alice"
        assert scope.sandbox_target == "/workspace/.memory"
        assert scope.container_root == tmp_path


class TestFileToolsGuardActiveAndRefusesSymlinks:
    """scope set + target 下的 symlink：客户端直接拒，**永不**调 sandbox。"""

    async def test_file_read_refuses_symlink_in_mount_scope(
        self, tmp_path: Path
    ) -> None:
        # 1) 构造真实 fs：user dir + 一个 symlink 到外部 "敏感文件"
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        victim = tmp_path / "outside" / "sensitive.txt"
        victim.parent.mkdir()
        victim.write_text("secret", encoding="utf-8")

        symlink = user_dir / "evil.md"
        try:
            os.symlink(victim, symlink)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        # 2) 构造 scope + mock sandbox
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock()  # 不应被调用

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        file_read = _tool_by_name(tools, "file_read")

        # 3) agent 请求读 symlink path（bind mount 内）
        result = await file_read.ainvoke(
            {"filepath": "/workspace/.memory/evil.md"}
        )

        # 4) 断言：sandbox.read_file **绝不**被调（HTTP 未发）
        sandbox.read_file.assert_not_called()
        # 5) 工具返回客户端 AllowError，content 含拒绝原因
        content = _extract_content(result)
        assert "symlink" in content.lower() or "refuse" in content.lower()
        assert "/workspace/.memory/evil.md" in content

    async def test_file_str_replace_refuses_symlink(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        victim = tmp_path / "outside" / "sensitive.txt"
        victim.parent.mkdir()
        victim.write_text("secret")
        symlink = user_dir / "evil.md"
        try:
            os.symlink(victim, symlink)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.replace_in_file = AsyncMock()

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        tool = _tool_by_name(tools, "file_str_replace")

        await tool.ainvoke(
            {
                "filepath": "/workspace/.memory/evil.md",
                "old_str": "x",
                "new_str": "y",
            }
        )

        sandbox.replace_in_file.assert_not_called()

    async def test_file_find_in_content_refuses_symlink(
        self, tmp_path: Path
    ) -> None:
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        victim = tmp_path / "outside" / "sensitive.txt"
        victim.parent.mkdir()
        victim.write_text("secret")
        symlink = user_dir / "evil.md"
        try:
            os.symlink(victim, symlink)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.search_in_file = AsyncMock()

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        tool = _tool_by_name(tools, "file_find_in_content")

        await tool.ainvoke(
            {"filepath": "/workspace/.memory/evil.md", "regex": ".*"}
        )

        sandbox.search_in_file.assert_not_called()

    async def test_file_read_refuses_when_category_dir_is_symlink(
        self, tmp_path: Path
    ) -> None:
        """**端到端祖先旁路回归**：攻击者把 ``{user_id}/user`` 做成 symlink，
        读 ``/workspace/.memory/user/secret.txt`` 必须被客户端守卫拦——
        sandbox.read_file 绝不被调。codex round-2 fix P1 钉死。"""
        real_etc = tmp_path / "real_etc"
        real_etc.mkdir()
        (real_etc / "secret.txt").write_text("sensitive")

        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        try:
            (user_dir / "user").symlink_to(real_etc)
        except (OSError, NotImplementedError):
            pytest.skip("fs 不支持 symlink")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock()  # 不应被调用

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        file_read = _tool_by_name(tools, "file_read")

        result = await file_read.ainvoke(
            {"filepath": "/workspace/.memory/user/secret.txt"}
        )

        sandbox.read_file.assert_not_called()
        content = _extract_content(result)
        assert "symlink" in content.lower() or "refuse" in content.lower()


class TestFileToolsGuardPassesThroughNonSymlinks:
    """scope set + path 是常规文件：正常透传给 sandbox，守护不干扰合法路径。"""

    async def test_file_read_delegates_for_regular_file(
        self, tmp_path: Path
    ) -> None:
        user_dir = tmp_path / "alice" / "user"
        user_dir.mkdir(parents=True)
        (user_dir / "abc.md").write_text("normal content", encoding="utf-8")

        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(return_value=_ok_result("normal content"))

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        file_read = _tool_by_name(tools, "file_read")

        result = await file_read.ainvoke(
            {"filepath": "/workspace/.memory/user/abc.md"}
        )

        sandbox.read_file.assert_awaited_once()
        assert "normal content" in _extract_content(result)

    async def test_file_read_delegates_for_path_outside_scope(
        self, tmp_path: Path
    ) -> None:
        """path 不在 sandbox_target 下——守护跳过，正常透传。即便 path
        上有 symlink 也不属于本守护职责（memory mount 外的沙箱路径遵循
        既有沙箱安全边界）。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(return_value=_ok_result("other"))

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        file_read = _tool_by_name(tools, "file_read")

        await file_read.ainvoke({"filepath": "/workspace/other_area/x.md"})

        # 守护未拦截——正常走 sandbox HTTP
        sandbox.read_file.assert_awaited_once()

    async def test_file_read_delegates_for_path_with_dotdot(
        self, tmp_path: Path
    ) -> None:
        """path 含 ``..`` → map_to_host_path 返 None → 跳过 guard 透传。
        sandbox 自己应该拒绝异常路径（非本守护职责）。"""
        scope = MemoryMountScope(
            container_root=tmp_path,
            user_id="alice",
            sandbox_target="/workspace/.memory",
        )
        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(return_value=_ok_result("x"))

        tools = _make_file_tools(EagerSandboxAccessor(sandbox), memory_mount_scope=scope)
        file_read = _tool_by_name(tools, "file_read")

        await file_read.ainvoke(
            {"filepath": "/workspace/.memory/../../etc/passwd"}
        )

        sandbox.read_file.assert_awaited_once()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _ok_result(text: str):
    """构造一个 sandbox.read_file 成功返回的 ToolResult 模拟对象。

    ``_invoke_result_tool`` 读取的属性 roughly: .ok 或 dict。做个轻量 stub。
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        text=text,
        content=text,
        ok=True,
        is_success=True,
        data={"content": text},
        error=None,
    )


def _extract_content(tool_result) -> str:
    """LangChain ``response_format='content_and_artifact'`` 工具的返回是
    ``(content_str, artifact)``；但 ``ainvoke`` 可能返字符串或元组。统一取 str。"""
    if isinstance(tool_result, tuple) and len(tool_result) == 2:
        return str(tool_result[0])
    if isinstance(tool_result, str):
        return tool_result
    return str(tool_result)
