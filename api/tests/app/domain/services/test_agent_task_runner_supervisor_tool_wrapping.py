"""B3-core PR-3b Task 5: runner-level supervisor tool wrapping guards."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import app.domain.services.tools.langchain_a2a as langchain_a2a
import app.domain.services.tools.langchain_dynamic_skill_tools as dynamic_skill_tools
import app.domain.services.tools.langchain_mcp as langchain_mcp
import app.domain.services.tools.langchain_skill_tools as langchain_skill_tools
import app.domain.services.tools.langchain_tools as langchain_tools
import app.domain.services.tools.memory_tools as memory_tools
from app.application.services.agent_service import AgentService
from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.tools._supervisor_tool_wrapper import (
    SupervisorAwareToolWrapper,
)


class _ToolArgs(BaseModel):
    value: str


def _make_tool(name: str):
    async def _run(value: str) -> str:
        return value

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=f"{name} desc",
        args_schema=_ToolArgs,
    )


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _call_names(node: ast.AST, name: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            calls.append(child)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            calls.append(child)
    return calls


def test_build_lc_tools_full_passes_supervisor_to_native_factory() -> None:
    source_path = Path(inspect.getfile(AgentTaskRunner))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = _function_node(tree, "_build_lc_tools_full")

    calls = _call_names(method, "create_native_tools")
    assert calls, "sanity: _build_lc_tools_full must call create_native_tools"

    missing = []
    for call in calls:
        kwarg_names = {kw.arg for kw in call.keywords if kw.arg}
        if "supervisor" not in kwarg_names:
            missing.append(call.lineno)

    assert not missing, (
        "create_native_tools calls in _build_lc_tools_full missing "
        f"supervisor= at lines {missing}"
    )


def test_build_lc_tools_full_wraps_final_aggregate() -> None:
    source_path = Path(inspect.getfile(AgentTaskRunner))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = _function_node(tree, "_build_lc_tools_full")

    returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
    assert returns, "sanity: _build_lc_tools_full must return tools"

    assert any(
        isinstance(ret.value, ast.Call)
        and isinstance(ret.value.func, ast.Name)
        and ret.value.func.id == "wrap_tool_list_for_supervisor"
        for ret in returns
    ), "_build_lc_tools_full must return wrap_tool_list_for_supervisor(...)"


def test_build_lc_tools_full_wraps_non_native_aggregate_tools(
    monkeypatch,
) -> None:
    supervisor = object()
    native_inner = _make_tool("native_wrapped")
    native_wrapped = SupervisorAwareToolWrapper(
        inner=native_inner,
        supervisor=supervisor,
    )
    mcp_tool = _make_tool("mcp_mock")
    a2a_tool = _make_tool("call_remote_agent")
    static_skill_tool = _make_tool("generate_skill")
    dynamic_skill_tool = _make_tool("dynamic_skill_mock")
    guide_tool = _make_tool("get_skill_guide")
    memory_tool = _make_tool("memory_search")

    monkeypatch.setattr(
        langchain_tools,
        "create_native_tools",
        lambda **kwargs: [native_wrapped],
    )
    monkeypatch.setattr(
        langchain_mcp,
        "create_mcp_langchain_tools",
        lambda *args, **kwargs: [mcp_tool],
    )
    monkeypatch.setattr(
        langchain_a2a,
        "create_a2a_langchain_tools",
        lambda *args, **kwargs: [a2a_tool],
    )
    monkeypatch.setattr(
        langchain_skill_tools,
        "create_skill_langchain_tools",
        lambda *args, **kwargs: [static_skill_tool],
    )
    monkeypatch.setattr(
        dynamic_skill_tools,
        "create_dynamic_skill_langchain_tools",
        lambda *args, **kwargs: [dynamic_skill_tool],
    )
    monkeypatch.setattr(
        langchain_skill_tools,
        "create_skill_guide_tool",
        lambda *args, **kwargs: guide_tool,
    )
    monkeypatch.setattr(
        memory_tools,
        "create_memory_tools",
        lambda *args, **kwargs: [memory_tool],
    )

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._sandbox_accessor = EagerSandboxAccessor(MagicMock())
    runner._browser_accessor = EagerBrowserAccessor(MagicMock())
    runner._search_engine = MagicMock()
    runner._file_processor_lookup = None
    runner._supports_vision = True
    runner._supports_pdf_input = False
    runner._build_memory_mount_scope = lambda: None
    runner._execution_supervisor = supervisor
    runner._mcp_tool = SimpleNamespace(
        get_tools=lambda: [{"function": {"name": "mcp_mock"}}],
    )
    runner._image_url_map = {}
    runner._upload_sandbox_file_for_mcp = MagicMock()
    runner._a2a_tool = MagicMock()
    runner._brainstorm_skill_tool = MagicMock()
    runner._create_skill_tool = MagicMock()
    runner._skill_tool = MagicMock()
    runner._session_skill_pool = []
    runner._skill_bundle_sync = SimpleNamespace(
        get_file_listing_all=lambda: {},
        sandbox_skill_root="/workspace/.skills",
    )
    runner._memory_embedding_provider = MagicMock()
    runner._memory_session_factory = MagicMock()
    runner._memory_repo_factory = MagicMock()
    runner._memory_write_service = MagicMock()
    runner._memory_session_redis = MagicMock()
    runner._memory_session_save_cap = 20
    runner._user_id = "user-1"
    runner._session_id = "session-1"
    runner._flow = SimpleNamespace(
        _memory_config=SimpleNamespace(half_life_days=30, mmr_lambda=0.5),
    )
    # B12/a1d84f6: _build_lc_tools_full reads self._tool_runtime.file_view_* at
    # agent_task_runner.py:2233 (unconditional) — provide a real config.
    from app.domain.models.app_config import ToolRuntimeConfig
    runner._tool_runtime = ToolRuntimeConfig()

    tools = AgentTaskRunner._build_lc_tools_full(runner)

    assert tools[0] is native_wrapped
    assert all(isinstance(tool, SupervisorAwareToolWrapper) for tool in tools)
    assert [tool.name for tool in tools] == [
        "native_wrapped",
        "mcp_mock",
        "call_remote_agent",
        "generate_skill",
        "dynamic_skill_mock",
        "get_skill_guide",
        "memory_search",
    ]


def test_agent_service_threads_supervisor_into_task_runner() -> None:
    source_path = Path(inspect.getfile(AgentService))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = _function_node(tree, "_create_task")

    calls = _call_names(method, "AgentTaskRunner")
    assert calls, "sanity: AgentService._create_task must construct AgentTaskRunner"
    runner_call = calls[0]

    kwarg_names = {kw.arg for kw in runner_call.keywords if kw.arg}
    assert "execution_supervisor" in kwarg_names
