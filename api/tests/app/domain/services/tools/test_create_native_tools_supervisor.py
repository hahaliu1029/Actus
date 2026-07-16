from unittest.mock import AsyncMock

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.tools._supervisor_tool_wrapper import (
    SupervisorAwareToolWrapper,
)
from app.domain.services.tools.langchain_tools import create_native_tools


def test_create_native_tools_without_supervisor_returns_inner_tools() -> None:
    tools = create_native_tools(
        sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
        browser_accessor=EagerBrowserAccessor(AsyncMock()),
        search_engine=AsyncMock(),
    )

    assert tools
    assert not any(isinstance(tool, SupervisorAwareToolWrapper) for tool in tools)


def test_create_native_tools_with_supervisor_wraps_all_tools() -> None:
    supervisor = object()

    tools = create_native_tools(
        sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
        browser_accessor=EagerBrowserAccessor(AsyncMock()),
        search_engine=AsyncMock(),
        supervisor=supervisor,
    )

    assert tools
    assert all(isinstance(tool, SupervisorAwareToolWrapper) for tool in tools)


def test_create_native_tools_supervisor_wrapper_preserves_risk_tool_metadata() -> None:
    tools = create_native_tools(
        sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
        browser_accessor=EagerBrowserAccessor(AsyncMock()),
        search_engine=AsyncMock(),
        supervisor=object(),
    )

    shell_execute = next(tool for tool in tools if tool.name == "shell_execute")

    assert isinstance(shell_execute, SupervisorAwareToolWrapper)
    assert shell_execute.response_format == "content_and_artifact"
    assert shell_execute.metadata["risk_level"] == "high"
