import asyncio

import pytest
from app.application.services.app_config_service import (
    AppConfigService,
    _run_probe_with_timeout,
)
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
    MCPTransport,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _InMemoryAppConfigRepo:
    def __init__(self, app_config: AppConfig) -> None:
        self._app_config = app_config

    def load(self) -> AppConfig:
        return self._app_config

    def save(self, app_config: AppConfig) -> None:
        self._app_config = app_config


def _build_app_config() -> AppConfig:
    return AppConfig(
        llm_config=LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="test",
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
        ),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(
            mcpServers={
                "demo-mcp": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://example.com/mcp",
                )
            }
        ),
        a2a_config=A2AConfig(
            a2a_servers=[
                A2AServerConfig(
                    id="a2a-demo",
                    base_url="https://example.com/a2a",
                    enabled=True,
                )
            ]
        ),
    )


async def test_get_mcp_servers_degrades_when_probe_is_cancelled(monkeypatch) -> None:
    cleanup_called = 0

    class _FakeMCPClientManager:
        def __init__(self, mcp_config: MCPConfig) -> None:
            self.tools = {}

        async def initialize(self) -> None:
            raise asyncio.CancelledError("probe cancelled")

        async def cleanup(self) -> None:
            nonlocal cleanup_called
            cleanup_called += 1

    monkeypatch.setattr(
        "app.application.services.app_config_service.MCPClientManager",
        _FakeMCPClientManager,
    )

    service = AppConfigService(_InMemoryAppConfigRepo(_build_app_config()))
    servers = await service.get_mcp_servers()

    assert len(servers) == 1
    assert servers[0].server_name == "demo-mcp"
    assert servers[0].tools == []
    assert cleanup_called == 1


async def test_get_a2a_servers_degrades_when_probe_is_cancelled(monkeypatch) -> None:
    cleanup_called = 0

    class _FakeA2AClientManager:
        def __init__(self, a2a_config: A2AConfig) -> None:
            self.agent_cards = {}

        async def initialize(self) -> None:
            raise asyncio.CancelledError("probe cancelled")

        async def cleanup(self) -> None:
            nonlocal cleanup_called
            cleanup_called += 1

    monkeypatch.setattr(
        "app.application.services.app_config_service.A2AClientManager",
        _FakeA2AClientManager,
    )

    service = AppConfigService(_InMemoryAppConfigRepo(_build_app_config()))
    servers = await service.get_a2a_servers()

    assert len(servers) == 1
    assert servers[0].id == "a2a-demo"
    assert servers[0].name == "example.com/a2a"
    assert cleanup_called == 1


async def test_run_probe_with_timeout_runs_in_current_task_context() -> None:
    parent_task = asyncio.current_task()
    probe_task = None

    async def probe() -> None:
        nonlocal probe_task
        probe_task = asyncio.current_task()

    await _run_probe_with_timeout(probe(), timeout_seconds=1)

    assert probe_task is parent_task
