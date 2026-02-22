import asyncio
from dataclasses import dataclass

import pytest
from app.domain.models.app_config import A2AConfig, A2AServerConfig
from app.domain.services.tools.a2a import (
    A2AClientManager,
    A2ATool,
    _run_with_timeout_isolated,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class _FakeResponse:
    payload: dict

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeHTTPClient:
    def __init__(self) -> None:
        self.timeout_calls: list[float | None] = []

    async def get(self, url: str, timeout: float | None = None) -> _FakeResponse:
        self.timeout_calls.append(timeout)
        if "slow.example.com" in url:
            await asyncio.sleep(0.5)
        return _FakeResponse(
            {
                "name": url,
                "description": "ok",
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "capabilities": {"streaming": True},
            }
        )


async def test_get_a2a_agent_cards_continues_when_single_server_times_out(
    monkeypatch,
) -> None:
    manager = A2AClientManager(
        A2AConfig(
            a2a_servers=[
                A2AServerConfig(
                    id="slow",
                    base_url="https://slow.example.com",
                    enabled=True,
                ),
                A2AServerConfig(
                    id="fast",
                    base_url="https://fast.example.com",
                    enabled=True,
                ),
            ]
        )
    )
    fake_client = _FakeHTTPClient()
    manager._httpx_client = fake_client

    monkeypatch.setattr(
        "app.domain.services.tools.a2a.A2A_AGENT_CARD_TIMEOUT_SECONDS",
        0.05,
    )

    await asyncio.wait_for(manager._get_a2a_agent_cards(), timeout=0.2)

    assert "fast" in manager.agent_cards
    assert "slow" not in manager.agent_cards
    assert fake_client.timeout_calls
    assert fake_client.timeout_calls[0] == 0.05


async def test_get_a2a_agent_cards_skips_disabled_servers() -> None:
    manager = A2AClientManager(
        A2AConfig(
            a2a_servers=[
                A2AServerConfig(
                    id="disabled",
                    base_url="https://disabled.example.com",
                    enabled=False,
                ),
                A2AServerConfig(
                    id="enabled",
                    base_url="https://enabled.example.com",
                    enabled=True,
                ),
            ]
        )
    )
    fake_client = _FakeHTTPClient()
    manager._httpx_client = fake_client

    await manager._get_a2a_agent_cards()

    assert "enabled" in manager.agent_cards
    assert "disabled" not in manager.agent_cards


async def test_a2a_tool_initialize_degrades_when_manager_init_fails(monkeypatch) -> None:
    async def fake_initialize(self) -> None:
        raise RuntimeError("boom")

    async def fake_cleanup(self) -> None:
        return None

    monkeypatch.setattr(A2AClientManager, "initialize", fake_initialize)
    monkeypatch.setattr(A2AClientManager, "cleanup", fake_cleanup)

    tool = A2ATool()
    config = A2AConfig(
        a2a_servers=[
            A2AServerConfig(
                id="bad-agent",
                base_url="https://bad.example.com",
                enabled=True,
            )
        ]
    )

    await tool.initialize(config)
    cards = await tool.get_remote_agent_cards()

    assert cards.success is True
    assert cards.data == []


async def test_run_with_timeout_isolated_runs_in_current_task_context() -> None:
    parent_task = asyncio.current_task()
    probe_task = None

    async def probe() -> None:
        nonlocal probe_task
        probe_task = asyncio.current_task()

    await _run_with_timeout_isolated(probe(), timeout_seconds=1)

    assert probe_task is parent_task
