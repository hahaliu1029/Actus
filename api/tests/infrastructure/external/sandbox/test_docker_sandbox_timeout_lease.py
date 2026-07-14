"""DockerSandbox reset-timeout lease adapter contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.external.sandbox import Sandbox
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sandbox(response: MagicMock) -> tuple[DockerSandbox, AsyncMock]:
    sandbox = DockerSandbox(ip="1.2.3.4")
    post = AsyncMock(return_value=response)
    sandbox.client = SimpleNamespace(post=post)
    return sandbox, post


async def test_renew_timeout_lease_posts_default_window_payload() -> None:
    response = MagicMock()
    sandbox, post = _sandbox(response)

    result = await sandbox.renew_timeout_lease()

    assert result is None
    post.assert_awaited_once_with(
        "http://1.2.3.4:8080/api/supervisor/reset-timeout",
        json={"minutes": None},
    )
    response.raise_for_status.assert_called_once_with()


async def test_renew_timeout_lease_posts_explicit_window() -> None:
    response = MagicMock()
    sandbox, post = _sandbox(response)

    await sandbox.renew_timeout_lease(7)

    assert post.await_args.kwargs["json"] == {"minutes": 7}


async def test_renew_timeout_lease_propagates_http_error() -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = RuntimeError("sandbox unavailable")
    sandbox, _ = _sandbox(response)

    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        await sandbox.renew_timeout_lease()


def test_sandbox_protocol_declares_timeout_lease_renewal() -> None:
    assert "renew_timeout_lease" in Sandbox.__dict__

