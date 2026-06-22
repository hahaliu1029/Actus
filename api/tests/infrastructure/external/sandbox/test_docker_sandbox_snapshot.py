"""S2 PR-1 — DockerSandbox.snapshot_workspace POSTs to /api/file/snapshot-workspace."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_snapshot_workspace_posts_with_caps() -> None:
    sandbox = DockerSandbox(ip="1.2.3.4")
    resp = MagicMock()
    resp.json = MagicMock(return_value={
        "code": 200, "msg": "ok",
        "data": {"entries": {}, "truncated": False},
    })
    sandbox.client.post = AsyncMock(return_value=resp)

    result = await sandbox.snapshot_workspace(
        max_paths=1000, max_files=500, max_total_bytes=1_000_000,
        max_seconds=5.0,
    )

    assert result.success is True
    assert result.data == {"entries": {}, "truncated": False}
    sandbox.client.post.assert_awaited_once()
    call = sandbox.client.post.call_args
    assert call.args[0] == "http://1.2.3.4:8080/api/file/snapshot-workspace"
    assert call.kwargs["json"] == {
        "root": "/home/ubuntu",
        "max_paths": 1000,
        "max_files": 500,
        "max_total_bytes": 1_000_000,
        "max_seconds": 5.0,
    }
