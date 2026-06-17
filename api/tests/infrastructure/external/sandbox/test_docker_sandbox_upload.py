import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

# api suite runs async tests via anyio (NOT pytest-asyncio); the session-scoped
# `anyio_backend` fixture comes from api/tests/conftest.py:25. Bare `async def`
# under this module marker — matches the 387-file api convention.
pytestmark = pytest.mark.anyio


def _resp():
    # Mirror the live sandbox HTTP envelope consumed by
    # ``ToolResult.from_sandbox(code, msg, data, **kwargs)`` — the real wire
    # shape is ``{"code", "msg", "data"}`` (NOT success/message). The kwarg
    # under test is the multipart ``data`` dict on the POST, so the precise
    # response body is incidental; it just has to deserialize cleanly.
    r = MagicMock()
    r.json.return_value = {"code": 200, "msg": "ok", "data": {}}
    return r


async def test_upload_file_puts_refuse_special_in_multipart_form(monkeypatch):
    sb = DockerSandbox.__new__(DockerSandbox)  # bypass __init__/container
    sb._base_url = "http://sbx"
    sb.client = MagicMock()
    sb.client.post = AsyncMock(return_value=_resp())

    await sb.upload_file(io.BytesIO(b"x"), "workspace/p", refuse_special=True)
    _, kwargs = sb.client.post.call_args
    assert kwargs["data"]["refuse_special"] == "true"

    await sb.upload_file(io.BytesIO(b"y"), "workspace/p2")  # default
    _, kwargs2 = sb.client.post.call_args
    assert kwargs2["data"]["refuse_special"] == "false"
