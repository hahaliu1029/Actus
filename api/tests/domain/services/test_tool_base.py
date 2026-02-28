import pytest

from app.domain.services.tools.base import BaseTool


class _DummyTool(BaseTool):
    name = "dummy"


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_invoke_raises_when_tool_not_found() -> None:
    tool = _DummyTool()
    with pytest.raises(ValueError, match="未找到"):
        await tool.invoke("not_exists")
