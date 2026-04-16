from unittest.mock import AsyncMock

import pytest

from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.tool_result import ToolResult
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


async def test_handle_tool_event_syncs_file_view_artifact_to_storage() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._sandbox = type(
        "FakeSandbox",
        (),
        {
            "read_file": AsyncMock(
                return_value=ToolResult(
                    success=True,
                    data={
                        "filepath": "/home/ubuntu/final-report.pdf",
                        "content": "[二进制文件] final-report.pdf (1.0 MB)",
                    },
                )
            )
        },
    )()
    runner._sync_file_to_storage = AsyncMock()

    event = ToolEvent(
        tool_call_id="call-1",
        tool_name="file",
        function_name="file_view",
        function_args={"filepath": "/home/ubuntu/final-report.pdf"},
        status=ToolEventStatus.CALLED,
    )

    await runner._handle_tool_event(event)

    runner._sandbox.read_file.assert_awaited_once_with("/home/ubuntu/final-report.pdf")
    runner._sync_file_to_storage.assert_awaited_once_with(
        "/home/ubuntu/final-report.pdf"
    )
