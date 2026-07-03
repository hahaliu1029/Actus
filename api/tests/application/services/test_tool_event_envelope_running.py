"""B1-1b: RUNNING → "running" skeleton projection (spec §4.2, INV-B1-6)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.application.services.tool_event_envelope_v1 import (
    project_tool_event_to_envelope_v1,
)
from app.domain.models.event import ToolEvent, ToolEventStatus


def _running_event() -> ToolEvent:
    return ToolEvent(
        tool_call_id="t1",
        tool_name="file",
        function_name="file_write",
        function_args={"path": "/x"},
        status=ToolEventStatus.RUNNING,
    )


def test_running_projects_to_skeleton_with_running_status():
    envelope = project_tool_event_to_envelope_v1(_running_event())
    assert envelope.status == "running"
    assert envelope.function_result is None  # skeleton — 同 CALLING，无 outcome
    assert envelope.envelope_version == 1
    assert envelope.tool_call_id == "t1"


def test_calling_and_called_projection_unchanged():
    calling = _running_event().model_copy(update={"status": ToolEventStatus.CALLING})
    assert project_tool_event_to_envelope_v1(calling).status == "calling"


def test_wire_schema_accepts_running_literal():
    from app.interfaces.schemas.event import ToolEventEnvelopeV1

    envelope = project_tool_event_to_envelope_v1(_running_event())
    wire = ToolEventEnvelopeV1.model_validate(
        envelope.model_dump(by_alias=True)
    )
    assert wire.status == "running"


@pytest.mark.anyio
async def test_runner_handle_tool_event_passes_running_through():
    """_handle_tool_event 只 enrich CALLED；RUNNING 必须原样直通（无分支炸裂）。"""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    evt = _running_event()
    stub_self = MagicMock()
    await AgentTaskRunner._handle_tool_event(stub_self, evt)
    assert evt.tool_content is None, "RUNNING must not be enriched"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
