import pytest

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.flows.planner_react import PlannerReActFlow


class _FakeCheckpointer:
    """Minimal stand-in for testing close() lifecycle."""
    pass


def _make_flow(checkpointer=None):
    """Create a PlannerReActFlow with minimal required dependencies stubbed out."""
    from unittest.mock import MagicMock

    return PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=MagicMock(),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
        checkpointer=checkpointer,
    )


@pytest.mark.anyio
async def test_close_sets_checkpointer_to_none():
    """close() should clear the _checkpointer reference."""
    flow = _make_flow(checkpointer=_FakeCheckpointer())
    assert flow._checkpointer is not None

    await flow.close()
    assert flow._checkpointer is None


@pytest.mark.anyio
async def test_close_is_idempotent():
    """Calling close() multiple times should not raise."""
    flow = _make_flow(checkpointer=_FakeCheckpointer())

    await flow.close()
    await flow.close()  # second call should not raise
    assert flow._checkpointer is None


@pytest.mark.anyio
async def test_close_on_flow_without_checkpointer():
    """close() on a flow that was never used (no checkpointer) should not raise."""
    flow = _make_flow(checkpointer=None)
    assert flow._checkpointer is None

    await flow.close()  # should not raise
    assert flow._checkpointer is None
