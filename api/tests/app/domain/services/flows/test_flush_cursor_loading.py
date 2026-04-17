"""Tests for Task 5: _flush_cursor and _pending_flush_batch attributes + cursor loading in invoke().

Verifies:
1. Default init values (0 and None)
2. Cursor loading from DB happens BEFORE aget_state
3. DB error fallback (cursor defaults to 0)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.models.memory import Memory
from app.domain.models.message import Message

from tests.conftest import TEST_USER_ID_FIXED


def _make_uow_factory(session_repo):
    """Create a proper async context manager uow_factory."""
    @asynccontextmanager
    async def uow_factory():
        uow = MagicMock()
        uow.session = session_repo
        yield uow

    return uow_factory


def _make_flow(**overrides):
    """Create a PlannerReActFlow with all required mocks."""
    from app.domain.services.flows.planner_react import PlannerReActFlow

    kwargs = {
        "llm": MagicMock(),
        "agent_config": AgentConfig(),
        "session_id": "test-session",
        "user_id": TEST_USER_ID_FIXED,
        "uow_factory": MagicMock(),
        "browser": MagicMock(),
        "sandbox": MagicMock(),
        "search_engine": MagicMock(),
        "mcp_tool": MagicMock(),
        "a2a_tool": MagicMock(),
        "skill_tool": MagicMock(),
    }
    kwargs.update(overrides)
    return PlannerReActFlow(**kwargs)


class TestFlushCursorDefaultInit:
    """Task 5a: _flush_cursor and _pending_flush_batch default values."""

    def test_flush_cursor_default_zero(self) -> None:
        """_flush_cursor 初始值为 0。"""
        flow = _make_flow()
        assert flow._flush_cursor == 0

    def test_pending_flush_batch_default_none(self) -> None:
        """_pending_flush_batch 初始值为 None。"""
        flow = _make_flow()
        assert flow._pending_flush_batch is None


class TestFlushCursorLoadingInInvoke:
    """Task 5b: Cursor loading from DB happens BEFORE aget_state."""

    @pytest.mark.anyio
    async def test_invoke_loads_cursor_before_resume_check(self) -> None:
        """invoke() 在调用 aget_state 之前从 DB 加载 flush_cursor=42。"""
        # Arrange: mock UoW returning Memory(flush_cursor=42)
        mock_session_repo = AsyncMock()
        mock_session_repo.get_memory = AsyncMock(
            return_value=Memory(flush_cursor=42)
        )
        mock_session_repo.get_summary = AsyncMock(return_value=[])
        mock_session_repo.save_memory = AsyncMock()

        uow_factory = _make_uow_factory(mock_session_repo)
        flow = _make_flow(uow_factory=uow_factory)

        # Capture _flush_cursor value at the time aget_state is called
        captured_cursor_at_aget_state = None

        async def fake_aget_state(config):
            nonlocal captured_cursor_at_aget_state
            captured_cursor_at_aget_state = flow._flush_cursor
            # Return a state with no pending interrupts (not a resume)
            state = MagicMock()
            state.next = None
            return state

        # Stub _try_drive_skill_graph to return None (not a skill continuation)
        flow._try_drive_skill_graph = AsyncMock(return_value=None)
        # Stub _ensure_graphs to be a no-op but set _main_graph
        mock_main_graph = MagicMock()
        mock_main_graph.aget_state = fake_aget_state

        async def fake_ensure_graphs():
            flow._main_graph = mock_main_graph
            flow._graphs_built = True

        flow._ensure_graphs = fake_ensure_graphs

        # Stub the bridge to not actually run a graph
        with patch(
            "app.domain.services.flows.planner_react.GraphEventBridge"
        ) as MockBridge:
            mock_bridge_instance = MagicMock()
            mock_bridge_instance.final_state = {
                "plan": None,
                "messages": [],
                "should_interrupt": False,
            }

            async def fake_bridge_run(*args, **kwargs):
                return
                yield  # make it an async generator

            mock_bridge_instance.run = fake_bridge_run
            MockBridge.return_value = mock_bridge_instance

            # Act: consume the async generator
            message = Message(message="test input")
            events = []
            async for event in flow.invoke(message):
                events.append(event)

        # Assert: cursor was loaded to 42 BEFORE aget_state was called
        assert captured_cursor_at_aget_state == 42
        assert flow._flush_cursor == 42

    @pytest.mark.anyio
    async def test_invoke_cursor_db_error_defaults_to_zero(self) -> None:
        """DB 读取 flush_cursor 失败时回退到 0。"""
        # Arrange: mock UoW where get_memory raises on the cursor-loading call
        call_count = 0

        async def get_memory_side_effect(session_id, agent_name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is for cursor loading — simulate DB error
                raise RuntimeError("DB connection lost")
            # Subsequent calls return normal memory
            return Memory(messages=[], flush_cursor=99)

        mock_session_repo = AsyncMock()
        mock_session_repo.get_memory = AsyncMock(side_effect=get_memory_side_effect)
        mock_session_repo.get_summary = AsyncMock(return_value=[])
        mock_session_repo.save_memory = AsyncMock()

        uow_factory = _make_uow_factory(mock_session_repo)
        flow = _make_flow(uow_factory=uow_factory)

        # Capture _flush_cursor value at the time aget_state is called
        captured_cursor_at_aget_state = None

        async def fake_aget_state(config):
            nonlocal captured_cursor_at_aget_state
            captured_cursor_at_aget_state = flow._flush_cursor
            state = MagicMock()
            state.next = None
            return state

        flow._try_drive_skill_graph = AsyncMock(return_value=None)
        mock_main_graph = MagicMock()
        mock_main_graph.aget_state = fake_aget_state

        async def fake_ensure_graphs():
            flow._main_graph = mock_main_graph
            flow._graphs_built = True

        flow._ensure_graphs = fake_ensure_graphs

        with patch(
            "app.domain.services.flows.planner_react.GraphEventBridge"
        ) as MockBridge:
            mock_bridge_instance = MagicMock()
            mock_bridge_instance.final_state = {
                "plan": None,
                "messages": [],
                "should_interrupt": False,
            }

            async def fake_bridge_run(*args, **kwargs):
                return
                yield

            mock_bridge_instance.run = fake_bridge_run
            MockBridge.return_value = mock_bridge_instance

            message = Message(message="test input")
            async for _ in flow.invoke(message):
                pass

        # Assert: cursor defaulted to 0 despite DB error
        assert captured_cursor_at_aget_state == 0
        assert flow._flush_cursor == 0
