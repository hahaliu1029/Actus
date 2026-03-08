from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.skill_creation_state import SkillCreationState
from app.infrastructure.repositories.db_session_repository import DBSessionRepository

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def db_session() -> AsyncMock:
    return AsyncMock()


async def test_get_skill_creation_state_returns_model(db_session: AsyncMock) -> None:
    db_session.execute.return_value = SimpleNamespace(
        scalar_one_or_none=lambda: {
            "pending_action": "generate",
            "approval_status": "pending",
            "last_tool_name": "brainstorm_skill",
            "last_tool_call_id": "call_123",
            "saved_tool_result_json": '{"success": true}',
            "blueprint": {"skill_name": "meeting-audio-analyzer"},
            "blueprint_json": '{"skill_name":"meeting-audio-analyzer"}',
            "skill_data": "",
        }
    )
    repo = DBSessionRepository(db_session)

    result = await repo.get_skill_creation_state("s1")

    assert result is not None
    assert result.pending_action == "generate"
    assert result.last_tool_name == "brainstorm_skill"


async def test_get_skill_creation_state_returns_none_when_missing(
    db_session: AsyncMock,
) -> None:
    db_session.execute.return_value = SimpleNamespace(scalar_one_or_none=lambda: None)
    repo = DBSessionRepository(db_session)

    result = await repo.get_skill_creation_state("s1")

    assert result is None


async def test_save_and_clear_skill_creation_state(db_session: AsyncMock) -> None:
    db_session.execute.side_effect = [
        MagicMock(rowcount=1),
        MagicMock(rowcount=1),
    ]
    repo = DBSessionRepository(db_session)
    state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
        last_tool_name="generate_skill",
    )

    await repo.save_skill_creation_state("s1", state)
    await repo.clear_skill_creation_state("s1")

    assert db_session.execute.await_count == 2

