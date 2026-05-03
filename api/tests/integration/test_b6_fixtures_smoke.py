"""B6 fixture smoke tests — verify all Task-0 fixtures resolve without errors.

Requires a live PostgreSQL (manus_test).  In CI this runs automatically.
Locally: see CLAUDE.md §"Local Test Infrastructure" for setup.

Run: cd api && uv run pytest -m integration tests/integration/test_b6_fixtures_smoke.py -v
"""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_uow_factory_exists(uow_factory):
    """uow_factory() returns a context-manager whose db_session is not None."""
    async with uow_factory() as uow:
        assert uow.db_session is not None


@pytest.mark.anyio
async def test_seed_session_has_user_id(seed_session):
    """seed_session has both .id and .user_id populated."""
    assert seed_session.id
    assert seed_session.user_id


@pytest.mark.anyio
async def test_seed_other_user_session_is_different_user(seed_session, seed_other_user_session):
    """The two seed fixtures belong to *different* users."""
    assert seed_other_user_session.user_id != seed_session.user_id


def test_messages_at_85_percent_is_nonempty(messages_at_85_percent):
    """Fixture returns a non-empty list of BaseMessage objects."""
    from langchain_core.messages import BaseMessage

    assert len(messages_at_85_percent) > 0
    assert all(isinstance(m, BaseMessage) for m in messages_at_85_percent)


def test_memory_at_85_percent_has_messages(memory_at_85_percent):
    """memory_at_85_percent.messages is a non-empty list of dicts."""
    assert isinstance(memory_at_85_percent.messages, list)
    assert len(memory_at_85_percent.messages) > 0


@pytest.mark.anyio
async def test_fake_summary_llm_returns_ai_message(fake_summary_llm):
    """fake_summary_llm.ainvoke returns an AIMessage with non-empty content."""
    from langchain_core.messages import AIMessage

    result = await fake_summary_llm.ainvoke("any prompt")
    assert isinstance(result, AIMessage)
    assert result.content


@pytest.mark.anyio
async def test_planner_react_with_compactor_fields(planner_react_with_compactor, seed_session):
    """planner_react_with_compactor exposes the fields B6 tests will use."""
    flow = planner_react_with_compactor
    assert flow._compactor is not None
    assert flow._summary_llm is not None
    assert flow._uow_factory is not None
    assert flow._session_id == seed_session.id
    assert flow._overflow_config is not None
    assert flow._overflow_config.context_window == 80_000
