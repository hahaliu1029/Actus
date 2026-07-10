"""Factory persisters translate deleted-parent FK violations into
``SessionRowGoneError`` so the handler can skip instead of marker-chasing.

Shape locked against real asyncpg-under-SQLAlchemy behavior (see
tests/integration/test_cost_persist_after_session_delete.py for the live
repro): ``sqlalchemy.exc.IntegrityError`` whose ``orig`` chain carries
``sqlstate == "23503"`` and the violated constraint name — either as a
``constraint_name`` attribute or embedded in the message text.

Only the two cost_records *parent-row* FKs translate (session / user —
the handler's session_id/user_id are fixed for its lifetime, so either
parent going away is the same "row can never land again" class). Any
other integrity error must keep the existing failure-count + degraded
marker semantics untouched.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from sqlalchemy.exc import IntegrityError

from app.application.services.cost_callback_factory import (
    build_cost_callback_handler,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _llm_result() -> LLMResult:
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="hi",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    )
                )
            ]
        ]
    )


async def _drive_one_llm_call(handler) -> None:
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(_llm_result(), run_id=run_id)
    await handler.flush_pending()


class _PgErrorWithDiag(Exception):
    """asyncpg-shaped error: carries sqlstate + constraint_name attributes."""

    def __init__(self, message: str, *, sqlstate: str, constraint_name: str | None):
        super().__init__(message)
        self.sqlstate = sqlstate
        self.constraint_name = constraint_name


class _PgErrorMessageOnly(Exception):
    """Driver-wrapper-shaped error: sqlstate only, constraint just in text."""

    def __init__(self, message: str, *, sqlstate: str):
        super().__init__(message)
        self.sqlstate = sqlstate


def _integrity_error(orig: Exception) -> IntegrityError:
    return IntegrityError("INSERT INTO cost_records ...", {}, orig)


class _RaisingDbSession:
    """Raises the given exception on every execute; counts attempts."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.execute_calls: int = 0

    async def execute(self, stmt: Any) -> Any:
        self.execute_calls += 1
        raise self._exc

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _FakeUoW:
    def __init__(self, db_session: _RaisingDbSession) -> None:
        self.db_session = db_session

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Mimic DBUnitOfWork: rollback on error, swallow its own failures.
        try:
            if exc_type:
                await self.db_session.rollback()
        except Exception:
            pass
        await self.db_session.close()


def _handler_with_failing_db(exc: Exception):
    db = _RaisingDbSession(exc)
    handler = build_cost_callback_handler(
        session_id="sess-x", user_id="u", uow_factory=lambda: _FakeUoW(db)
    )
    return handler, db


async def test_session_fk_violation_with_diag_attrs_translates() -> None:
    """23503 + constraint_name attribute → session-gone skip (no marker)."""
    orig = _PgErrorWithDiag(
        'insert or update on table "cost_records" violates foreign key '
        'constraint "fk_cost_records_session_id_sessions"',
        sqlstate="23503",
        constraint_name="fk_cost_records_session_id_sessions",
    )
    handler, db = _handler_with_failing_db(_integrity_error(orig))

    await _drive_one_llm_call(handler)

    assert db.execute_calls == 1, "marker retry must not hit the dead FK"
    assert handler.persist_failure_count == 0


async def test_session_fk_violation_message_only_translates() -> None:
    """23503 with the constraint name only in the message text still counts."""
    orig = _PgErrorMessageOnly(
        'insert or update on table "cost_records" violates foreign key '
        'constraint "fk_cost_records_session_id_sessions"',
        sqlstate="23503",
    )
    handler, db = _handler_with_failing_db(_integrity_error(orig))

    await _drive_one_llm_call(handler)

    assert db.execute_calls == 1
    assert handler.persist_failure_count == 0


async def test_user_fk_violation_translates_too() -> None:
    """User row gone → this handler's user_id can never resolve again."""
    orig = _PgErrorWithDiag(
        'insert or update on table "cost_records" violates foreign key '
        'constraint "fk_cost_records_user_id_users"',
        sqlstate="23503",
        constraint_name="fk_cost_records_user_id_users",
    )
    handler, db = _handler_with_failing_db(_integrity_error(orig))

    await _drive_one_llm_call(handler)

    assert db.execute_calls == 1
    assert handler.persist_failure_count == 0


async def test_unique_violation_keeps_failure_semantics() -> None:
    """23505 (or any non-parent-FK error) → existing failure + marker path."""
    orig = _PgErrorWithDiag(
        'duplicate key value violates unique constraint "uq_cost_records_run_id"',
        sqlstate="23505",
        constraint_name="uq_cost_records_run_id",
    )
    handler, db = _handler_with_failing_db(_integrity_error(orig))

    await _drive_one_llm_call(handler)

    # Primary attempt + degraded-marker attempt — untouched behavior.
    assert db.execute_calls == 2
    assert handler.persist_failure_count == 1


async def test_foreign_23503_on_other_constraint_not_translated() -> None:
    """A 23503 against some future non-parent FK must not be swallowed."""
    orig = _PgErrorWithDiag(
        'insert or update on table "cost_records" violates foreign key '
        'constraint "fk_cost_records_something_else"',
        sqlstate="23503",
        constraint_name="fk_cost_records_something_else",
    )
    handler, db = _handler_with_failing_db(_integrity_error(orig))

    await _drive_one_llm_call(handler)

    assert db.execute_calls == 2
    assert handler.persist_failure_count == 1


async def test_supervisor_aware_builder_translates_as_well() -> None:
    """Both factory builders share the same persister semantics."""
    from app.application.services.cost_callback_factory import (
        build_supervisor_aware_callback_handler,
    )

    class _NoopSupervisor:
        async def inflight_inc(self, **kwargs) -> None:
            return None

        async def inflight_dec(self, **kwargs) -> None:
            return None

    orig = _PgErrorWithDiag(
        'insert or update on table "cost_records" violates foreign key '
        'constraint "fk_cost_records_session_id_sessions"',
        sqlstate="23503",
        constraint_name="fk_cost_records_session_id_sessions",
    )
    db = _RaisingDbSession(_integrity_error(orig))
    handler = build_supervisor_aware_callback_handler(
        _NoopSupervisor(),
        session_id="sess-x",
        user_id="u",
        uow_factory=lambda: _FakeUoW(db),
    )

    await _drive_one_llm_call(handler)

    assert db.execute_calls == 1
    assert handler.persist_failure_count == 0
