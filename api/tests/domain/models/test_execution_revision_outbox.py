from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB

from app.domain.models.event import ExecutionStatePayload, PendingExecutionEvent
from app.domain.models.session import Session
from app.infrastructure.models.session import SessionModel
from app.interfaces.schemas.session import SupervisorSnapshot

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "alembic/versions/task8_add_execution_revision_outbox.py"
)
_SPEC = importlib.util.spec_from_file_location("task8_execution_revision_migration", _MIGRATION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def _pending(revision: int = 1) -> PendingExecutionEvent:
    return PendingExecutionEvent(
        payload=ExecutionStatePayload(
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            execution_revision=revision,
        )
    )


def test_pending_execution_event_requires_positive_revision() -> None:
    with pytest.raises(ValidationError, match="execution_revision >= 1"):
        _pending(0)


def test_session_execution_outbox_round_trips_through_orm_jsonb() -> None:
    session = Session(
        id="session-1",
        execution_revision=7,
        pending_execution_event=_pending(7),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    model = SessionModel.from_domain(session)
    assert model.execution_revision == 7
    assert model.pending_execution_event == session.pending_execution_event.model_dump(
        mode="json"
    )
    # SQLAlchemy fills these server-managed columns on INSERT/flush.  This is a
    # DB-free round-trip test, so provide the values the ORM would receive.
    model.created_at = session.created_at
    model.updated_at = session.updated_at

    restored = model.to_domain()
    assert restored.execution_revision == 7
    assert restored.pending_execution_event == session.pending_execution_event


def test_orm_and_snapshot_expose_stable_revision_contract() -> None:
    revision_column = SessionModel.__table__.c.execution_revision
    pending_column = SessionModel.__table__.c.pending_execution_event
    assert isinstance(revision_column.type, BigInteger)
    assert revision_column.nullable is False
    assert str(revision_column.server_default.arg) == "0"
    assert isinstance(pending_column.type, JSONB)
    assert pending_column.nullable is True

    snapshot = SupervisorSnapshot(
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        is_alive=True,
        execution_revision=9,
    )
    assert snapshot.execution_revision == 9


def test_task8_migration_is_single_head_extension_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_column = MagicMock()
    drop_column = MagicMock()
    monkeypatch.setattr(migration.op, "add_column", add_column)
    monkeypatch.setattr(migration.op, "drop_column", drop_column)

    assert migration.down_revision == "d1a_add_extension_registry"
    migration.upgrade()
    assert [call.args[1].name for call in add_column.call_args_list] == [
        "execution_revision",
        "pending_execution_event",
    ]

    migration.downgrade()
    assert [call.args for call in drop_column.call_args_list] == [
        ("sessions", "pending_execution_event"),
        ("sessions", "execution_revision"),
    ]
