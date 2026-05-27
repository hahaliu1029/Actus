"""ORM model for ``coordinator_result_envelope_store`` (C2 PR-7 §13.x).

Schema mirrors ``api/alembic/versions/c2pr7_add_coordinator_result_envelope_store``.
See the migration docstring for column semantics, the partial unique
``ux_result_store_run_wu_terminal`` rationale, and the
minimum-rehydrate-fields contract on ``payload``.

The model is write-mostly on the producer side (mailbox supervisor /
coordinator orchestrator persist a terminal envelope when a child
reaches RESULT_READY or CANCEL_ACK) and read-mostly on the consumer
side (PR-7 ``CoordinatorRehydrateService`` reads back rows by
``coordinator_run_id`` during crash recovery). The repo at
``infrastructure/repositories/db_coordinator_result_envelope_store_repository``
encapsulates the short-session insert/select pattern.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import TIMESTAMP, BigInteger, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.models.base import Base


class CoordinatorResultEnvelopeStore(Base):
    """One row per terminal envelope per (coordinator_run_id, work_unit_id).

    The unique index on (coordinator_run_id, work_unit_id) enforces
    at-most-once persistence — repeated inserts with the same key raise
    IntegrityError, which the repo's caller catches as the idempotent
    no-op signal.

    ``payload`` carries the JSONB minimum-rehydrate subset (see repo
    ``_MIN_REHYDRATE_KEYS``); the application layer is responsible for
    filtering down to that subset before handing it to the repo.
    """

    __tablename__ = "coordinator_result_envelope_store"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    coordinator_run_id: Mapped[str] = mapped_column(String(320))
    work_unit_id: Mapped[str] = mapped_column(String(64))
    child_session_id: Mapped[str] = mapped_column(String(255))
    # Allowed values: 'RESULT_READY' | 'CANCEL_ACK'. Caller validates;
    # no CHECK constraint at the DB layer to keep schema migrations
    # cheap if a third terminal kind is ever added.
    envelope_type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    received_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("NOW()"),
    )
