"""Atomic authority boundary for coordinator child terminal transitions.

The terminal envelope is untrusted input. Authority comes only from the child
session row read under ``FOR UPDATE`` and kept locked through the terminal CAS
and commit in the same unit of work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from app.domain.models.session import Session, SessionStatus
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.session.session_state_machine import SessionStateMachine


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExpectedCoordinatorLineage:
    """Lineage asserted by the envelope and owning Supervisor context."""

    child_session_id: str
    parent_session_id: str
    root_session_id: str
    coordinator_run_id: str


@dataclass(frozen=True, slots=True)
class CoordinatorTerminalCommand:
    """Typed terminal request passed across the Supervisor composition port."""

    lineage: ExpectedCoordinatorLineage
    status: SessionStatus
    reason: str


def _matches_locked_authority(
    row: Session | None,
    expected: ExpectedCoordinatorLineage,
) -> bool:
    """Validate every persisted ownership axis after acquiring the row lock."""
    return (
        row is not None
        and row.id == expected.child_session_id
        and row.worker_type == "subagent"
        and row.subagent_control_plane == "mailbox"
        and row.tool_filter_preset == COORDINATOR_STEP_PRESET
        and row.parent_session_id == expected.parent_session_id
        and expected.parent_session_id == expected.root_session_id
        and row.root_session_id == expected.root_session_id
        and bool(row.coordinator_run_id)
        and row.coordinator_run_id == expected.coordinator_run_id
        and bool(row.work_unit_id)
    )


async def terminalize_authoritative_coordinator_child(
    command: CoordinatorTerminalCommand,
    *,
    state_machine: SessionStateMachine,
    uow_factory: Callable[[], IUnitOfWork],
) -> bool:
    """Lock, revalidate and terminalize one coordinator child atomically.

    A coordinator run id embeds its attempt index, so exact equality is the
    cross-attempt fence. Missing/mismatched authority is a safe no-op; lock,
    CAS and commit failures propagate so the mailbox envelope stays in PEL.
    """
    expected = command.lineage
    async with uow_factory() as uow:
        row = await uow.session.get_by_id_for_update(expected.child_session_id)
        if not _matches_locked_authority(row, expected):
            logger.warning(
                "coordinator terminal DB write refused child=%s "
                "expected_parent=%s expected_root=%s expected_run=%s "
                "row_present=%s row_id=%s row_parent=%s row_root=%s "
                "worker_type=%s control_plane=%s preset=%s row_run=%s "
                "work_unit_id=%s",
                expected.child_session_id,
                expected.parent_session_id,
                expected.root_session_id,
                expected.coordinator_run_id,
                row is not None,
                getattr(row, "id", None),
                getattr(row, "parent_session_id", None),
                getattr(row, "root_session_id", None),
                getattr(row, "worker_type", None),
                getattr(row, "subagent_control_plane", None),
                getattr(row, "tool_filter_preset", None),
                getattr(row, "coordinator_run_id", None),
                getattr(row, "work_unit_id", None),
            )
            return False

        transitioned = await state_machine.terminate(
            expected.child_session_id,
            command.status,
            command.reason,
            session_repo=uow.session,
        )
        # Explicit commit surfaces commit errors before the context manager's
        # defensive best-effort exit path can absorb them.
        await uow.commit()
        return transitioned
