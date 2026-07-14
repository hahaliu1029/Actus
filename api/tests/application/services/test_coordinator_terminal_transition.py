"""Atomic Supervisor terminal authority: locked read, validation, CAS write."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_terminal_transition import (
    CoordinatorTerminalCommand,
    ExpectedCoordinatorLineage,
    terminalize_authoritative_coordinator_child,
)
from app.domain.models.session import Session, SessionStatus


pytestmark = pytest.mark.anyio


def _row(**overrides: Any) -> Session:
    values: dict[str, Any] = {
        "id": "child-1",
        "parent_session_id": "root-1",
        "root_session_id": "root-1",
        "worker_type": "subagent",
        "subagent_control_plane": "mailbox",
        "tool_filter_preset": "coordinator_step",
        "coordinator_run_id": "root-1:step-hash:a2",
        "work_unit_id": "wu-1",
        "status": SessionStatus.RUNNING,
    }
    values.update(overrides)
    return Session(**values)


def _command(**overrides: Any) -> CoordinatorTerminalCommand:
    lineage_values: dict[str, str] = {
        "child_session_id": "child-1",
        "parent_session_id": "root-1",
        "root_session_id": "root-1",
        "coordinator_run_id": "root-1:step-hash:a2",
    }
    lineage_values.update(overrides.pop("lineage", {}))
    values: dict[str, Any] = {
        "lineage": ExpectedCoordinatorLineage(**lineage_values),
        "status": SessionStatus.COMPLETED,
        "reason": "natural",
    }
    values.update(overrides)
    return CoordinatorTerminalCommand(**values)


class _UoW:
    def __init__(self, repo: Any) -> None:
        self.session = repo
        self.commit = AsyncMock()

    async def __aenter__(self) -> "_UoW":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


async def test_locked_authority_read_and_ssm_write_share_one_uow() -> None:
    row = _row()
    repo = MagicMock()
    repo.get_by_id_for_update = AsyncMock(return_value=row)
    uow = _UoW(repo)
    state_machine = MagicMock()
    state_machine.terminate = AsyncMock(return_value=True)

    transitioned = await terminalize_authoritative_coordinator_child(
        _command(), state_machine=state_machine, uow_factory=lambda: uow
    )

    assert transitioned is True
    repo.get_by_id_for_update.assert_awaited_once_with("child-1")
    state_machine.terminate.assert_awaited_once_with(
        "child-1",
        SessionStatus.COMPLETED,
        "natural",
        session_repo=repo,
    )
    uow.commit.assert_awaited_once()


@pytest.mark.parametrize(
    ("row", "lineage"),
    [
        (None, {}),
        (_row(id="different-child"), {}),
        (_row(worker_type="root", parent_session_id=None), {}),
        (_row(parent_session_id="other-root"), {}),
        (_row(subagent_control_plane="legacy"), {}),
        (_row(tool_filter_preset="subagent_research"), {}),
        (_row(root_session_id="other-root"), {}),
        (_row(coordinator_run_id="root-1:step-hash:a1"), {}),
        (_row(), {"parent_session_id": "other-root"}),
        (_row(), {"root_session_id": "other-root"}),
        (_row(work_unit_id=None), {}),
    ],
)
async def test_lineage_mismatch_refuses_transition_without_commit(
    row: Session | None,
    lineage: dict[str, str],
) -> None:
    repo = MagicMock()
    repo.get_by_id_for_update = AsyncMock(return_value=row)
    uow = _UoW(repo)
    state_machine = MagicMock()
    state_machine.terminate = AsyncMock(return_value=True)

    transitioned = await terminalize_authoritative_coordinator_child(
        _command(lineage=lineage),
        state_machine=state_machine,
        uow_factory=lambda: uow,
    )

    assert transitioned is False
    state_machine.terminate.assert_not_awaited()
    uow.commit.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("subagent_control_plane", "legacy"),
        ("tool_filter_preset", "subagent_research"),
        ("coordinator_run_id", "root-1:step-hash:a3"),
    ],
)
async def test_lock_time_row_flip_is_revalidated_before_cas(
    field: str,
    replacement: str,
) -> None:
    """The helper trusts only the row returned by the blocking lock read."""
    row = _row()
    entered_lock_read = asyncio.Event()
    release_lock_read = asyncio.Event()

    class _BlockingRepo:
        async def get_by_id_for_update(self, session_id: str) -> Session:
            assert session_id == "child-1"
            entered_lock_read.set()
            await release_lock_read.wait()
            return row

    repo = _BlockingRepo()
    uow = _UoW(repo)
    state_machine = MagicMock()
    state_machine.terminate = AsyncMock(return_value=True)
    task = asyncio.create_task(
        terminalize_authoritative_coordinator_child(
            _command(), state_machine=state_machine, uow_factory=lambda: uow
        )
    )

    await asyncio.wait_for(entered_lock_read.wait(), timeout=0.1)
    setattr(row, field, replacement)
    release_lock_read.set()

    assert await asyncio.wait_for(task, timeout=0.1) is False
    state_machine.terminate.assert_not_awaited()
    uow.commit.assert_not_awaited()


async def test_locked_read_failure_propagates_without_cas_or_commit() -> None:
    repo = MagicMock()
    repo.get_by_id_for_update = AsyncMock(side_effect=RuntimeError("lock failed"))
    uow = _UoW(repo)
    state_machine = MagicMock()
    state_machine.terminate = AsyncMock(return_value=True)

    with pytest.raises(RuntimeError, match="lock failed"):
        await terminalize_authoritative_coordinator_child(
            _command(), state_machine=state_machine, uow_factory=lambda: uow
        )

    state_machine.terminate.assert_not_awaited()
    uow.commit.assert_not_awaited()


async def test_redelivery_preserves_ssm_cas_idempotence() -> None:
    repo = MagicMock()
    repo.get_by_id_for_update = AsyncMock(return_value=_row())
    uows = [_UoW(repo), _UoW(repo)]
    state_machine = MagicMock()
    state_machine.terminate = AsyncMock(side_effect=[True, False])

    first = await terminalize_authoritative_coordinator_child(
        _command(), state_machine=state_machine, uow_factory=lambda: uows[0]
    )
    second = await terminalize_authoritative_coordinator_child(
        _command(), state_machine=state_machine, uow_factory=lambda: uows[1]
    )

    assert (first, second) == (True, False)
    assert state_machine.terminate.await_count == 2
    for uow in uows:
        uow.commit.assert_awaited_once()
