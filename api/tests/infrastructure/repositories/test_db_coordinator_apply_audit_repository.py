from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.repositories.db_coordinator_apply_audit_repository import (
    DbCoordinatorApplyAuditRepository,
)

pytestmark = pytest.mark.anyio


class _Session:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.statements = []
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=self.rowcount)


@pytest.mark.parametrize(("rowcount", "expected"), [(1, True), (0, False)])
async def test_update_terminal_is_first_terminal_wins_cas(
    rowcount: int,
    expected: bool,
) -> None:
    session = _Session(rowcount)
    repo = DbCoordinatorApplyAuditRepository(lambda: session)

    updated = await repo.update_terminal(41, status="success")

    assert updated is expected
    assert len(session.statements) == 1
    compiled = session.statements[0].compile()
    assert 41 in compiled.params.values()
    assert "in_progress" in compiled.params.values()
    session.commit.assert_awaited_once()
