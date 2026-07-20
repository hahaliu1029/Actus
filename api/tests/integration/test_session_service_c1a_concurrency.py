"""C1a real-PG concurrency: FOR UPDATE + descendants cap serializes spawn.

Seed root to MAX_DESCENDANTS_PER_ROOT - 1 children, then fire two
create_session_with_parent calls concurrently. Exactly one must succeed; the
other must raise SpawnCapExceeded('descendants').
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from app.application.services.session_service import SessionService
from app.domain.services.subagent_limits import (
    MAX_DESCENDANTS_PER_ROOT,
    SpawnCapExceeded,
)

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
async def seeded_root(uow_factory):
    """Insert root + (MAX_DESCENDANTS_PER_ROOT - 1) subagent children.

    The user is committed through the same independent UoW family as the
    concurrent spawn calls, so every connection can resolve its FK.
    """
    from app.domain.models.session import Session
    from app.infrastructure.models.user import UserModel

    user_id = str(uuid.uuid4())
    root_id = uuid.uuid4().hex
    async with uow_factory() as uow:
        uow.db_session.add(
            UserModel(
                id=user_id,
                username=f"c1a_{user_id[:8]}",
                password_hash="x",
            )
        )
        await uow.db_session.flush()
        await uow.session.save(
            Session(id=root_id, user_id=user_id, worker_type="root", title="root")
        )
    svc = SessionService(uow_factory=uow_factory)
    for _ in range(MAX_DESCENDANTS_PER_ROOT - 1):
        await svc.create_session_with_parent(
            user_id=user_id,
            parent_session_id=root_id,
            tool_filter_preset="subagent_research",
        )
    return root_id, user_id


async def test_concurrent_spawn_at_cap_minus_one_serializes(seeded_root, uow_factory):
    svc = SessionService(uow_factory=uow_factory)
    root_id, user_id = seeded_root

    async def attempt():
        try:
            await svc.create_session_with_parent(
                user_id=user_id,
                parent_session_id=root_id,
                tool_filter_preset="subagent_research",
            )
            return "ok"
        except SpawnCapExceeded as exc:
            return f"capped:{exc.kind}"

    results = await asyncio.gather(attempt(), attempt(), return_exceptions=True)
    succeeded = [r for r in results if r == "ok"]
    capped = [r for r in results if isinstance(r, str) and r.startswith("capped:")]
    assert len(succeeded) == 1, results
    assert len(capped) == 1, results
    assert capped[0] == "capped:descendants"
