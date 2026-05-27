"""Integration test for SessionRepository.find_children_by_coordinator_run (C2 PR-7 §12.3).

Real PG. Seeds (parent + 3 children: 2 sharing a coordinator_run_id and 1
with a different run/parent) and asserts the query is correctly scoped.
"""
from __future__ import annotations

import uuid

import pytest

from app.infrastructure.repositories.db_session_repository import DBSessionRepository

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
async def seeded_coordinator_children(db_session, sample_user):
    """Insert 1 parent + 3 children directly via ORM.

    Returns dict::

        {"parent_id": ..., "run_id": ..., "wu_ids": ["wu1", "wu2"],
         "other_parent_id": ..., "other_run_id": ...}

    Layout:
      parent_a/coord_run=r1 → c1 (wu1), c2 (wu2)
      parent_a/coord_run=r2 → c3 (wu_other)   ← different run, same parent
      parent_b/coord_run=r1 → c4 (wu1)        ← same run, different parent
    """
    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    parent_a = f"sess-pr7-parA-{uuid.uuid4().hex[:8]}"
    parent_b = f"sess-pr7-parB-{uuid.uuid4().hex[:8]}"
    run_id_a = f"{parent_a}:abcd1234abcd1234:a1"
    run_id_b = f"{parent_b}:abcd1234abcd1234:a1"
    common = dict(user_id=sample_user.id, status=SessionStatus.RUNNING.value)

    db_session.add(SessionModel(id=parent_a, worker_type="root", title="par_a", **common))
    db_session.add(SessionModel(id=parent_b, worker_type="root", title="par_b", **common))
    await db_session.flush()

    db_session.add(SessionModel(
        id=f"c-pr7-{uuid.uuid4().hex[:8]}-1",
        worker_type="subagent",
        parent_session_id=parent_a,
        coordinator_run_id=run_id_a,
        work_unit_id="abcd1234abcd1234.a1.0",
        tool_filter_preset="coordinator_step",
        title="c1",
        **common,
    ))
    db_session.add(SessionModel(
        id=f"c-pr7-{uuid.uuid4().hex[:8]}-2",
        worker_type="subagent",
        parent_session_id=parent_a,
        coordinator_run_id=run_id_a,
        work_unit_id="abcd1234abcd1234.a1.1",
        tool_filter_preset="coordinator_step",
        title="c2",
        **common,
    ))
    db_session.add(SessionModel(
        id=f"c-pr7-{uuid.uuid4().hex[:8]}-3",
        worker_type="subagent",
        parent_session_id=parent_a,
        coordinator_run_id=f"{parent_a}:wxyz5678wxyz5678:a1",
        work_unit_id="wxyz5678wxyz5678.a1.0",
        tool_filter_preset="coordinator_step",
        title="c3-other-run",
        **common,
    ))
    db_session.add(SessionModel(
        id=f"c-pr7-{uuid.uuid4().hex[:8]}-4",
        worker_type="subagent",
        parent_session_id=parent_b,
        coordinator_run_id=run_id_a,
        work_unit_id="abcd1234abcd1234.a1.0",
        tool_filter_preset="coordinator_step",
        title="c4-other-parent",
        **common,
    ))
    await db_session.flush()

    return {
        "parent_a": parent_a,
        "parent_b": parent_b,
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
    }


async def test_find_children_returns_only_matching_run_and_parent(
    db_session, seeded_coordinator_children,
):
    repo = DBSessionRepository(db_session)
    children = await repo.find_children_by_coordinator_run(
        coordinator_run_id=seeded_coordinator_children["run_id_a"],
        parent_session_id=seeded_coordinator_children["parent_a"],
    )
    work_unit_ids = sorted(c.work_unit_id for c in children)
    assert work_unit_ids == ["abcd1234abcd1234.a1.0", "abcd1234abcd1234.a1.1"], (
        f"expected exactly the two parent_a/run_a children; got: "
        f"{[(c.id, c.work_unit_id, c.parent_session_id) for c in children]}"
    )
    for c in children:
        assert c.parent_session_id == seeded_coordinator_children["parent_a"]
        assert c.coordinator_run_id == seeded_coordinator_children["run_id_a"]


async def test_find_children_empty_when_no_match(
    db_session, seeded_coordinator_children,
):
    repo = DBSessionRepository(db_session)
    children = await repo.find_children_by_coordinator_run(
        coordinator_run_id="no_such_run_id",
        parent_session_id=seeded_coordinator_children["parent_a"],
    )
    assert children == []


async def test_find_children_scoped_to_parent_not_just_run(
    db_session, seeded_coordinator_children,
):
    """Defense-in-depth: passing run_id_a but parent_b returns only c4."""
    repo = DBSessionRepository(db_session)
    children = await repo.find_children_by_coordinator_run(
        coordinator_run_id=seeded_coordinator_children["run_id_a"],
        parent_session_id=seeded_coordinator_children["parent_b"],
    )
    assert len(children) == 1
    assert children[0].parent_session_id == seeded_coordinator_children["parent_b"]
