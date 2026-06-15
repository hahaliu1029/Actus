"""C2 coordinator-cancel — integration query tests (CI only; needs Postgres).

Run: cd api && SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \
     uv run pytest tests/integration/test_coordinator_cancel_queries.py -v

Covers the real SQL for ``find_running_mailbox_children_for_parent`` (fanout)
and ``find_terminal_coordinator_children_with_active_sandbox`` (reaper). The
orchestration is unit-tested in the test_coordinator_parent_cancel_fanout /
test_sandbox_terminal_reaper modules.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from app.infrastructure.models.session import SessionModel
from app.infrastructure.models.user import UserModel
from app.infrastructure.repositories.db_session_repository import DBSessionRepository

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def _mk_user(db_session) -> str:
    uid = str(_uuid.uuid4())
    db_session.add(UserModel(id=uid, username=f"c2cancel_{uid[:8]}", password_hash="x"))
    await db_session.flush()
    return uid


async def _mk_root(db_session, uid: str) -> str:
    rid = f"root-{_uuid.uuid4().hex[:12]}"
    db_session.add(SessionModel(
        id=rid, user_id=uid, status="running", title="coord root", worker_type="root",
    ))
    await db_session.flush()
    return rid


def _child(*, sid, uid, parent, status="running", plane="mailbox",
           mode="foreground", run_id="run-1", wu_id=None,
           preset="coordinator_step", sandbox_state="active"):
    # Each coordinator child needs a UNIQUE (parent, run_id, work_unit_id):
    # the partial UNIQUE index ``ux_sessions_coordinator_wu``
    # (alembic c2pr1_add_coordinator_columns.py) enforces it WHERE both lineage
    # cols are non-null. Derive a per-sid work_unit_id so multiple children
    # under one parent (sharing a run_id, as real siblings do) don't collide.
    # Null-lineage rows (run_id=None) are exempt — NULLs aren't constrained.
    if run_id is not None and wu_id is None:
        wu_id = f"wu-{sid}"
    return SessionModel(
        id=sid, user_id=uid, parent_session_id=parent, status=status,
        worker_type="subagent", subagent_control_plane=plane,
        execution_mode=mode, coordinator_run_id=run_id, work_unit_id=wu_id,
        tool_filter_preset=preset, sandbox_state=sandbox_state, title="child",
    )


async def test_fanout_selects_only_this_parents_running_coordinator_children(db_session):
    uid = await _mk_user(db_session)
    p1 = await _mk_root(db_session, uid)
    p2 = await _mk_root(db_session, uid)
    db_session.add_all([
        _child(sid="keep-1", uid=uid, parent=p1),                        # keep
        _child(sid="keep-2", uid=uid, parent=p1),                        # keep
        _child(sid="other-parent", uid=uid, parent=p2),                  # other parent
        _child(sid="terminal", uid=uid, parent=p1, status="completed"),  # terminal
        _child(sid="research", uid=uid, parent=p1, preset="subagent_research",
               run_id=None, wu_id=None),                                 # non-coordinator
        _child(sid="null-lineage", uid=uid, parent=p1, run_id=None, wu_id=None),  # null lineage
        _child(sid="background", uid=uid, parent=p1, mode="background"),  # background
    ])
    await db_session.flush()
    repo = DBSessionRepository(db_session=db_session)
    rows = await repo.find_running_mailbox_children_for_parent(p1)
    assert {r.session_id for r in rows} == {"keep-1", "keep-2"}


async def test_reaper_selects_only_terminal_active_coordinator_children(db_session):
    uid = await _mk_user(db_session)
    p1 = await _mk_root(db_session, uid)
    db_session.add_all([
        _child(sid="reap-1", uid=uid, parent=p1, status="completed", sandbox_state="active"),  # keep
        _child(sid="reap-2", uid=uid, parent=p1, status="timed_out", sandbox_state="active"),  # keep
        _child(sid="running", uid=uid, parent=p1, status="running", sandbox_state="active"),   # not terminal
        _child(sid="destroyed", uid=uid, parent=p1, status="completed", sandbox_state="destroyed"),  # not active
        _child(sid="suspended", uid=uid, parent=p1, status="completed", sandbox_state="suspended"),  # not active
        _child(sid="unbound", uid=uid, parent=p1, status="completed", sandbox_state="unbound"),      # not active
        _child(sid="research", uid=uid, parent=p1, status="completed", sandbox_state="active",
               preset="subagent_research", run_id=None, wu_id=None),                                 # non-coordinator
    ])
    await db_session.flush()
    repo = DBSessionRepository(db_session=db_session)
    rows = await repo.find_terminal_coordinator_children_with_active_sandbox()
    assert {r.session_id for r in rows} == {"reap-1", "reap-2"}
