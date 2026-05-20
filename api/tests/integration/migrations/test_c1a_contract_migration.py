"""C1a PR-4 contract migration tests."""
from __future__ import annotations

import pytest
import sqlalchemy as sa

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_sample_session_id_column_dropped(db_session):
    cols = (await db_session.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            " WHERE table_name='sessions' AND column_name='sample_session_id'"
        )
    )).all()
    assert cols == [], "PR-4 must drop sample_session_id"


async def test_child_must_have_preset_check_now_references_parent_session_id(db_session):
    """ck_sessions_child_must_have_preset must reference parent_session_id, not the dropped column."""
    rows = (await db_session.execute(
        sa.text(
            "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
            " WHERE conrelid='sessions'::regclass AND conname='ck_sessions_child_must_have_preset'"
        )
    )).all()
    assert len(rows) == 1
    assert "parent_session_id" in rows[0].def_, rows[0].def_
    assert "sample_session_id" not in rows[0].def_, rows[0].def_


async def test_mirror_function_dropped(db_session):
    rows = (await db_session.execute(
        sa.text(
            "SELECT proname FROM pg_proc WHERE proname='mirror_sample_to_parent'"
        )
    )).all()
    assert rows == []


async def test_parent_user_trigger_swapped_to_parent_only(db_session):
    """PR-4 must swap ensure_parent_same_user() + trg_sessions_parent_user_match
    to parent-only form. The original c1a function body referenced
    NEW.sample_session_id via COALESCE; after the column drop, the body must
    reference only parent_session_id or the next INSERT/UPDATE will raise
    "record has no field sample_session_id"."""
    func_body = (await db_session.execute(
        sa.text(
            "SELECT pg_get_functiondef(oid) AS def FROM pg_proc "
            " WHERE proname='ensure_parent_same_user'"
        )
    )).first()
    assert func_body is not None
    assert "sample_session_id" not in func_body.def_, func_body.def_
    assert "parent_session_id" in func_body.def_, func_body.def_

    trig_def = (await db_session.execute(
        sa.text(
            "SELECT pg_get_triggerdef(oid) AS def FROM pg_trigger "
            " WHERE tgname='trg_sessions_parent_user_match'"
        )
    )).first()
    assert trig_def is not None
    assert "sample_session_id" not in trig_def.def_, trig_def.def_
