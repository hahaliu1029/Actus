"""C1a /cost/tree endpoint integration test."""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
import sqlalchemy as sa

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def _insert_cost_row(db_session, *, session_id, user_id, usd, status="actual"):
    await db_session.execute(
        sa.text(
            "INSERT INTO cost_records (id, run_id, session_id, user_id, node_name, "
            "  provider, model, step_ix, total_usd, cost_status, pricing_version, "
            "  created_at) "
            "VALUES (:id, :rid, :sid, :uid, 'n', 'p', 'm', 0, :usd, :st, 'v1', now())"
        ),
        {
            "id": uuid.uuid4().hex,
            "rid": uuid.uuid4().hex,
            "sid": session_id,
            "uid": user_id,
            "usd": usd,
            "st": status,
        },
    )


async def _insert_child_session(db_session, *, child_id, user_id, parent_id):
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, parent_session_id, worker_type, "
            "  tool_filter_preset, title, latest_message, status, "
            "  events, files, memories) "
            "VALUES (:id, :uid, :pid, 'subagent', 'subagent_research', '', '', 'PENDING', "
            "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": child_id, "uid": user_id, "pid": parent_id},
    )


async def test_cost_tree_rolls_up_self_and_descendants(
    asgi_client, db_session, sample_user, sample_session, sample_user_token,
):
    child_id = uuid.uuid4().hex
    await _insert_child_session(
        db_session, child_id=child_id, user_id=sample_user.id, parent_id=sample_session.id,
    )
    await _insert_cost_row(
        db_session, session_id=sample_session.id, user_id=sample_user.id, usd="1.00",
    )
    await _insert_cost_row(
        db_session, session_id=child_id, user_id=sample_user.id, usd="0.50",
    )
    await db_session.commit()

    resp = await asgi_client.get(
        f"/api/sessions/{sample_session.id}/cost/tree",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["session_id"] == sample_session.id
    assert Decimal(body["self_cost"]["total_usd"]) == Decimal("1.00")
    assert Decimal(body["descendants_cost"]["total_usd"]) == Decimal("0.50")
    assert Decimal(body["total_cost"]["total_usd"]) == Decimal("1.50")
    assert body["descendant_ids"] == [child_id]
    assert body["truncated"] is False


async def test_cost_tree_returns_404_for_foreign_user(
    asgi_client, sample_session, other_user_token,
):
    resp = await asgi_client.get(
        f"/api/sessions/{sample_session.id}/cost/tree",
        headers={"Authorization": f"Bearer {other_user_token}"},
    )
    assert resp.status_code == 404


async def test_cost_tree_total_partial_when_descendant_partial(
    asgi_client, db_session, sample_user, sample_session, sample_user_token,
):
    child_id = uuid.uuid4().hex
    await _insert_child_session(
        db_session, child_id=child_id, user_id=sample_user.id, parent_id=sample_session.id,
    )
    await _insert_cost_row(
        db_session, session_id=sample_session.id, user_id=sample_user.id, usd="1.00",
    )
    await _insert_cost_row(
        db_session, session_id=child_id, user_id=sample_user.id, usd="0.50", status="partial",
    )
    await db_session.commit()

    resp = await asgi_client.get(
        f"/api/sessions/{sample_session.id}/cost/tree",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    body = resp.json()["data"]
    assert body["total_cost"]["cost_status"] == "partial"
