"""C1a /children endpoint integration test."""
from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_happy_path_returns_descendants(
    asgi_client, uow_factory, sample_user, sample_session, sample_user_token,
):
    """Seed 3 children via SessionService and assert /children returns them.

    Order is set-based: the recursive CTE orders by (depth ASC, id ASC) but
    with all children at depth=1 the ordering is purely id ASC. We compare
    sorted lists so future ordering tweaks don't flake this test.
    """
    from app.application.services.session_service import SessionService
    svc = SessionService(uow_factory=uow_factory)
    child_ids = []
    for _ in range(3):
        child = await svc.create_session_with_parent(
            user_id=sample_user.id,
            parent_session_id=sample_session.id,
            tool_filter_preset="subagent_research",
        )
        child_ids.append(child.id)
    expected = sorted(child_ids)

    resp = await asgi_client.get(
        f"/api/sessions/{sample_session.id}/children",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()["data"]
    returned_ids = [item["id"] for item in body["descendants"]]
    assert sorted(returned_ids) == expected
    assert body["truncated"] is False
    assert body["depth_applied"] == 1


async def test_foreign_user_session_returns_404(
    asgi_client, sample_session, other_user_token,
):
    """ID enumeration defense: foreign-user session must return 404, not 403."""
    resp = await asgi_client.get(
        f"/api/sessions/{sample_session.id}/children",
        headers={"Authorization": f"Bearer {other_user_token}"},
    )
    assert resp.status_code == 404


async def test_missing_session_returns_404(asgi_client, sample_user_token):
    resp = await asgi_client.get(
        f"/api/sessions/{uuid.uuid4().hex}/children",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert resp.status_code == 404


async def test_truncated_when_descendants_exceed_cap(
    asgi_client, db_session, sample_user, sample_session, sample_user_token,
):
    """Seed > MAX_DESCENDANTS_PER_ROOT via direct DB to bypass the per-spawn cap."""
    for _ in range(11):
        cid = uuid.uuid4().hex
        await db_session.execute(
            sa.text(
                "INSERT INTO sessions (id, user_id, parent_session_id, worker_type, "
                "  tool_filter_preset, title, latest_message, status, "
                "  events, files, memories) "
                "VALUES (:id, :uid, :pid, 'subagent', 'subagent_research', '', '', 'PENDING', "
                "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
            ),
            {"id": cid, "uid": sample_user.id, "pid": sample_session.id},
        )
    await db_session.commit()
    resp = await asgi_client.get(
        f"/api/sessions/{sample_session.id}/children",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["truncated"] is True
    assert len(body["descendants"]) == 10
