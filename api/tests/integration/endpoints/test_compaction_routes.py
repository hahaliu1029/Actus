"""Integration tests for /api/sessions/{session_id}/compactions HTTP endpoints.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md § Section 3

Uses committed-seed pattern (uow_factory() + explicit commit + try/finally cleanup)
so that ASGI route handlers opening independent DB connections can see the rows.

Auth: api_client_for_user / api_client_for_other_user / api_client_unauthenticated
fixtures from tests/integration/endpoints/conftest.py override get_current_user with
a stub that bypasses the live JWT lookup.

Local pg unreachable → CI-only; acceptable per project convention.
Run: cd api && uv run pytest -m integration tests/integration/endpoints/test_compaction_routes.py -v
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# ── helpers ──────────────────────────────────────────────────────────────────

async def _seed_committed(uow_factory, *, with_compaction: bool = False, pre_compact_checkpoint_id=None):
    """Commit user + session (+ optionally one compaction row) via uow_factory.

    Returns a dict with keys: user_id, session_id, compaction_id (or None).
    """
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.models.conversation_compaction import ConversationCompactionModel

    user_id = str(_uuid.uuid4())
    session_id = f"sess-t18-{_uuid.uuid4().hex[:12]}"
    compaction_id = None

    async with uow_factory() as uow:
        uow.db_session.add_all([
            UserModel(id=user_id, username=f"t18_{user_id[:8]}", password_hash="x"),
            SessionModel(id=session_id, user_id=user_id, status="pending", title="t18 test"),
        ])
        if with_compaction:
            compaction_id = _uuid.uuid4().hex[:16]
            uow.db_session.add(
                ConversationCompactionModel(
                    compaction_id=compaction_id,
                    session_id=session_id,
                    summary="test summary for route test",
                    summary_tokens=42,
                    operations=[{"kind": "llm_summary", "tokens_before": 100, "tokens_after": 50}],
                    parent_compaction_id=None,
                    first_visible_event_id=None,
                    last_visible_event_id=None,
                    pre_compact_checkpoint_id=pre_compact_checkpoint_id,
                    tokens_before_total=100,
                    tokens_after_total=50,
                    messages_removed_total=5,
                    created_at=datetime.now(timezone.utc),
                )
            )
        await uow.db_session.commit()

    return {"user_id": user_id, "session_id": session_id, "compaction_id": compaction_id}


async def _cleanup(uow_factory, user_id: str, session_id: str, compaction_id: str | None = None):
    from app.infrastructure.models.user import UserModel
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.models.conversation_compaction import ConversationCompactionModel

    async with uow_factory() as uow:
        if compaction_id:
            await uow.db_session.execute(
                delete(ConversationCompactionModel).where(
                    ConversationCompactionModel.compaction_id == compaction_id
                )
            )
        await uow.db_session.execute(
            delete(SessionModel).where(SessionModel.id == session_id)
        )
        await uow.db_session.execute(
            delete(UserModel).where(UserModel.id == user_id)
        )
        await uow.db_session.commit()


from contextlib import asynccontextmanager


@asynccontextmanager
async def _build_test_client(async_session_factory, user_for_auth):
    """Build an httpx ASGI client with both get_uow + get_current_user overridden.

    The 3 shared api_client_* conftest fixtures already do this for fixed users;
    use this helper when an ad-hoc user identity is needed (e.g., cross-user 403).

    ``user_for_auth`` must be a ``User`` domain object (or any object accepted by
    the route's ``CurrentUser`` dependency).  Pass ``None`` to skip the
    ``get_current_user`` override (useful when testing unauthenticated paths).
    """
    import httpx
    from app.infrastructure.repositories.db_uow import DBUnitOfWork
    from app.infrastructure.storage.postgres import get_uow as _real_get_uow
    from app.interfaces.dependencies.auth import get_current_user
    from app.main import app

    def _override_get_uow():
        return DBUnitOfWork(session_factory=async_session_factory)

    app.dependency_overrides[_real_get_uow] = _override_get_uow
    if user_for_auth is not None:
        app.dependency_overrides[get_current_user] = lambda: user_for_auth
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.pop(_real_get_uow, None)
        if user_for_auth is not None:
            app.dependency_overrides.pop(get_current_user, None)


# ── list endpoint ─────────────────────────────────────────────────────────────

async def test_list_endpoint_returns_records_sorted_desc(uow_factory, async_session_factory):
    """GET /api/sessions/{id}/compactions returns items in descending created_at order.

    Inserts TWO compactions with distinct created_at values and asserts the
    response lists the newer one first.
    """
    from app.infrastructure.models.conversation_compaction import ConversationCompactionModel
    from app.infrastructure.repositories.db_uow import DBUnitOfWork
    from app.infrastructure.storage.postgres import get_uow as _real_get_uow
    from app.interfaces.dependencies.auth import get_current_user
    from app.main import app
    import httpx
    from app.domain.models.user import User, UserRole, UserStatus

    seed = await _seed_committed(uow_factory)
    user_id = seed["user_id"]
    session_id = seed["session_id"]

    # Two compaction rows with known distinct timestamps so we can assert sort order.
    cid_older = _uuid.uuid4().hex[:16]
    cid_newer = _uuid.uuid4().hex[:16]
    ts_older = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    ts_newer = datetime(2026, 5, 2, 12, 0, 0, tzinfo=timezone.utc)

    async with uow_factory() as uow:
        uow.db_session.add_all([
            ConversationCompactionModel(
                compaction_id=cid_older,
                session_id=session_id,
                summary="older compaction",
                summary_tokens=10,
                operations=[{"kind": "llm_summary", "tokens_before": 80, "tokens_after": 40}],
                parent_compaction_id=None,
                first_visible_event_id=None,
                last_visible_event_id=None,
                pre_compact_checkpoint_id=None,
                tokens_before_total=80,
                tokens_after_total=40,
                messages_removed_total=3,
                created_at=ts_older,
            ),
            ConversationCompactionModel(
                compaction_id=cid_newer,
                session_id=session_id,
                summary="newer compaction",
                summary_tokens=15,
                operations=[{"kind": "llm_summary", "tokens_before": 90, "tokens_after": 45}],
                parent_compaction_id=None,
                first_visible_event_id=None,
                last_visible_event_id=None,
                pre_compact_checkpoint_id=None,
                tokens_before_total=90,
                tokens_after_total=45,
                messages_removed_total=4,
                created_at=ts_newer,
            ),
        ])
        await uow.db_session.commit()

    try:
        user = User(id=user_id, username=f"t18_{user_id[:8]}", role=UserRole.USER, status=UserStatus.ACTIVE)
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[_real_get_uow] = lambda: DBUnitOfWork(
            session_factory=async_session_factory
        )
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(f"/api/sessions/{session_id}/compactions")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "items" in body
            ids = [item["compaction_id"] for item in body["items"]]
            # Both compactions must be present and the newer one first (DESC order).
            assert cid_newer in ids
            assert cid_older in ids
            assert ids.index(cid_newer) < ids.index(cid_older), (
                f"Expected newer ({cid_newer}) before older ({cid_older}), got: {ids}"
            )
        finally:
            app.dependency_overrides.pop(_real_get_uow, None)
            app.dependency_overrides.pop(get_current_user, None)
    finally:
        # Clean up both compactions then the session/user.
        from app.infrastructure.models.conversation_compaction import ConversationCompactionModel as _CCM
        from app.infrastructure.models.session import SessionModel
        from app.infrastructure.models.user import UserModel
        from sqlalchemy import delete as _delete
        async with uow_factory() as uow:
            await uow.db_session.execute(
                _delete(_CCM).where(_CCM.compaction_id.in_([cid_older, cid_newer]))
            )
            await uow.db_session.execute(
                _delete(SessionModel).where(SessionModel.id == session_id)
            )
            await uow.db_session.execute(
                _delete(UserModel).where(UserModel.id == user_id)
            )
            await uow.db_session.commit()


async def test_list_endpoint_403_when_no_auth(api_client_unauthenticated):
    """GET /api/sessions/{id}/compactions without auth returns 401/403."""
    resp = await api_client_unauthenticated.get("/api/sessions/nonexistent-session/compactions")
    assert resp.status_code in (401, 403)


async def test_list_endpoint_cross_user_403(uow_factory, async_session_factory):
    """GET compactions for a session owned by user A returns 403 when client is user B."""
    from app.domain.models.user import User, UserRole, UserStatus

    seed_a = await _seed_committed(uow_factory)   # session owned by user A
    seed_b = await _seed_committed(uow_factory)   # user B (different)

    try:
        # Authenticate as user B, request session owned by user A
        user_b = User(
            id=seed_b["user_id"],
            username=f"t18b_{seed_b['user_id'][:8]}",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )
        async with _build_test_client(async_session_factory, user_for_auth=user_b) as client:
            resp = await client.get(f"/api/sessions/{seed_a['session_id']}/compactions")
        assert resp.status_code == 403, resp.text
    finally:
        await _cleanup(uow_factory, seed_a["user_id"], seed_a["session_id"])
        await _cleanup(uow_factory, seed_b["user_id"], seed_b["session_id"])


# ── detail endpoint ───────────────────────────────────────────────────────────

async def test_detail_endpoint_404_when_missing(uow_factory, async_session_factory):
    """GET /api/sessions/{id}/compactions/{cid} returns 404 for unknown compaction_id."""
    from app.domain.models.user import User, UserRole, UserStatus

    seed = await _seed_committed(uow_factory)
    try:
        user = User(
            id=seed["user_id"],
            username=f"t18_{seed['user_id'][:8]}",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )
        async with _build_test_client(async_session_factory, user_for_auth=user) as client:
            resp = await client.get(
                f"/api/sessions/{seed['session_id']}/compactions/nonexistent0000001"
            )
        assert resp.status_code == 404, resp.text
    finally:
        await _cleanup(uow_factory, seed["user_id"], seed["session_id"])


async def test_detail_endpoint_403_or_404_cross_user(uow_factory, async_session_factory):
    """GET compaction detail for session owned by user A returns 403 when client is user B."""
    from app.domain.models.user import User, UserRole, UserStatus

    seed_a = await _seed_committed(uow_factory)
    seed_b = await _seed_committed(uow_factory)

    try:
        user_b = User(
            id=seed_b["user_id"],
            username=f"t18b_{seed_b['user_id'][:8]}",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )
        async with _build_test_client(async_session_factory, user_for_auth=user_b) as client:
            resp = await client.get(
                f"/api/sessions/{seed_a['session_id']}/compactions/any-compaction-id"
            )
        assert resp.status_code == 403, resp.text
    finally:
        await _cleanup(uow_factory, seed_a["user_id"], seed_a["session_id"])
        await _cleanup(uow_factory, seed_b["user_id"], seed_b["session_id"])


# ── original-content endpoint ─────────────────────────────────────────────────

async def test_original_content_410_when_pre_compact_checkpoint_id_null(
    uow_factory, async_session_factory
):
    """GET .../original-content returns 410 when pre_compact_checkpoint_id is null."""
    from app.domain.models.user import User, UserRole, UserStatus

    seed = await _seed_committed(uow_factory, with_compaction=True, pre_compact_checkpoint_id=None)
    try:
        user = User(
            id=seed["user_id"],
            username=f"t18_{seed['user_id'][:8]}",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )
        async with _build_test_client(async_session_factory, user_for_auth=user) as client:
            resp = await client.get(
                f"/api/sessions/{seed['session_id']}/compactions"
                f"/{seed['compaction_id']}/original-content"
            )
        assert resp.status_code == 410, resp.text
        body = resp.json()
        assert body["error"] == "checkpointer_expired"
        assert body["summary_still_available"] is True
    finally:
        await _cleanup(
            uow_factory,
            seed["user_id"],
            seed["session_id"],
            seed["compaction_id"],
        )


async def test_original_content_200_happy_path(uow_factory, async_session_factory, monkeypatch):
    """GET .../original-content returns 200 + recovered_messages when recovery succeeds.

    Monkeypatches ``recover_original_messages`` in the routes module so the test
    does not require a live LangGraph checkpointer.  Asserts:
    - 200 status code
    - ``recovered_messages[0].type == "human"`` and ``.content == "hello"``
    - ``recovered_messages[1].type == "ai"`` and ``.content == "hi"``
    - compaction row is seeded with ``pre_compact_checkpoint_id == "ck_xyz_test_001"``
    """
    from app.domain.models.user import User, UserRole, UserStatus
    from app.main import app
    from langchain_core.messages import AIMessage, HumanMessage
    import app.interfaces.endpoints.session_compaction_routes as _routes_mod

    ck_id = "ck_xyz_test_001"
    seed = await _seed_committed(
        uow_factory, with_compaction=True, pre_compact_checkpoint_id=ck_id
    )

    # Monkeypatch recover_original_messages at the routes-module level so the
    # import already in session_compaction_routes.py is replaced.
    async def _fake_recover(pool, session_id, pre_compact_checkpoint_id):
        return [HumanMessage(content="hello"), AIMessage(content="hi")]

    monkeypatch.setattr(_routes_mod, "recover_original_messages", _fake_recover)

    # Seed app.state.checkpointer_pool with a sentinel so that the route's
    # ``pool = request.app.state.checkpointer_pool`` access doesn't raise
    # AttributeError.  ASGITransport skips lifespan, so the real pool is never
    # initialised.  The sentinel is never actually used because recover_original_messages
    # is monkeypatched to ignore its first argument.
    from unittest.mock import MagicMock
    _had_pool = hasattr(app.state, "checkpointer_pool")
    _prior_pool = getattr(app.state, "checkpointer_pool", None)
    if not _had_pool:
        app.state.checkpointer_pool = MagicMock()

    try:
        user = User(
            id=seed["user_id"],
            username=f"t18_{seed['user_id'][:8]}",
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )
        async with _build_test_client(async_session_factory, user_for_auth=user) as client:
            resp = await client.get(
                f"/api/sessions/{seed['session_id']}/compactions"
                f"/{seed['compaction_id']}/original-content"
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        msgs = body["recovered_messages"]
        assert len(msgs) == 2
        assert msgs[0]["type"] == "human"
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["type"] == "ai"
        assert msgs[1]["content"] == "hi"
        assert body["pre_compact_checkpoint_id"] == ck_id
    finally:
        # Restore the original app.state.checkpointer_pool so this test doesn't
        # leak the sentinel (or our `_prior_pool` snapshot) into other tests.
        if _had_pool:
            app.state.checkpointer_pool = _prior_pool
        else:
            try:
                del app.state.checkpointer_pool
            except AttributeError:
                pass
        await _cleanup(
            uow_factory,
            seed["user_id"],
            seed["session_id"],
            seed["compaction_id"],
        )
