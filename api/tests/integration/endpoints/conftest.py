"""HTTP integration test fixtures for B6 compaction routes.

These fixtures build on the DB fixtures defined in the parent
``api/tests/integration/conftest.py`` (``seed_session``,
``seed_other_user_session``) and add an authenticated
``httpx.AsyncClient`` layer.

Auth strategy: project uses HTTPBearer JWT (see
``api/app/interfaces/dependencies/auth.py:93-111`` and
``api/core/security.py``).  For integration tests we bypass the live
user-lookup by overriding ``get_current_user`` with a stub that returns
a minimal ``User`` domain object constructed from the seeded user_id.
This avoids standing up a full auth service while still exercising the
route authz logic.

No pg/Redis connections are made by the fixture itself — those are
delegated to the route handlers under test.
"""
from __future__ import annotations

import httpx
import pytest

from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.main import app


def _make_user(user_id: str) -> User:
    """Build a minimal domain User from a known user_id (no DB lookup)."""
    return User(
        id=user_id,
        username=f"integ_{user_id[:8]}",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


@pytest.fixture
async def api_client_for_user(seed_session):
    """``httpx.AsyncClient`` authenticated as ``seed_session.user_id``.

    Overrides ``get_current_user`` to return the seeded user without a
    live DB lookup, so the client works even when the route itself needs
    ``current_user.id`` for ownership checks.
    """
    user = _make_user(seed_session.user_id)
    app.dependency_overrides[get_current_user] = lambda: user
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def api_client_for_other_user(seed_other_user_session):
    """``httpx.AsyncClient`` authenticated as ``seed_other_user_session.user_id``.

    Used for cross-user 403 tests.
    """
    user = _make_user(seed_other_user_session.user_id)
    app.dependency_overrides[get_current_user] = lambda: user
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def api_client_unauthenticated():
    """``httpx.AsyncClient`` with *no* auth header — for 401 tests.

    Does NOT override ``get_current_user``; the real HTTPBearer dependency
    raises HTTP 401 when no Authorization header is present.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client
