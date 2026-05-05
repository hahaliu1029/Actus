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

UoW strategy: ``httpx.ASGITransport`` does NOT execute FastAPI lifespan
handlers, so ``get_uow()``'s underlying ``get_postgres().session_factory``
would point to an uninitialized singleton rather than the test DB.  Every
``api_client_*`` fixture therefore overrides ``get_uow`` to return a
``DBUnitOfWork`` bound to the test ``async_session_factory`` — ensuring
that the route handler reads from the same DB the test wrote to.
"""
from __future__ import annotations

import httpx
import pytest

from app.domain.models.user import User, UserRole, UserStatus
from app.infrastructure.repositories.db_uow import DBUnitOfWork
from app.infrastructure.storage.postgres import get_uow as _real_get_uow
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
async def api_client_for_user(seed_session, async_session_factory):
    """``httpx.AsyncClient`` authenticated as ``seed_session.user_id``.

    Overrides ``get_current_user`` to return the seeded user without a
    live DB lookup, so the client works even when the route itself needs
    ``current_user.id`` for ownership checks.

    Overrides ``get_uow`` with a factory bound to the test
    ``async_session_factory`` so that route handlers see the same DB the
    test wrote to (ASGITransport skips lifespan, so the real singleton
    would point to an uninitialised connection).
    """
    user = _make_user(seed_session.user_id)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[_real_get_uow] = lambda: DBUnitOfWork(
        session_factory=async_session_factory
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(_real_get_uow, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def api_client_for_other_user(seed_other_user_session, async_session_factory):
    """``httpx.AsyncClient`` authenticated as ``seed_other_user_session.user_id``.

    Used for cross-user 403 tests.  Also overrides ``get_uow`` — see
    ``api_client_for_user`` docstring for rationale.
    """
    user = _make_user(seed_other_user_session.user_id)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[_real_get_uow] = lambda: DBUnitOfWork(
        session_factory=async_session_factory
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(_real_get_uow, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def api_client_unauthenticated(async_session_factory):
    """``httpx.AsyncClient`` with *no* auth header — for 401 tests.

    Does NOT override ``get_current_user``; the real HTTPBearer dependency
    raises HTTP 401 when no Authorization header is present.

    Still overrides ``get_uow`` so that any route code that runs before
    auth rejection uses the test DB.
    """
    app.dependency_overrides[_real_get_uow] = lambda: DBUnitOfWork(
        session_factory=async_session_factory
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(_real_get_uow, None)
