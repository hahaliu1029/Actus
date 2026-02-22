import asyncio

import pytest
from app.domain.models.user import User
from app.interfaces.dependencies.auth import get_current_user_ws_query
from fastapi import HTTPException


def test_ws_query_auth_requires_token() -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_current_user_ws_query(None))

    assert exc.value.status_code == 401


def test_ws_query_auth_resolves_user(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolver(token: str) -> User:
        assert token == "valid-token"
        return User(id="u1", username="demo")

    monkeypatch.setattr(
        "app.interfaces.dependencies.auth.resolve_user_from_access_token", fake_resolver
    )

    user = asyncio.run(get_current_user_ws_query("valid-token"))
    assert user.id == "u1"
