"""Sandbox timeout middleware and reset control endpoint tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from starlette.requests import Request

from app.core import middleware
from app.interfaces.endpoints import supervisor as supervisor_endpoints
from app.interfaces.schemas.supervisor import TimeoutRequest
from app.models.supervisor import SupervisorTimeout


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def _supervisor() -> SimpleNamespace:
    return SimpleNamespace(
        timeout_active=True,
        expand_enabled=True,
        reset_timeout=AsyncMock(),
        extend_timeout=AsyncMock(),
        disable_expand=MagicMock(),
    )


async def test_normal_api_resets_default_window_instead_of_extending(monkeypatch) -> None:
    service = _supervisor()
    monkeypatch.setattr(
        middleware,
        "get_settings",
        lambda: SimpleNamespace(server_timeout_minutes=60),
    )
    monkeypatch.setattr(middleware, "get_supervisor_service", lambda: service)
    response = object()
    call_next = AsyncMock(return_value=response)

    actual = await middleware.auto_extend_timeout_middleware(
        _request("/api/file/read-file"), call_next
    )

    assert actual is response
    service.reset_timeout.assert_awaited_once_with()
    service.extend_timeout.assert_not_awaited()


async def test_reset_control_endpoint_is_ignored_by_auto_renew(monkeypatch) -> None:
    service = _supervisor()
    monkeypatch.setattr(
        middleware,
        "get_settings",
        lambda: SimpleNamespace(server_timeout_minutes=60),
    )
    monkeypatch.setattr(middleware, "get_supervisor_service", lambda: service)

    await middleware.auto_extend_timeout_middleware(
        _request("/api/supervisor/reset-timeout"), AsyncMock(return_value=object())
    )

    service.reset_timeout.assert_not_awaited()
    service.extend_timeout.assert_not_awaited()


async def test_reset_endpoint_does_not_disable_future_auto_renew() -> None:
    route = next(
        (
            route
            for route in supervisor_endpoints.router.routes
            if route.path == "/supervisor/reset-timeout"
        ),
        None,
    )
    assert route is not None
    service = _supervisor()
    service.reset_timeout.return_value = SupervisorTimeout(
        status="timeout_reset",
        active=True,
        timeout_minutes=12,
        remaining_seconds=720,
    )

    response = await route.endpoint(
        TimeoutRequest(minutes=12), supervisor_service=service
    )

    assert response.data.timeout_minutes == 12
    service.reset_timeout.assert_awaited_once_with(12)
    service.disable_expand.assert_not_called()


def test_timeout_request_rejects_non_positive_minutes() -> None:
    for minutes in (0, -1, True):
        try:
            TimeoutRequest(minutes=minutes)
        except ValueError:
            continue
        raise AssertionError(f"minutes={minutes} should be rejected")
