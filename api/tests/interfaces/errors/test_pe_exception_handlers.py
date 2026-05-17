"""PE domain exceptions map to spec-defined HTTP codes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.services.permission.errors import (
    PolicyConflict,
    SessionModeViolation,
    WriterIntegrityError,
)
from app.interfaces.errors.exception_handlers import register_exception_handlers


@pytest.fixture
def client():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/policy-conflict")
    async def raise_policy_conflict():
        raise PolicyConflict("approval_already_claimed")

    @app.get("/raise/session-mode-violation")
    async def raise_session_mode_violation():
        raise SessionModeViolation("session in finishing")

    @app.get("/raise/writer-integrity")
    async def raise_writer_integrity():
        raise WriterIntegrityError("integrity error")

    return TestClient(app, raise_server_exceptions=False)


def test_policy_conflict_maps_to_409(client):
    r = client.get("/raise/policy-conflict")
    assert r.status_code == 409
    body = r.json()
    assert "approval_already_claimed" in body.get("error", "")


def test_session_mode_violation_maps_to_410(client):
    r = client.get("/raise/session-mode-violation")
    assert r.status_code == 410


def test_writer_integrity_maps_to_500(client):
    r = client.get("/raise/writer-integrity")
    assert r.status_code == 500
    body = r.json()
    assert body.get("error") == "internal"
    assert "correlation_id" in body
