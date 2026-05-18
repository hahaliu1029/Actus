"""PE domain exceptions map to spec-defined HTTP codes.

T16 (PE-1 §5.2) extends this with HTTP handlers for ``UnsupportedSource``
(422) and ``PEInfrastructureUnavailable`` (503). Pre-T16 tests cover
PolicyConflict (409) / SessionModeViolation (410) / WriterIntegrityError
(500); the new ``TestUnsupportedSource422`` / ``TestPEInfraUnavailable503``
classes live alongside them so future PRs see one canonical location for
PE → HTTP mapping coverage.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.services.permission.errors import (
    PEInfrastructureUnavailable,
    PolicyConflict,
    SessionModeViolation,
    UnsupportedSource,
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


# T16 (PE-1 §5.2) — HTTP exception handlers for UnsupportedSource (422) +
# PEInfrastructureUnavailable (503).


@pytest.fixture
def app_with_handlers() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/unsupported/{src}")
    async def _raise_unsupported(src: str):
        raise UnsupportedSource(src)

    @app.get("/raise/infra")
    async def _raise_infra():
        raise PEInfrastructureUnavailable("redis_get_fail_key: timeout")

    return app


class TestUnsupportedSource422:
    def test_returns_422(self, app_with_handlers):
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        r = client.get("/raise/unsupported/mystery")
        assert r.status_code == 422

    def test_body_includes_source_and_supported_sources(self, app_with_handlers):
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        r = client.get("/raise/unsupported/mystery")
        body = r.json()
        assert body["error"] == "unsupported_tool_source"
        assert body["source"] == "mystery"
        assert "native" in body["supported_sources"]
        assert "skill" in body["supported_sources"]


class TestPEInfraUnavailable503:
    def test_returns_503(self, app_with_handlers):
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        r = client.get("/raise/infra")
        assert r.status_code == 503

    def test_body_includes_error_and_correlation_id(self, app_with_handlers):
        client = TestClient(app_with_handlers, raise_server_exceptions=False)
        r = client.get("/raise/infra")
        body = r.json()
        assert body["error"] == "pe_infrastructure_unavailable"
        # correlation_id is a 16-char hex string for ops tracing
        assert isinstance(body["correlation_id"], str)
        assert len(body["correlation_id"]) == 16
