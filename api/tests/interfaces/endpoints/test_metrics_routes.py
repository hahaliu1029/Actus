"""B5 PR-S3-3: ``/api/v1/metrics`` endpoint contract.

Locks the auth surface and rendering shape:

- Token unset (default) → 404 with FastAPI's standard
  ``{"detail":"Not Found"}`` body — the same shape an unauthenticated
  probe would see for any non-existent path. Endpoint is invisible
  to unauthenticated probes; default deployments leak nothing.
- Token set + missing / wrong-scheme / wrong-value Authorization →
  401 + ``WWW-Authenticate: Bearer`` header. Constant-time compare
  via ``secrets.compare_digest``.
- Token set + correct ``Authorization: Bearer <token>`` → 200 with
  ``Content-Type`` ``prometheus_client.CONTENT_TYPE_LATEST`` and a
  Prometheus exposition body (``# HELP`` / ``# TYPE`` lines).
- The route is excluded from OpenAPI (``include_in_schema=False``)
  so it doesn't surface in ``/docs`` for end users.
- An OTel-side instrument written through ``OtelMeter`` after
  ``setup_observability`` MUST appear in the scrape body — locks the
  end-to-end pull-bridge from OTel SDK → ``prometheus_client.REGISTRY``
  → endpoint output.
- Production registered-handler chain: when ``register_exception_handlers``
  is mounted, the global ``HTTPException`` handler MUST merge
  ``exc.headers`` into the response — without that, ``WWW-Authenticate:
  Bearer`` is silently stripped on every 401 (auth-flow + metrics).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.infrastructure.observability import (
    setup_observability,
    teardown_observability,
)
from app.interfaces.endpoints.metrics_routes import router as metrics_router
from core.config import get_settings


@pytest.fixture(autouse=True)
def _reset_provider_state():
    """Each test starts and ends with no providers + clean registry."""
    teardown_observability()
    yield
    teardown_observability()


@pytest.fixture(autouse=True)
def _clean_default_env(monkeypatch):
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    monkeypatch.delenv("OTLP_PROTOCOL", raising=False)
    monkeypatch.delenv("METRICS_ENDPOINT_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _build_app() -> FastAPI:
    """Minimal FastAPI app exposing only the metrics router.

    Mirrors the production mount: ``main.py`` includes ``api_router``
    with ``prefix="/api"``; ``api_router`` includes
    ``metrics_router`` (``prefix="/v1/metrics"``). The composed path
    is ``/api/v1/metrics``. Bypassing the full app keeps these tests
    independent of the postgres / redis / lifespan stack.
    """
    app = FastAPI()
    app.include_router(metrics_router, prefix="/api")
    return app


def _client(monkeypatch, *, token: str = "") -> TestClient:
    if token:
        monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", token)
    else:
        monkeypatch.delenv("METRICS_ENDPOINT_TOKEN", raising=False)
    get_settings.cache_clear()
    return TestClient(_build_app())


# ---------------------------------------------------------------------------
# Disabled state.
# ---------------------------------------------------------------------------


def test_disabled_endpoint_returns_404(monkeypatch):
    """Default config (token empty) → endpoint returns 404."""
    client = _client(monkeypatch)
    response = client.get("/api/v1/metrics")
    assert response.status_code == 404


def test_disabled_endpoint_returns_404_even_with_authorization(monkeypatch):
    """Token unset → 404 regardless of Authorization header value.

    Ensures the disabled-mode 404 takes precedence over any auth
    parsing — no information about the disabled state should leak.
    """
    client = _client(monkeypatch)
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer anything"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Auth failure.
# ---------------------------------------------------------------------------


def test_missing_authorization_returns_401(monkeypatch):
    client = _client(monkeypatch, token="secret-1")
    response = client.get("/api/v1/metrics")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


def test_wrong_scheme_returns_401(monkeypatch):
    """``Basic <token>`` is not accepted — only Bearer."""
    client = _client(monkeypatch, token="secret-1")
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Basic c2VjcmV0LTE="},
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


def test_wrong_token_returns_401(monkeypatch):
    client = _client(monkeypatch, token="secret-correct")
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer secret-wrong"},
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


def test_empty_bearer_token_returns_401(monkeypatch):
    """``Authorization: Bearer  `` (empty after whitespace) is 401."""
    client = _client(monkeypatch, token="secret-correct")
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer    "},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Non-ASCII safety — reviewer round P3.
#
# ``secrets.compare_digest`` rejects non-ASCII ``str`` arguments with
# ``TypeError``. Without the UTF-8 bytes encoding in
# ``metrics_routes.get_metrics``, a non-ASCII configured token would
# crash the handler and surface as HTTP 500 instead of the spec's 401.
#
# The realistic threat model is the **configured token** carrying
# non-ASCII (operators paste a passphrase like ``αβγ-secret``); a
# real client cannot put non-ASCII bytes into the ``Authorization``
# header value because the HTTP/1.1 transport layer (httpx, browsers,
# starlette) rejects them at encode time. So the lock is exactly:
# non-ASCII expected + ASCII presented + mismatch must produce 401,
# not 500. Without the bytes-compare fix this test fails because
# ``secrets.compare_digest("ascii-wrong", "αβγ-secret")`` raises
# ``TypeError`` from inside the handler.
# ---------------------------------------------------------------------------


def test_non_ascii_configured_token_with_ascii_wrong_bearer_returns_401(
    monkeypatch,
):
    """Configured token has non-ASCII chars, presented bearer is plain
    ASCII (and wrong). Without bytes-compare this would 500.
    """
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "αβγ-secret")
    get_settings.cache_clear()
    client = TestClient(_build_app())
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer ascii-wrong"},
    )
    assert response.status_code == 401, (
        f"non-ASCII configured token must still produce 401 on mismatch; "
        f"got {response.status_code} body[:200]={response.text[:200]!r}"
    )
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


# ---------------------------------------------------------------------------
# Auth success.
# ---------------------------------------------------------------------------


def test_correct_token_returns_200_prometheus_format(monkeypatch):
    """Right token → 200 + Prometheus content-type + body has
    exposition syntax (``# HELP`` and / or ``# TYPE`` lines).
    """
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-ok")
    get_settings.cache_clear()
    setup_observability()  # registers PrometheusMetricReader
    client = TestClient(_build_app())

    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer secret-ok"},
    )
    assert response.status_code == 200
    ct = response.headers.get("content-type", "")
    # Lock against ``prometheus_client.CONTENT_TYPE_LATEST`` itself
    # rather than a literal version string — recent
    # ``prometheus_client`` releases ship OpenMetrics text format
    # (``version=1.0.0``) while older ones emit
    # legacy Prometheus 0.0.4. The contract is "whatever the library
    # currently considers latest", which is what the endpoint
    # advertises and what scrapers negotiate against.
    from prometheus_client import CONTENT_TYPE_LATEST

    assert ct == CONTENT_TYPE_LATEST, (
        f"unexpected content-type: {ct!r}; expected {CONTENT_TYPE_LATEST!r}"
    )

    body = response.text
    # Default ``prometheus_client`` collectors (GCCollector,
    # PlatformCollector, ProcessCollector) always produce ``# HELP`` /
    # ``# TYPE`` lines, so this lock holds even when the OTel-side
    # collector has no data yet.
    assert "# HELP" in body, (
        f"Prometheus body missing # HELP lines; got body[:200]={body[:200]!r}"
    )
    assert "# TYPE" in body, (
        f"Prometheus body missing # TYPE lines; got body[:200]={body[:200]!r}"
    )


def test_otel_meter_counter_appears_in_scrape_body(monkeypatch):
    """End-to-end pull bridge: counter incremented via OtelMeter
    surfaces in the Prometheus exposition body.

    Locks the OTel SDK → ``PrometheusMetricReader`` → ``REGISTRY``
    → ``generate_latest`` chain — without this the endpoint could
    silently render only the default ``prometheus_client`` collectors
    while OTel-side metrics never reach the wire.
    """
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-e2e")
    get_settings.cache_clear()
    providers = setup_observability()

    from opentelemetry import metrics as otel_metrics

    meter = otel_metrics.get_meter("test-metrics-endpoint")
    counter = meter.create_counter(
        "actus_metrics_endpoint_smoke_total",
        unit="",
        description="smoke test counter",
    )
    counter.add(7, attributes={"phase": "test"})
    # PrometheusMetricReader is pull-based; ``collect()`` runs at
    # scrape time. ``generate_latest`` calls each collector's
    # ``collect()``, which in turn drains the OTel pipeline.

    client = TestClient(_build_app())
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer secret-e2e"},
    )
    assert response.status_code == 200
    body = response.text
    # ``prometheus_client`` exposition normalizes ``_total`` counter
    # suffixes — match the metric NAME root so the test survives
    # naming policy tweaks.
    assert "actus_metrics_endpoint_smoke" in body, (
        f"OTel-recorded counter missing from scrape body; "
        f"snippet={body[-2000:]!r}"
    )
    # Specifically lock the value we wrote (7) to prevent the test
    # passing on stale state from a previous run leaking through.
    assert "7.0" in body or "7\n" in body, (
        f"counter value 7 not visible in scrape body; "
        f"snippet={body[-2000:]!r}"
    )
    assert providers is not None  # silence unused warning


# ---------------------------------------------------------------------------
# OpenAPI surface.
# ---------------------------------------------------------------------------


def test_metrics_route_excluded_from_openapi_schema():
    """``include_in_schema=False`` → ``/api/v1/metrics`` is invisible
    in ``/docs`` and ``/openapi.json``.
    """
    app = _build_app()
    schema = app.openapi()
    paths = list(schema.get("paths", {}).keys())
    assert "/api/v1/metrics" not in paths, (
        f"metrics route must be excluded from OpenAPI; got paths={paths!r}"
    )


# ---------------------------------------------------------------------------
# Registered-handler chain (production parity) — reviewer round.
#
# The bare-router tests above mount only ``metrics_router`` on a fresh
# FastAPI app, so an ``HTTPException`` raised inside ``get_metrics``
# is rendered by FastAPI's *default* HTTP exception handler, which
# preserves ``exc.headers`` verbatim. Production
# (``app/main.py``) calls ``register_exception_handlers(app)``, which
# overrides that default with a project-specific handler that
# normalises the response body shape — and previously dropped
# ``exc.headers`` in the process. These tests mount the production
# handler and lock that ``WWW-Authenticate: Bearer`` survives the
# round-trip on every 401 path.
# ---------------------------------------------------------------------------


def _build_app_with_handlers() -> FastAPI:
    """Like ``_build_app`` but also registers the production global
    exception handlers — the chain ``app/main.py`` actually mounts.
    """
    from app.interfaces.errors.exception_handlers import (
        register_exception_handlers,
    )

    app = FastAPI()
    app.include_router(metrics_router, prefix="/api")
    register_exception_handlers(app)
    return app


def test_registered_handler_preserves_www_authenticate_on_missing_auth(
    monkeypatch,
):
    """Production parity: missing ``Authorization`` → 401 + the
    ``WWW-Authenticate: Bearer`` header survives the global
    ``http_exception_handler``'s response normalisation.
    """
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "production-secret")
    get_settings.cache_clear()
    client = TestClient(_build_app_with_handlers())
    response = client.get("/api/v1/metrics")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower() == "bearer", (
        f"global handler must propagate WWW-Authenticate; "
        f"headers={dict(response.headers)!r}"
    )


def test_registered_handler_preserves_www_authenticate_on_wrong_token(
    monkeypatch,
):
    """Production parity: wrong token → 401 + ``WWW-Authenticate:
    Bearer`` header survives the global handler.
    """
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "production-secret")
    get_settings.cache_clear()
    client = TestClient(_build_app_with_handlers())
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer not-the-secret"},
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower() == "bearer", (
        f"global handler must propagate WWW-Authenticate; "
        f"headers={dict(response.headers)!r}"
    )


def test_registered_handler_disabled_endpoint_returns_404_via_global_handler(
    monkeypatch,
):
    """Production parity: token unset → 404 still goes through the
    global handler cleanly (no header to preserve, but the handler
    must not crash on ``exc.headers is None``).
    """
    monkeypatch.delenv("METRICS_ENDPOINT_TOKEN", raising=False)
    get_settings.cache_clear()
    client = TestClient(_build_app_with_handlers())
    response = client.get("/api/v1/metrics")
    assert response.status_code == 404
    # 404 path raises HTTPException without ``headers=`` so
    # ``exc.headers is None``; the merged dict should be empty
    # (or contain only X-Request-ID when middleware is mounted —
    # this minimal app has no middleware so it's empty).
    assert "www-authenticate" not in {
        k.lower() for k in response.headers.keys()
    }, (
        f"404 path must not invent a WWW-Authenticate header; "
        f"headers={dict(response.headers)!r}"
    )
