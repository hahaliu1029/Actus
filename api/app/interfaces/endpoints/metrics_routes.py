"""B5 PR-S3-3: ``/api/v1/metrics`` Prometheus scrape endpoint.

Internal use only — exposes the OTel SDK's accumulated metric data
in Prometheus exposition format (``text/plain; version=0.0.4``) so a
sidecar Prometheus / VictoriaMetrics / OpenObserve can scrape it.
This is a pull-based bridge: ``setup_observability`` registers a
``PrometheusMetricReader`` (via
``infrastructure/observability/init.py``) when ``METRICS_ENDPOINT_TOKEN``
is set, and that reader's ``_CustomCollector`` is auto-registered with
``prometheus_client.REGISTRY``. ``generate_latest(REGISTRY)`` renders
the snapshot at request time.

Auth contract
-------------
- ``METRICS_ENDPOINT_TOKEN`` empty (default) → endpoint behaves as
  if the route is not mounted: GET returns ``404`` with FastAPI's
  standard ``{"detail":"Not Found"}`` body — the exact shape an
  unauthenticated probe would see for any non-existent path. Default
  deployments have zero discoverability for this endpoint, zero
  ``prometheus_client`` registration overhead.
- ``METRICS_ENDPOINT_TOKEN`` set + missing / malformed / wrong
  ``Authorization: Bearer <token>`` → 401 + ``WWW-Authenticate: Bearer``.
  Constant-time compare via ``secrets.compare_digest`` on the
  UTF-8-encoded bytes (NOT the raw ``str``) — ``compare_digest``
  rejects non-ASCII ``str`` arguments with ``TypeError``, which
  would otherwise turn a non-ASCII configured token or non-ASCII
  bearer header into an HTTP 500 instead of the spec's 401.
- Right token → 200 + Prometheus exposition body.

The endpoint is hidden from OpenAPI (``include_in_schema=False``)
because Prometheus scraping is operational, not user-facing — leaking
its existence in ``/docs`` runs counter to the spec's
"内部使用，不暴露给终端用户" constraint.

No rate limiting — the token IS the security boundary, and Prometheus
scrapers hit at fixed intervals (typically 15-60s) which a sane rate
limit would either throttle or be useless against. If a deployment
needs IP allowlisting it belongs at the reverse proxy layer.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response as FastAPIResponse

from core.config import get_settings

router = APIRouter(prefix="/v1/metrics", tags=["可观测性"])


def _extract_bearer_token(request: Request) -> str | None:
    """Pull the bearer token from ``Authorization: Bearer <token>``.

    Returns ``None`` if the header is missing, has the wrong scheme,
    or the token portion is empty / whitespace-only. The caller is
    responsible for distinguishing "no token presented" from
    "wrong token presented" — both produce 401 in this endpoint, so
    the helper collapses them.
    """
    auth = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


@router.get(
    "",
    summary="Prometheus 指标抓取（内部使用）",
    description=(
        "Prometheus exposition format. 仅在 ``METRICS_ENDPOINT_TOKEN`` "
        "设置时启用；要求 ``Authorization: Bearer <token>`` 鉴权。"
    ),
    include_in_schema=False,
)
async def get_metrics(request: Request) -> FastAPIResponse:
    settings = get_settings()
    expected = settings.metrics_endpoint_token
    if not expected:
        # Endpoint disabled — pretend the route doesn't exist.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    presented = _extract_bearer_token(request)
    # ``secrets.compare_digest`` raises ``TypeError`` when either
    # ``str`` argument contains non-ASCII codepoints, which would
    # turn a misconfigured non-ASCII token (or a bearer header
    # carrying non-ASCII bytes) into an HTTP 500 — leaking server
    # behaviour and breaking the 401 contract this endpoint
    # advertises. Encode to UTF-8 bytes first; ``compare_digest``
    # handles bytes-like values uniformly and remains constant-time.
    if presented is None or not secrets.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Lazy import so deployments with the token unset never pay the
    # ``prometheus_client`` import cost. By the time we reach this
    # branch, ``setup_observability`` has registered a
    # ``PrometheusMetricReader`` whose ``_CustomCollector`` is in
    # ``REGISTRY`` — ``generate_latest`` walks all registered
    # collectors and produces the exposition payload.
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        generate_latest,
    )

    body = generate_latest(REGISTRY)
    return FastAPIResponse(content=body, media_type=CONTENT_TYPE_LATEST)
