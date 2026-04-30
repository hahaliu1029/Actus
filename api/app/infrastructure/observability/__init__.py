"""Infrastructure-side observability adapters and contextvars.

PR-S1-1 ships only the minimal ``context`` stub needed by
``domain/external/observability.build_canonical_attributes``. PR-S1-2
extends this package with the full contextvars helper surface
(``set_trace_context`` / ``bind_session_context`` / ``bind_request_context``).
PR-S2-1 adds the OTel SDK bootstrap (``setup_observability``) and the
first ``LoggerPort`` impl (``OtelLogger``).
"""

from app.infrastructure.observability.init import (
    ObservabilityProviders,
    get_providers,
    setup_observability,
    teardown_observability,
)
from app.infrastructure.observability.otel_logger import OtelLogger

__all__ = (
    "ObservabilityProviders",
    "OtelLogger",
    "get_providers",
    "setup_observability",
    "teardown_observability",
)
