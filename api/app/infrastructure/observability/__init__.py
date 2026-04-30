"""Infrastructure-side observability adapters and contextvars.

PR-S1-1 ships only the minimal ``context`` stub needed by
``domain/external/observability.build_canonical_attributes``. PR-S1-2
extends this package with the full contextvars helper surface
(``set_trace_context`` / ``bind_session_context`` / ``bind_request_context``).
PR-S2-1 adds the OTel SDK bootstrap (``setup_observability``) and the
first ``LoggerPort`` impl (``OtelLogger``). PR-S2-2 adds the
``TracerPort`` impl (``OtelTracer``), the LangGraph node tracing
decorator (``traced_node``), and the LangChain tool-span callback
(``OtelToolSpanCallback``).
"""

from app.infrastructure.observability.init import (
    ObservabilityProviders,
    get_providers,
    setup_observability,
    teardown_observability,
)
from app.infrastructure.observability.otel_logger import OtelLogger
from app.infrastructure.observability.otel_tool_span import OtelToolSpanCallback
from app.infrastructure.observability.otel_tracer import OtelTracer
from app.infrastructure.observability.traced_node import traced_node

__all__ = (
    "ObservabilityProviders",
    "OtelLogger",
    "OtelToolSpanCallback",
    "OtelTracer",
    "get_providers",
    "setup_observability",
    "teardown_observability",
    "traced_node",
)
