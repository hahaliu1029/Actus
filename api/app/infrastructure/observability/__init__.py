"""Infrastructure-side observability adapters and contextvars.

PR-S1-1 ships only the minimal ``context`` stub needed by
``domain/external/observability.build_canonical_attributes``. PR-S1-2
extends this package with the full contextvars helper surface
(``set_trace_context`` / ``bind_session_context`` / ``bind_request_context``).
PR-S2-1 adds the OTel SDK bootstrap (``setup_observability``) and the
first ``LoggerPort`` impl (``OtelLogger``). PR-S2-2 adds the
``TracerPort`` impl (``OtelTracer``), the LangGraph node tracing
decorator (``traced_node``), and the LangChain tool-span callback
(``OtelToolSpanCallback``). PR-S2-3 adds the ``MeterPort`` impl
(``OtelMeter``) and the LangChain LLM metrics callback
(``OtelLLMMetricsCallback`` — ``llm.latency_ms`` histogram +
``cost_usd_micro`` counter).
"""

from app.infrastructure.observability.decision_trace import record_decision
from app.infrastructure.observability.init import (
    ObservabilityProviders,
    get_providers,
    setup_observability,
    teardown_observability,
)
from app.infrastructure.observability.otel_llm_metrics import (
    OtelLLMMetricsCallback,
)
from app.infrastructure.observability.otel_logger import OtelLogger
from app.infrastructure.observability.otel_meter import OtelMeter
from app.infrastructure.observability.otel_tool_span import OtelToolSpanCallback
from app.infrastructure.observability.otel_tracer import OtelTracer
from app.infrastructure.observability.traced_node import traced_node

__all__ = (
    "ObservabilityProviders",
    "OtelLLMMetricsCallback",
    "OtelLogger",
    "OtelMeter",
    "OtelToolSpanCallback",
    "OtelTracer",
    "get_providers",
    "record_decision",
    "setup_observability",
    "teardown_observability",
    "traced_node",
)
