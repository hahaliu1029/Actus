"""B5 PR-S2-2 / PR-S2-3: graph DI factory.

Single site that knows about both ``app.domain.services.graphs.*`` and
``app.infrastructure.observability.*``. Returns the artifacts the
domain build site needs to opt into observability without taking an
OTel dependency:

- ``build_traced_node_decorator(tracer=None)`` — returns the decorator
  callable to pass as ``build_main_graph(node_decorator=...)``.
- ``build_observability_callbacks(tracer=None, meter=None)`` — returns
  a list of LangChain callback handlers to merge into
  ``cfg["callbacks"]``. PR-S2-2 emitted ``[OtelToolSpanCallback]``;
  PR-S2-3 adds ``OtelLLMMetricsCallback`` (latency histogram + cost
  counter) so a single composition call covers tracer + meter wiring.

All factories accept explicit ports (so tests can inject in-memory
backed instances) and default to ``OtelTracer()`` / ``OtelMeter()``
which read the OTel global providers — same path as
``setup_observability()`` consumers.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.domain.external.observability import MeterPort, TracerPort
from app.infrastructure.observability.otel_llm_metrics import (
    OtelLLMMetricsCallback,
)
from app.infrastructure.observability.otel_meter import OtelMeter
from app.infrastructure.observability.otel_tool_span import OtelToolSpanCallback
from app.infrastructure.observability.otel_tracer import OtelTracer
from app.infrastructure.observability.traced_node import traced_node


def build_traced_node_decorator(
    tracer: TracerPort | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Return the ``traced_node`` decorator pre-bound to a tracer.

    The decorator goes into ``build_main_graph(node_decorator=...)``.
    ``tracer=None`` defaults to ``OtelTracer()`` which reads the OTel
    global tracer — same path as ``setup_observability()`` consumers.
    """
    if tracer is None:
        tracer = OtelTracer()
    return traced_node(tracer)


def build_observability_callbacks(
    tracer: TracerPort | None = None,
    meter: MeterPort | None = None,
) -> list[Any]:
    """Return the observability callback handler list.

    Emits ``[OtelToolSpanCallback, OtelLLMMetricsCallback]``:

    - ``OtelToolSpanCallback`` — one ``tool.<name>`` span per tool call
      with ``tool_args_hash`` (sha256[:16]) + ``tool_args_size``.
    - ``OtelLLMMetricsCallback`` — ``llm.latency_ms`` histogram +
      ``cost_usd_micro`` counter per LLM invocation, attributes
      ``model`` / ``llm_provider`` / ``graph_node``.

    ``tracer=None`` / ``meter=None`` default to ``OtelTracer()`` /
    ``OtelMeter()`` which read the OTel globals.
    """
    if tracer is None:
        tracer = OtelTracer()
    if meter is None:
        meter = OtelMeter()
    return [
        OtelToolSpanCallback(tracer),
        OtelLLMMetricsCallback(meter),
    ]
