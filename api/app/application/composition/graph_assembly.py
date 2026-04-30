"""B5 PR-S2-2: graph DI factory.

Single site that knows about both ``app.domain.services.graphs.*`` and
``app.infrastructure.observability.*``. Returns the artifacts the
domain build site needs to opt into observability without taking an
OTel dependency:

- ``build_traced_node_decorator(tracer=None)`` — returns the decorator
  callable to pass as ``build_main_graph(node_decorator=...)``.
- ``build_observability_callbacks(tracer=None)`` — returns a list of
  LangChain callback handlers to merge into ``cfg["callbacks"]``.
  Currently emits ``[OtelToolSpanCallback]``; PR-S2-3 will add the
  cost meter handler here too.

Both factories accept an explicit ``tracer`` (so tests can inject an
``InMemorySpanExporter``-backed instance) and default to
``OtelTracer()`` which pulls the global ``TracerProvider`` installed
by ``setup_observability()``.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.domain.external.observability import TracerPort
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
) -> list[Any]:
    """Return the observability callback handler list.

    Currently emits ``[OtelToolSpanCallback]``. PR-S2-3 (MeterPort) will
    add a cost callback handler here so a single composition call gives
    the build site everything it needs.
    """
    if tracer is None:
        tracer = OtelTracer()
    return [OtelToolSpanCallback(tracer)]
