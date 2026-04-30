"""B5 PR-S2-2: ``TracerPort`` implementation backed by OTel SDK.

``OtelTracer`` satisfies the ``TracerPort`` Protocol declared in
``app/domain/external/observability.py``. Domain code (graph nodes,
tool dispatchers, application services) holds the Protocol and gets
``OtelTracer`` injected at composition time; the infra → domain
direction is one-way so the domain import graph stays free of OTel.

Implementation notes
--------------------
- The ctor takes an explicit OTel ``Tracer`` for tests (so an
  ``InMemorySpanExporter`` can be wired without touching globals) and
  defaults to ``opentelemetry.trace.get_tracer(name)`` for production
  use after ``setup_observability()``.
- ``None`` attribute values are dropped at the boundary because the
  OTel SDK silently rejects them but downstream JSONL / Prometheus
  exporters could surface ``"step_id": null`` lines — emit only present
  values to keep the wire format clean across exporters. This matches
  ``OtelLogger`` extra-filtering semantics for parity.
- ``start_span`` returns a raw OTel ``Span``; ``start_as_current_span``
  returns an OTel context manager that activates the span and tears it
  down on exit. Both pass through the OTel SDK behaviour verbatim so
  parent-child relationships (and cross-task propagation via
  ``opentelemetry.context``) remain unchanged.
"""
from __future__ import annotations

from typing import Any

from opentelemetry import trace as otel_trace


class OtelTracer:
    """``TracerPort`` impl wrapping an OTel ``Tracer``."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Any | None = None, *, name: str = "actus") -> None:
        if tracer is None:
            tracer = otel_trace.get_tracer(name)
        self._tracer = tracer

    def start_span(
        self, name: str, *, attributes: dict[str, Any] | None = None
    ) -> Any:
        return self._tracer.start_span(
            name=name,
            attributes=_filter_attributes(attributes),
        )

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None
    ) -> Any:
        return self._tracer.start_as_current_span(
            name=name,
            attributes=_filter_attributes(attributes),
        )


def _filter_attributes(
    attrs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if attrs is None:
        return None
    return {k: v for k, v in attrs.items() if v is not None}
