"""B5 PR-S2-3: ``MeterPort`` implementation backed by OTel SDK.

``OtelMeter`` satisfies the ``MeterPort`` Protocol declared in
``app/domain/external/observability.py``. Domain code holds the
Protocol; infra adapter wraps an OTel ``opentelemetry.metrics.Meter``
so the domain import graph stays free of OTel.

Implementation notes
--------------------
- The ctor accepts an explicit OTel ``Meter`` for tests (so an
  ``InMemoryMetricReader`` can be wired without touching globals)
  and defaults to ``opentelemetry.metrics.get_meter(name)`` for
  production use after ``setup_observability()``.
- Returned instruments are pass-through OTel objects. Domain code
  treats them as ``Any`` per the Protocol shape. Tests can call
  ``.add()`` / ``.record()`` directly.
- All instrument-creation parameters (unit / description) are
  forwarded verbatim. Defaults: empty string, matching OTel's own
  defaults so a no-arg call doesn't crash if a deployment installs
  a stricter OTel build that rejects ``None``.
"""
from __future__ import annotations

from typing import Any

from opentelemetry import metrics as otel_metrics


class OtelMeter:
    """``MeterPort`` impl wrapping an OTel ``Meter``."""

    __slots__ = ("_meter",)

    def __init__(self, meter: Any | None = None, *, name: str = "actus") -> None:
        if meter is None:
            meter = otel_metrics.get_meter(name)
        self._meter = meter

    def create_counter(
        self, name: str, *, unit: str = "", description: str = ""
    ) -> Any:
        return self._meter.create_counter(
            name=name, unit=unit, description=description
        )

    def create_histogram(
        self, name: str, *, unit: str = "", description: str = ""
    ) -> Any:
        return self._meter.create_histogram(
            name=name, unit=unit, description=description
        )

    def create_up_down_counter(
        self, name: str, *, unit: str = "", description: str = ""
    ) -> Any:
        return self._meter.create_up_down_counter(
            name=name, unit=unit, description=description
        )
