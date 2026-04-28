"""Contextvars-backed ``TraceContext`` carrier.

PR-S1-1 ships only the read path so that
``domain.external.observability.build_canonical_attributes`` can resolve
its lazy import and the helper test (`test_build_canonical_attributes_helper.py`)
can monkey-patch the getter. PR-S1-2 will extend this module with
``set_trace_context`` / ``bind_session_context`` / ``bind_request_context``
plus the ``ObservabilityMiddleware`` wiring (PR-S1-5).

The ``ContextVar`` lives at module scope (not inside a class) because
asyncio task scheduling reads contextvars from the binding scope at task
creation time — class-attribute storage would introduce a per-instance
binding that breaks propagation across spawned tasks.
"""
from __future__ import annotations

from contextvars import ContextVar

from app.domain.external.observability import TraceContext

_trace_context_var: ContextVar[TraceContext | None] = ContextVar(
    "actus_trace_context",
    default=None,
)


def get_trace_context() -> TraceContext | None:
    """Return the current request-scoped ``TraceContext`` or ``None``.

    Returns ``None`` when called outside a request scope (CLI scripts,
    startup paths, unit tests without context binding). Callers that
    require a context (every span emit, every JSONL write) treat this
    as the contract: ``build_canonical_attributes`` generates a fallback
    trace_id / request_id when the getter returns ``None`` so the
    canonical contract's NEVER-null guarantee still holds.
    """
    return _trace_context_var.get()
