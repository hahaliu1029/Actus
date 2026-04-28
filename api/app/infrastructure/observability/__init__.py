"""Infrastructure-side observability adapters and contextvars.

PR-S1-1 ships only the minimal ``context`` stub needed by
``domain/external/observability.build_canonical_attributes``. PR-S1-2
extends this package with the full contextvars helper surface
(``set_trace_context`` / ``bind_session_context`` / ``bind_request_context``).
Sprint 2 adds OTel logger / tracer / meter adapters that conform to the
domain ports.
"""
