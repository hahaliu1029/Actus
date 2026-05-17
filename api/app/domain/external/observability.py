"""Domain-side observability protocol and canonical attribute contract.

B5 Sprint 1 (PR-S1-1) deliverable. Defines the cross-data-stream join-key
contract (`CANONICAL_ATTRIBUTES`) plus the minimal port shapes
(`LoggerPort` / `TracerPort` / `MeterPort`) consumed by the domain layer
without taking an OpenTelemetry SDK dependency. Sprint 2 wires concrete
OTel adapters that conform to these ports.

Contract change rule
--------------------
``CANONICAL_ATTRIBUTES`` and ``REQUIRED_ATTRIBUTES`` are FROZEN. Any
field add/remove/type change goes through spec review (see the
``Canonical Attribute Contract v1`` section of the B5 design doc). The
PR that mutates this tuple must also update ``docs/observability-contract.md``
and the ``validate_attributes`` test surface.

Domain purity
-------------
Only ``__future__`` and stdlib imports live at module top-level. The
``build_canonical_attributes`` helper does a *lazy* import of
``app.infrastructure.observability.context`` inside its body so that the
domain module's import graph stays free of infrastructure dependencies.
A unit test (``test_validate_attributes_domain_purity.py``) AST-scans
this file's top-level imports to enforce the rule.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Request-scoped trace correlation carrier.

    Bound to a contextvar by ``ObservabilityMiddleware`` (PR-S1-5) and read
    by every emit site through ``build_canonical_attributes``. Frozen so
    that downstream code cannot mutate the live context after binding.
    """

    trace_id: str
    request_id: str
    event_id: str
    session_id: str | None = None
    user_id_hash: str | None = None
    graph_node: str | None = None
    step_id: str | None = None


class LoggerPort(Protocol):
    """Minimal logger shape — no OTel dependency.

    Sprint 1 implementation: a stdlib ``logging.Logger`` adapter.
    Sprint 2 implementation: an OTel logging-bridge adapter that emits the
    same call sites as OTel log records. Domain code uses this Protocol
    so swapping the implementation does not require domain edits.
    """

    def info(self, message: str, *, extra: dict[str, Any] | None = ...) -> None: ...

    def warning(self, message: str, *, extra: dict[str, Any] | None = ...) -> None: ...

    def error(self, message: str, *, extra: dict[str, Any] | None = ...) -> None: ...

    def exception(self, message: str, *, extra: dict[str, Any] | None = ...) -> None: ...


class TracerPort(Protocol):
    """Minimal tracer shape for span creation.

    Sprint 1: no-op stub. Sprint 2: OTel Tracer adapter. Returned span
    objects are typed as ``Any`` so the Protocol does not leak the OTel
    span class into domain.
    """

    def start_span(
        self, name: str, *, attributes: dict[str, Any] | None = ...
    ) -> Any: ...

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = ...
    ) -> Any: ...


class MeterPort(Protocol):
    """Minimal meter shape for instrument creation.

    Sprint 1: no-op stub. Sprint 2: OTel Meter adapter. Returned
    instrument objects are typed as ``Any`` for the same isolation
    reason as ``TracerPort``.
    """

    def create_counter(
        self, name: str, *, unit: str = ..., description: str = ...
    ) -> Any: ...

    def create_histogram(
        self, name: str, *, unit: str = ..., description: str = ...
    ) -> Any: ...

    def create_up_down_counter(
        self, name: str, *, unit: str = ..., description: str = ...
    ) -> Any: ...


CANONICAL_ATTRIBUTES: tuple[str, ...] = (
    "trace_id",
    "request_id",
    "session_id",
    "user_id_hash",
    "graph_node",
    "step_id",
    "tool_name",
    "tool_call_id",
    "tool_args_hash",
    "tool_args_size",
    "llm_provider",
    "model",
    "attempt_ix",
    "event_id",
    "decision_reason",
    # PE-0 (2026-05-14): PermissionEngine decision-pipeline attrs
    "decision_stage",        # policy_get | stage_p1_reader | stage_p2_smart | decision_final | session_mode_check
    "tool_source",           # native | skill | mcp | a2a
    "confirmation_id_hash",  # sha256(confirmation_id)[:16]
    "session_mode",          # SessionStatus.value
)


REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "trace_id",
    "request_id",
    "event_id",
)


# v1 contract format constraints for required attrs.
#
# - ``trace_id``: 32 lowercase hex chars. Matches both the Sprint-1
#   ``uuid.uuid4().hex`` fallback and Sprint-2 OTel-native trace IDs
#   (16 random bytes rendered hex).
# - ``request_id`` / ``event_id``: UUIDv4 with dashes (8-4-4-4-12,
#   version nibble = ``4``, variant nibble in ``8|9|a|b``).
#
# ``ObservabilityMiddleware`` (PR-S1-5) is the boundary that normalises
# the inbound ``X-Request-ID`` header to UUIDv4 (or generates fresh) so
# that by the time ``validate_attributes`` sees a request_id, it is
# always canonical.
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_REQUIRED_ATTR_PATTERNS: dict[str, re.Pattern[str]] = {
    "trace_id": _TRACE_ID_RE,
    "request_id": _UUID4_RE,
    "event_id": _UUID4_RE,
}


def validate_attributes(d: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys and enforce the v1 join-key contract.

    Required attrs (``trace_id`` / ``request_id`` / ``event_id``) must:

    - be present in ``d``
    - be a non-empty ``str`` (NEVER ``None``)
    - match the v1 format pattern:
        * ``trace_id``: 32 lowercase hex chars
        * ``request_id``: UUIDv4 with dashes
        * ``event_id``: UUIDv4 with dashes

    Unknown keys are dropped silently so that ad-hoc fields cannot leak
    into downstream JSONL or OTel span attributes.

    ``attempt_ix`` is OPTIONAL: ``None`` means "no attempt context" (the
    initial LLM call before any Recovery retry has fired); ``0`` means
    "first retry has occurred". These are semantically distinct — never
    default a missing ``attempt_ix`` to ``0``. The dict-comprehension
    filter preserves a key only when the caller explicitly provided it.

    Raises:
        ValueError: a required key is missing, ``None``, empty, or fails
            the v1 format check.
        TypeError: a required value is not a ``str``.
    """
    missing = [k for k in REQUIRED_ATTRIBUTES if k not in d]
    if missing:
        raise ValueError(f"missing required canonical attrs: {missing}")

    for key, pattern in _REQUIRED_ATTR_PATTERNS.items():
        value = d[key]
        if value is None:
            raise ValueError(
                f"required canonical attr {key!r} must not be None"
            )
        if not isinstance(value, str):
            raise TypeError(
                f"required canonical attr {key!r} must be str, "
                f"got {type(value).__name__}"
            )
        if not value:
            raise ValueError(
                f"required canonical attr {key!r} must not be empty"
            )
        if not pattern.fullmatch(value):
            raise ValueError(
                f"required canonical attr {key!r} fails v1 format check: "
                f"got {value!r}, expected pattern {pattern.pattern!r}"
            )

    return {k: v for k, v in d.items() if k in CANONICAL_ATTRIBUTES}


def build_canonical_attributes(
    *,
    step_id: str | None = None,
    attempt_ix: int | None = None,
    graph_node: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_args_hash: str | None = None,
    tool_args_size: int | None = None,
    llm_provider: str | None = None,
    model: str | None = None,
    decision_reason: str | None = None,
    # PE-0 (2026-05-14): PermissionEngine decision-pipeline attrs
    decision_stage: str | None = None,
    tool_source: str | None = None,
    confirmation_id_hash: str | None = None,
    session_mode: str | None = None,
) -> dict[str, Any]:
    """Build a contract-conformant attribute dict for an emit site.

    Reads ``trace_id`` / ``request_id`` / ``session_id`` / ``user_id_hash``
    from the current ``TraceContext`` (bound by ``ObservabilityMiddleware``
    or ``bind_session_context``). Caller-supplied locals (``step_id`` /
    ``graph_node``) override the contextvar values when both are present
    — emit-site precision wins over middleware-bound coarse defaults.
    Generates a fresh ``event_id`` per call.

    Falls back to a generated ``trace_id`` / ``request_id`` when no
    context is bound (CLI scripts, startup paths, direct unit tests). The
    contract requires ``trace_id`` / ``request_id`` / ``event_id`` to be
    NEVER null, so empty-string fallbacks would silently violate
    ``validate_attributes``; we generate uuid4 instead.
    """
    from app.infrastructure.observability.context import get_trace_context

    ctx = get_trace_context()
    if ctx is not None:
        trace_id = ctx.trace_id
        request_id = ctx.request_id
        ctx_session_id = ctx.session_id
        ctx_user_id_hash = ctx.user_id_hash
        ctx_graph_node = ctx.graph_node
        ctx_step_id = ctx.step_id
    else:
        trace_id = uuid.uuid4().hex
        request_id = str(uuid.uuid4())
        ctx_session_id = None
        ctx_user_id_hash = None
        ctx_graph_node = None
        ctx_step_id = None

    return validate_attributes(
        {
            "trace_id": trace_id,
            "request_id": request_id,
            "session_id": ctx_session_id,
            "user_id_hash": ctx_user_id_hash,
            "graph_node": graph_node if graph_node is not None else ctx_graph_node,
            "step_id": step_id if step_id is not None else ctx_step_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_args_hash": tool_args_hash,
            "tool_args_size": tool_args_size,
            "llm_provider": llm_provider,
            "model": model,
            "attempt_ix": attempt_ix,
            "event_id": str(uuid.uuid4()),
            "decision_reason": decision_reason,
            # PE-0 (2026-05-14): PermissionEngine decision-pipeline attrs
            "decision_stage": decision_stage,
            "tool_source": tool_source,
            "confirmation_id_hash": confirmation_id_hash,
            "session_mode": session_mode,
        }
    )
