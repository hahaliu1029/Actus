"""B5 PR-S3-2: decision_trace helper.

Records point-in-time **decisions** (SmartApprove / Recovery / future
PermissionEngine) as OTel **span events** on the currently-active
span. Events are the OTel canonical way to attach discrete signals
within an existing span — they:

- Don't fragment the native trace tree (no new child spans).
- Carry their own attribute bundle, queryable in Phoenix / Jaeger.
- Become timeline pins on the parent span so operators can see
  ``smart_approve denied tool=shell_execute reason=...`` against the
  exact tool span that triggered it.

Spec anchor: ``TODO2.md`` #13 Sprint 3 PR-S3-2 — *"decision_trace
（PermissionEngine / SmartApprove / Recovery 决策点写 span 属性）"*.

API
---

``record_decision(name, outcome, *, reason=None, attrs=None)``

- ``name`` — decision-point identifier (``smart_approve`` /
  ``risk_assessor`` / ``recovery`` / ``permission_engine``). Becomes
  the event name as ``decision.<name>``.
- ``outcome`` — the decision verb (``approve`` / ``deny`` /
  ``escalate`` / ``allow`` / ``block`` / etc.). Recorded as
  ``decision_outcome`` event attribute.
- ``reason`` — optional human-readable rationale. Recorded as
  ``decision_reason`` (canonical attribute).
- ``attrs`` — optional extra attributes; filtered through the
  canonical whitelist + non-None filter so non-canonical fields
  don't leak.

When no span is currently active (CLI / startup / unit tests with
no middleware-installed root), the call is a no-op:
``trace.get_current_span()`` returns the OTel ``INVALID_SPAN``
sentinel whose ``add_event`` does nothing.

Privacy
-------
Same canonical-whitelist discipline as ``OtelToolSpanCallback``: we
filter caller-supplied extras through ``CANONICAL_ATTRIBUTES`` and
drop ``None`` values. Raw tool args / LLM prompts MUST NOT be
passed in — pass content hashes (``tool_args_hash``) or stable
categorical labels instead. ``reason`` strings are caller-controlled;
keep them short and free of secret material (the SmartApprove
LLM-rationale, for example, is the LLM's own one-line summary, not
the input it was given).
"""
from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace as otel_trace

from app.domain.external.observability import (
    CANONICAL_ATTRIBUTES,
    build_canonical_attributes,
)


logger = logging.getLogger(__name__)


# ``decision_outcome`` is the verb bucket (denied / approved /
# escalated). ``decision_reason`` is in the canonical contract; we
# expose it as an attribute key via ``build_canonical_attributes``
# below so whitelist enforcement covers it.
_DECISION_OUTCOME_KEY = "decision_outcome"


def record_decision(
    name: str,
    outcome: str,
    *,
    reason: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Record a decision as an event on the currently-active span.

    Safe to call from any context: when no span is active (CLI, unit
    tests without middleware), the underlying OTel call is a no-op
    (``INVALID_SPAN`` sentinel). When OTel is not installed (default
    no-op deployment), same behaviour — the SDK's ``ProxyTracer`` /
    NonRecording paths swallow it.

    Best-effort contract (PR-S3-2 reviewer round-2 P2 fix)
    ------------------------------------------------------
    EVERY observability failure becomes a debug log. The whole
    canonical-attr build + event emit runs under one
    ``try/except Exception``. Reasoning: a malformed
    ``TraceContext`` on the contextvar causes
    ``build_canonical_attributes → validate_attributes`` to raise
    ``ValueError``. Without the wrap, that ``ValueError`` escapes
    into ``SmartApprove.evaluate`` and gets misclassified by the
    outer ``except Exception`` as ``llm_error`` — silently turning
    legitimate ``approve``/``deny`` outcomes into ``escalate`` and
    changing **agent behaviour**. Recovery's ``_emit_recovery_event``
    similarly aborts before the existing telemetry hook fires.

    The discipline is the same as the OTel SDK's own callback
    handlers: telemetry MUST NOT crash the path it observes.
    """
    if not name or not outcome:
        # Defensive: callers MUST supply both. We don't raise (a
        # buggy caller shouldn't crash an LLM step), but DO log so
        # the gap is visible in dev.
        logger.debug(
            "record_decision: missing name=%r or outcome=%r", name, outcome
        )
        return

    try:
        # ``build_canonical_attributes`` reads the contextvar
        # (graph_node / step_id / trace_id / request_id / event_id /
        # session_id / ...) so the event carries the same canonical
        # join keys as the parent span. Caller-supplied attrs go
        # through the same whitelist + None-filter pass as
        # ``OtelToolSpanCallback`` to keep non-canonical fields out.
        canonical = build_canonical_attributes(decision_reason=reason)
        event_attrs: dict[str, Any] = {
            _DECISION_OUTCOME_KEY: outcome,
        }
        for key, value in canonical.items():
            if value is None:
                continue
            event_attrs[key] = value
        if attrs:
            for key, value in attrs.items():
                if key not in CANONICAL_ATTRIBUTES:
                    continue
                if value is None:
                    continue
                # Caller-supplied wins over canonical-from-context —
                # the decision site has the most specific knowledge
                # (e.g. ``tool_name`` set explicitly when SmartApprove
                # evaluates a tool call regardless of what's bound to
                # contextvar).
                event_attrs[key] = value

        span = otel_trace.get_current_span()
        span.add_event(f"decision.{name}", attributes=event_attrs)
    except Exception:
        # Telemetry MUST NOT crash the caller. Log at DEBUG so the
        # path stays visible without producing per-call ERROR noise
        # in production (a corrupt contextvar would fire on every
        # decision until restart — INFO/WARN would flood).
        logger.debug(
            "record_decision: observability emit failed for name=%r",
            name,
            exc_info=True,
        )
