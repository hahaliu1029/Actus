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

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping

if TYPE_CHECKING:
    from app.domain.services.permission.engine import PermissionEngine
    from app.domain.services.session.session_state_machine import SessionStateMachine

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


def build_session_state_machine(
    *,
    uow_factory: Any,
    redis: Any = None,
    event_publisher: Any = None,
) -> "SessionStateMachine":
    """Build a DefaultSessionStateMachine.

    Parameters
    ----------
    uow_factory:
        Callable that returns an IUnitOfWork context manager.  Passed through
        to DefaultSessionStateMachine for per-call DB access.
    redis:
        Optional raw redis.asyncio.Redis client reserved for future hot-path
        caching (A4-0 wires the real value; PE-0 passes None).
    event_publisher:
        Optional SseEventPublisher for SessionModeChangedEvent.  None →
        DefaultSessionStateMachine falls back to its internal _NoopPublisher.
    """
    from app.domain.services.session.default_state_machine import (
        DefaultSessionStateMachine,
    )

    return DefaultSessionStateMachine(
        uow_factory=uow_factory,
        redis=redis,
        event_publisher=event_publisher,
    )


def build_permission_engine(
    *,
    uow_factory: Any,
    writer: Any,
    queue: Any,
    session_machine: Any,
    reader: Any,
    summary_llm: Any,
    smart_approve_timeout_seconds: float = 30.0,
    smart_approve_enabled: bool = True,
    smart_approve_medium_only: bool = False,
    confirmation_timeout_seconds: int = 300,  # P2#3: deadline for ConfirmationQueue entries
    decision_recorder: Any = None,  # P3#1: OTel decision recorder callable
    sources: "Mapping[str, Any] | None" = None,
) -> "PermissionEngine":
    """Build a DefaultPermissionEngine wired with SmartApproveProvider.

    Parameters
    ----------
    uow_factory:
        Per-call UoW factory for policy lookup (C-R5-P1).
    writer:
        ApprovalStateWriter instance (R5 CS4 single writer).
    queue:
        ConfirmationQueue instance (already backed by raw Redis client).
    session_machine:
        SessionStateMachine instance; used only for read-only
        get_mode_with_revision (INV-2 enforces no mutator calls from PE).
    reader:
        ApprovalStateReader instance for grant lookup (Stage P.1).
    summary_llm:
        BaseChatModel used by SmartApprove for LLM-assisted risk evaluation.
        When None the escalation_registry is built with an empty dict so
        Stage P.2 is skipped and the flow falls directly to Asked enqueue.
    smart_approve_timeout_seconds:
        asyncio.wait_for timeout for the SmartApprove LLM call.
    smart_approve_enabled:
        P1#5: When False, SmartApproveProvider is NOT registered regardless of
        whether summary_llm is available.  Mirrors tool_confirmation.smart_approve_enabled.
        Defaults to True for backward-compat when the config field is absent.
    smart_approve_medium_only:
        When True, SmartApproveProvider will skip LLM evaluation for HIGH-risk
        tool calls and fall directly through to Asked (user confirmation required).
        This mirrors the legacy tool_confirmation behaviour where HIGH always
        requires explicit human confirmation when medium_only is enabled.
    confirmation_timeout_seconds:
        P2#3: deadline for ConfirmationQueue entries (seconds from now).
        Must match the ToolConfirmationEvent timeout sent to the frontend so
        the backend sweep and frontend countdown stay consistent.
        Mirrors ``tool_confirmation.timeout_seconds`` from AppConfig.
    decision_recorder:
        P3#1: Optional OTel-backed callable for recording PE decision events.
        When provided, forwarded to DefaultPermissionEngine so that
        ``_record_decision`` emits canonical OTel attributes (decision_stage,
        etc.) on every stage transition.  Callers should pass
        ``build_decision_recorder()`` here.  Defaults to None (no-op inside PE).
    sources:
        PE-1 §2.6: Mapping of tool_source → PermissionSource. Must contain
        at least every entry in PE_SUPPORTED_SOURCES_AFTER_PE_1. Caller
        constructs (typically NativeSource() + SkillSource(refresher, redis)
        wired with the per-task Redis client).
    """
    from app.domain.services.permission.default_engine import DefaultPermissionEngine
    from app.domain.services.permission.smart_approve_provider import SmartApproveProvider

    # PE-1 §2.6: validation is caller-driven (see _create_task + preflight_resume
    # late-registration). Build accepts partial source maps so callers can
    # register skill_source after AgentTaskRunner is constructed (skill_tool
    # lives on the runner, not on AgentService at PE build time).

    escalation_registry: dict[str, Any] = {}
    # P1#5: gate SmartApproveProvider on the config flag (not just summary_llm presence).
    # When smart_approve_enabled is False, escalation_registry stays empty → Stage P.2 is
    # skipped and all high/medium-risk tool calls go directly to Asked (user confirmation).
    # This prevents the LLM from auto-approving/denying when the operator disables SmartApprove.
    if smart_approve_enabled and summary_llm is not None:
        from app.domain.services.smart_approve import SmartApprove  # local import

        smart_approve = SmartApprove(llm=summary_llm)
        # P1#4: pass medium_only so HIGH-risk tools bypass LLM evaluation when
        # the operator has configured smart_approve_medium_only=True.
        provider = SmartApproveProvider(
            smart_approve,
            timeout_seconds=smart_approve_timeout_seconds,
            medium_only=smart_approve_medium_only,
        )
        escalation_registry[provider.name] = provider

    return DefaultPermissionEngine(
        uow_factory=uow_factory,
        writer=writer,
        queue=queue,
        session_machine=session_machine,
        reader=reader,
        escalation_registry=escalation_registry,
        confirmation_timeout_seconds=confirmation_timeout_seconds,
        decision_recorder=decision_recorder,  # P3#1: forward to PE for OTel emit
        sources=sources,
    )


def build_decision_recorder() -> Callable[..., None]:
    """Return the OTel-backed ``decision_recorder`` callable.

    PR-S3-2 reviewer round-2 P3: domain decision points
    (``SmartApprove`` and any future ``PermissionEngine``) take an
    optional ``decision_recorder: Callable[..., None] | None`` so they
    don't import ``app.infrastructure.observability`` directly. The
    composition layer is the single site that knows about the OTel
    helper — it returns the ``record_decision`` function as the
    injected callable. Tests can pass a fake callable directly, or
    leave the recorder ``None`` to suppress emits.

    Recovery records decisions inside the infrastructure-side
    ``ActusRecoveryChatModel`` wrapper, so it has no domain-import
    concern and doesn't need this factory.

    Import is intentionally lazy (inside function body) so that
    ``mock.patch("app.infrastructure.observability.decision_trace.record_decision")``
    intercepts correctly even after this composition module has already
    been imported at test collection time. A module-level import would
    cache the reference and bypass the patch.
    """
    from app.infrastructure.observability.decision_trace import record_decision  # noqa: PLC0415

    return record_decision


def validate_pe_source_registry(sources: "Mapping[str, Any]") -> None:
    """Raise PermissionConfigurationError when the registered ``sources``
    Mapping does not cover the entries that ``is_pe_enabled_for_source``
    will gate-pass at runtime.

    PE-1 §2.6: defense-in-depth at DI/factory time. The CI invariant test
    is the primary gate; this runtime check catches operator missteps
    (e.g., forgetting to wire SkillSource in a custom factory) before any
    tool call reaches PE and triggers UnsupportedSource silently.
    """
    from app.domain.services.permission.errors import (
        PermissionConfigurationError,
    )
    from app.domain.services.permission.sources import (
        PE_SUPPORTED_SOURCES_AFTER_PE_1,
    )

    missing = PE_SUPPORTED_SOURCES_AFTER_PE_1 - set(sources or {})
    if missing:
        raise PermissionConfigurationError(
            "PE_SUPPORTED_SOURCES_AFTER_PE_1 claims "
            f"{sorted(PE_SUPPORTED_SOURCES_AFTER_PE_1)} but DI registered "
            f"only {sorted((sources or {}).keys())}. Missing: {sorted(missing)}."
        )
