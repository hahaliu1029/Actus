"""react_graph — inner ReAct loop as a LangGraph StateGraph.

Replaces BaseAgent.invoke() and ReActAgent.execute_step().
Nodes: pre_llm_node, llm_node, tool_node
Edges: START → pre_llm_node → llm_node → route_after_llm → (tool_node → pre_llm_node) | END

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.3-4.4
"""

from __future__ import annotations

import asyncio
import base64 as _b64
import json
import logging
from typing import Any, Callable, Literal, NamedTuple, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from core.config import get_settings  # N1 — read sandbox_default_cwd for AST validator

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.file_processor import FileProcessResult
from app.domain.models.app_config import AgentConfig
from app.domain.models.event import (
    MessageEvent,
    ToolConfirmationEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    FileBlock,
    FilePayload,
    ImageUrlBlock,
    ImageUrlPayload,
    MultimodalPayload,
    Passthrough,
    ToolArtifact,
    ToolOutcome,
    TOOL_OUTCOME_ADAPTER,
    ToolResult,
)
from app.domain.services.json_envelope import unwrap_message_envelope
from app.domain.services.permission.child_scope_gate import (
    ChildScopeGate,
    ScopeDecision,
    extract_target_path,
)
from app.domain.services.permission.child_scope_violation import ChildScopeViolation
from app.domain.services.permission.errors import (
    PEInfrastructureUnavailable,
    PolicyConflict,
    SessionModeViolation,
    UnsupportedSource,
)
from app.domain.services.permission.sources import (
    is_pe_eligible_tool_source,
    is_pe_enabled_for_source,
)
from app.domain.services.risk_assessor import RiskAssessor
from app.domain.services.tools.tool_source_resolver import (
    ToolSource,
    ToolSourceUnknownError,
    resolve_tool_source,
)

from .message_utils import truncate_tool_content
from .state import ReactGraphState

if TYPE_CHECKING:
    from app.domain.models.app_config import ToolRuntimeConfig

    from .context_assembler import ContextAssembler

logger = logging.getLogger(__name__)

# Max ReAct iterations to prevent infinite loops
MAX_ITERATIONS = 30

from app.domain.external.file_processor import MAX_FILE_VIEW_IMAGES as _MAX_FILE_VIEW_IMAGES


# ===========================================================================
# C2 PR-4 §8.4 — Cooperative cancellation checkpoints
# ===========================================================================
# Coordinator children run inside this same react_graph. The parent cancels
# them by setting an ``asyncio.Event`` passed through
# ``config["configurable"]["cancel_event"]``. At well-defined checkpoints
# the nodes call ``_should_cancel(config)`` and, on True, raise
# ``CancelledByEventError`` with the checkpoint name. The exception unwinds
# the node, escapes the graph (via LangGraph's normal exception propagation),
# and is caught by ``CoordinatorChildRunner.run_work_unit`` which routes
# to ``_finalize_by_stop_reason`` (PR-4 Task 4.7).
#
# Active checkpoints in react_graph (5 of the 6 spec'd here; #4 deferred):
#   #2 react_loop_entry   — pre_llm_node top
#   #3 llm_node_entry     — llm_node top
#   #4 llm_chunk_boundary — DEFERRED: live llm_node uses ``ainvoke`` (atomic).
#                           When llm_node is refactored to ``astream``, wire
#                           the check inside the ``async for chunk in ...``
#                           loop. The string ``llm_chunk_boundary`` is kept
#                           in this docstring so a grep / a contract test
#                           (test_chunk_boundary_deferred_until_streaming)
#                           can detect a silent regression.
#   #5 llm_return         — llm_node bottom, before ``return {...}``
#   #6 tool_node_entry    — tool_node top
#   #7 tool_node_return   — tool_node bottom, before final ``return Command``
# Spec checkpoints #1 (worker start) + #8 (artifact upload pre-publish) live
# in ``CoordinatorChildRunner`` outside react_graph.
#
# For the default (non-coordinator) execution path, ``configurable`` has no
# ``cancel_event`` key, so ``_should_cancel`` returns False and every
# checkpoint is a no-op. This keeps the legacy / subagent_research / root
# session paths byte-identical to pre-PR-4 behavior.


class CancelledByEventError(Exception):
    """[C2 PR-4 §8.4] Raised at a react_graph checkpoint when the
    coordinator-side cancel_event is already set.

    The exception message carries the checkpoint name (e.g.
    ``CancelledByEventError("react_loop_entry")``) so the
    ``CoordinatorChildRunner`` finalizer can attribute the cancel point in
    its grievance summary."""


def _should_cancel(config: Any) -> bool:
    """Return True iff ``config["configurable"]["cancel_event"]`` is an
    event-like object whose ``is_set()`` currently returns True.

    Defensive: tolerates ``None`` config, missing keys, ``None`` value, and
    non-Event values (returns False) so a malformed configurable can never
    raise out of the checkpoint helper itself (which would mask the actual
    bug under a CancelledByEventError-shaped error).

    [r4 P2] When ``cancel_event`` is PRESENT but has the wrong type
    (missing callable ``is_set``), log at WARN. Silent fail-False would mask
    a real wiring bug — e.g. PR-5 runner_starter injects a plain dict by
    mistake; the child would never be cancellable and the symptom would only
    surface as waiter timeouts in PR-9 E2E. The WARN surfaces the
    misconfiguration where it happens.
    """
    if not config:
        return False
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not configurable:
        return False
    ce = configurable.get("cancel_event") if isinstance(configurable, dict) else None
    if ce is None:
        return False
    is_set = getattr(ce, "is_set", None)
    if not callable(is_set):
        logger.warning(
            "_should_cancel: configurable.cancel_event present but lacks "
            "callable is_set() — got %s (cancel will not propagate)",
            type(ce).__name__,
        )
        return False
    try:
        return bool(is_set())
    except Exception:
        logger.warning(
            "_should_cancel: cancel_event.is_set() raised — treating as "
            "not-cancelled",
            exc_info=True,
        )
        return False


def _extract_shell_images(result_str: str) -> tuple[str, list[dict]]:
    """Extract base64 image data URLs from shell output.

    Uses str.find() prefix detection + character-set boundary scan.
    Does NOT use regex (base64 payloads can be megabytes).
    """
    if "data:image/" not in result_str:
        return result_str, []

    _B64_CHARS = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )

    image_blocks: list[dict] = []
    cleaned_parts: list[str] = []
    pos = 0

    while pos < len(result_str) and len(image_blocks) < _MAX_FILE_VIEW_IMAGES:
        start = result_str.find("data:image/", pos)
        if start == -1:
            cleaned_parts.append(result_str[pos:])
            break

        cleaned_parts.append(result_str[pos:start])

        b64_marker = result_str.find(";base64,", start, start + 50)
        if b64_marker == -1:
            cleaned_parts.append(result_str[start:start + 20])
            pos = start + 20
            continue

        mime_type = result_str[start + 5:b64_marker]

        # Reject non-raster MIME types (e.g. SVG) — aligned with registry exclusion
        _ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}
        if mime_type not in _ALLOWED_IMAGE_MIMES:
            cleaned_parts.append(result_str[start:b64_marker + 8])
            pos = b64_marker + 8
            continue

        data_start = b64_marker + 8

        data_end = data_start
        while data_end < len(result_str) and result_str[data_end] in _B64_CHARS:
            data_end += 1

        b64_data = result_str[data_start:data_end]

        # Empty/too-short payload — not a valid image
        if len(b64_data) < 16:
            cleaned_parts.append(result_str[start:data_end])
            pos = data_end
            continue

        from app.infrastructure.external.llm.message_sanitizer import _MAX_IMAGE_B64_CHARS
        if len(b64_data) > _MAX_IMAGE_B64_CHARS:
            cleaned_parts.append("[image too large, skipped]")
            pos = data_end
            continue

        try:
            _b64.b64decode(b64_data, validate=True)
        except Exception:
            cleaned_parts.append(result_str[start:data_end])
            pos = data_end
            continue

        image_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64_data}",
                "detail": "auto",
            },
        })
        cleaned_parts.append("[image extracted]")
        pos = data_end

    if pos < len(result_str):
        cleaned_parts.append(result_str[pos:])

    return "".join(cleaned_parts), image_blocks


# ============================================================
# R2 CS2 PR-B Commit 1 — tool_node helpers (Layer 1 / 2 / 3)
# ============================================================
#
# These helpers back the tool_node dispatch path. The real services are wired
# from ``configurable`` (e.g. ``summary_llm`` for SmartApprove); tool-permission
# dispatch now flows through the PermissionEngine (``_pe_dispatch``) for eligible
# sources, with the legacy risk gate as fail-open fallback. Timeout constants are
# overridden by injected ``ToolRuntimeConfig`` values.

_SMART_APPROVE_TIMEOUT_SECONDS = 15
_MAX_WRAPPER_OUTPUT_BYTES = 1 << 20  # 1 MiB


class _SessionContext(NamedTuple):
    """Lightweight per-invocation context passed into Layer 1/2/3 helpers.

    Constructed by the tool_node dispatcher from ``configurable``. Kept
    narrow on purpose: only fields actually needed by policy stages live
    here, so the helpers stay easy to test in isolation.
    """

    session_id: str
    user_id: str


def _is_shell_category(tool_source: ToolSource) -> bool:
    """Stage S applies only to native shell tools.

    N1 may extend this to include ``native skill shell`` once skill tools
    that shell out are covered. For Commit 1 only native shell qualifies.
    """
    return tool_source.source == "native" and tool_source.category == "shell"


GuideInjector = Callable[[str], str | None]


# [C2b §4.3] Stateless, re-entrant child-scope gate shared by the tool_node
# guard. ChildScopeGate.check_in_scope is a pure function (no writer/queue/SSM
# writes), so a single module-level instance is safe.
_child_scope_gate = ChildScopeGate()


def _build_tool_call_spec_from_tc(
    tc: dict,
    configurable: dict,
    tool_source: ToolSource,
    assessment: "Any | None" = None,
    source_metadata: "Any | None" = None,
) -> "Any":
    """Build a ``ToolCallSpec`` from a tool_call dict + configurable slots.

    Extracted as a module-level helper so ``_pe_dispatch`` (inside
    ``build_react_graph``) and unit tests can call it without constructing
    a full graph.

    ``assessment`` is an optional ``RiskAssessment`` — pass the result from
    ``_risk_assessor.assess()`` when available (native tools with risk
    metadata), or ``None`` for low-risk / skill / unknown tools.

    ``source_metadata`` (PE-1 §3.2) is an optional ``SourceMetadata`` (e.g.,
    ``SkillCallMetadata``) populated by the caller for non-native sources.
    SkillSource requires this; NativeSource ignores it.
    """
    from app.domain.services.permission.tool_call_spec import ToolCallSpec

    args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
    user_id = configurable.get("user_id") or ""
    session_id = configurable.get("session_id") or ""

    primary_arg: str | None = None
    dir_arg: str | None = None
    arg_digest: str | None = None
    if assessment is not None:
        primary_arg = assessment.primary_arg or None
        dir_arg = assessment.dir_arg
        arg_digest = assessment.arg_digest or None

    return ToolCallSpec(
        tool_name=tc["name"],
        tool_args=args,
        tool_source=tool_source.source,
        user_id=user_id,
        session_id=session_id,
        primary_arg=primary_arg,
        dir_arg=dir_arg,
        arg_digest=arg_digest,
        risk_assessment=assessment,
        tool_call_id=tc.get("id", ""),
        source_metadata=source_metadata,
    )


async def _enforce_child_scope_or_raise(
    state: ReactGraphState, configurable: dict
) -> None:
    """[C2b §4.3] Child-scope enforcement at tool_node entry.

    When ``configurable`` carries a ``child_permission_context`` (only
    coordinator children — root sessions never inject it), validate EVERY
    not-yet-completed tool_call in the active batch against ``ChildScopeGate``
    BEFORE any tool executes (batch-atomic admission). Raises
    ``ChildScopeViolation`` on the first out-of-scope call. No-op for roots.

    Pure read: a single up-front SSM ``get_mode_with_revision`` (no TOCTOU
    re-read) + the pure ``check_in_scope`` — zero writes (INV-1).
    """
    cpc = configurable.get("child_permission_context")
    if cpc is None:
        return
    from app.domain.services.permission.context import EvaluationContext

    ssm = configurable.get("session_state_machine")
    _session_id = configurable.get("session_id") or ""  # == cpc.child_session_id
    mode, current_rev = await ssm.get_mode_with_revision(_session_id)
    ctx = EvaluationContext(
        session_mode=mode,
        session_mode_revision=current_rev,
        child_permission_context=cpc,
        request_id=configurable.get("request_id", "") or "",
    )
    # active AIMessage.tool_calls — backward scan (mirror _pe_dispatch:1135-1138);
    # NOT messages[-1] (on replay/partial completion the last message may not be
    # the active AIMessage).
    tool_calls: list[dict] = []
    for _msg in reversed(state["messages"]):
        if isinstance(_msg, AIMessage) and _msg.tool_calls:
            tool_calls = _msg.tool_calls
            break
    already_done = set(state.get("completed_tool_call_prefix", []) or [])
    for tc in tool_calls:
        if tc["id"] in already_done:
            continue
        # resolve_tool_source THROWS ToolSourceUnknownError (does not return a
        # sentinel) — mirror the existing catch (react_graph.py:1291) so an unknown
        # child tool reaches the gate as OUT_OF_TOOL_ALLOWLIST instead of a
        # pre-gate crash.
        try:
            tool_source = resolve_tool_source(tc["name"])
        except ToolSourceUnknownError:
            tool_source = ToolSource(
                source="native", category="unknown", canonical_name=tc["name"],
            )
        call_spec = _build_tool_call_spec_from_tc(tc, configurable, tool_source)
        decision = await _child_scope_gate.check_in_scope(call_spec, ctx, cpc)
        if decision != ScopeDecision.IN_SCOPE:
            raise ChildScopeViolation(
                decision,
                tool_name=tc["name"],
                target_path=extract_target_path(call_spec),
            )


def _session_ctx_from(config: RunnableConfig | None) -> _SessionContext:
    """Build a ``_SessionContext`` from the LangGraph ``RunnableConfig``.

    Reads ``session_id`` / ``user_id`` from ``config['configurable']`` — the
    same slots the existing react_graph code uses at
    ``react_graph.py:451-452`` and ``agent_service.py``. Defaults to empty
    strings when absent, matching pre-R2 behavior.
    """
    configurable = (config or {}).get("configurable", {}) if config else {}
    return _SessionContext(
        session_id=configurable.get("session_id") or "",
        user_id=configurable.get("user_id") or "",
    )


def _interrupt_helper_early_return(
    state: ReactGraphState,
) -> Command[Literal["tool_node"]] | None:
    """Defensive pre-check for ``interrupt_helper``.

    Returns a ``Command(goto="tool_node", update={})`` if the node was
    routed to without valid ``pending_ask_*`` state (should not happen
    in normal flow), otherwise ``None`` to let the caller proceed to
    the real ``interrupt()`` handshake.

    Extracted as a module-level helper so the defensive branch can be
    unit tested without driving the full graph — ``interrupt_helper``
    itself is a closure inside ``build_react_graph`` and can only be
    reached via a compiled graph.
    """
    pending_id = state.get("pending_ask_tool_call_id")
    pending_artifact_dict = state.get("pending_ask_artifact")
    if pending_id is None or pending_artifact_dict is None:
        return Command(goto="tool_node", update={})
    return None


def _rehydrate_call_spec(state: ReactGraphState, tool_call_id: str) -> "Any":
    """Reconstruct a ToolCallSpec from pending_ask_* state fields.

    Used by the PE path of interrupt_helper to pass a ToolCallSpec to
    pe.commit_resume. Reads tool_name / tool_source from pending_ask_artifact
    and tool_args from pending_ask_tool_args. user_id / session_id are NOT
    available from state alone (they live in configurable); callers must
    pass them via a thin wrapper that has access to configurable.
    """
    from app.domain.services.permission.tool_call_spec import ToolCallSpec

    artifact_dict = state.get("pending_ask_artifact") or {}
    tool_args = state.get("pending_ask_tool_args") or {}
    tool_source_dict = artifact_dict.get("tool_source") or {}

    tool_name = artifact_dict.get("tool_name") or ""
    tool_source_str = (
        tool_source_dict.get("source") if isinstance(tool_source_dict, dict) else "native"
    ) or "native"

    return ToolCallSpec(
        tool_name=tool_name,
        tool_args=dict(tool_args),
        tool_source=tool_source_str,
        user_id="",  # filled in by interrupt_helper from configurable
        session_id="",  # filled in by interrupt_helper from configurable
        tool_call_id=tool_call_id,
    )


def _build_resume_error_command(
    state: ReactGraphState,
    exc: Exception,
) -> Command:  # type: ignore[type-arg]
    """Build a Command that surfaces a PolicyConflict/WriterIntegrityError as a ToolMessage.

    Returns Command(goto="tool_node") with a deny-flavoured ToolMessage so
    the agent loop can proceed without silently stalling.

    P2#6: The pending_id is added to completed_tool_call_prefix so that the
    next tool_node replay skips re-evaluating this tool_call (which would
    re-enter PE.evaluate and potentially re-emit another ToolConfirmationEvent
    or trigger a duplicate confirmation request).
    """
    pending_id = state.get("pending_ask_tool_call_id") or "unknown"
    artifact_dict = state.get("pending_ask_artifact") or {}
    tool_name = artifact_dict.get("tool_name") or "unknown_tool"

    existing_prefix = list(state.get("completed_tool_call_prefix", []) or [])

    error_msg = ToolMessage(
        content=f"[POLICY_CONFLICT] {exc}",
        tool_call_id=pending_id,
        name=tool_name,
        status="error",
    )
    return Command(
        goto="tool_node",
        update={
            "messages": [error_msg],
            "completed_tool_call_prefix": existing_prefix + [pending_id],
            "pending_ask_outcome": None,
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None,
            "pending_ask_tool_args": None,
        },
    )


def _merge_update(cmd: Command, extra: dict) -> Command:  # type: ignore[type-arg]
    """Merge ``extra`` dict into the ``update`` field of a ``Command``.

    Used by the pe_resume_outcomes replay path (Task 10.3) to append the
    ``consumed_update`` (clearing the replayed entry from pe_resume_outcomes)
    onto the Command returned by ``_dispatch_outcome`` — without mutating the
    original object.

    Both dicts are shallow-merged; keys in ``extra`` override those in
    ``cmd.update``.
    """
    merged: dict[str, Any] = dict(cmd.update or {})
    merged.update(extra)
    return Command(goto=cmd.goto, update=merged)


async def _translate_outcome(
    outcome: ToolOutcome,
    tool_call: ToolCall,
    tool_source: ToolSource,
    session_ctx: _SessionContext,
    *,
    tool_result_max_chars: int,
    guide_injector: GuideInjector | None,
    enabled_outcome_variants: list[str] | None = None,   # ← NEW (Round 2f P1)
) -> tuple[ToolMessage | None, list[HumanMessage], list[Any]]:
    """Layer 3: convert ``ToolOutcome`` → ``ToolMessage`` + deferred ``HumanMessage`` list + events.

    Returns ``(tool_msg, deferred_human_msgs, events)``:
    - ``tool_msg`` is ``None`` **iff** the outcome is ``Asked`` (interrupt
      path — the ``tool_node`` dispatcher should ``goto interrupt_helper``
      and emit no ToolMessage yet).
    - ``deferred_human_msgs`` is non-empty **only** on ``Passthrough`` —
      multimodal content blocks must be re-emitted as a separate
      ``HumanMessage`` because ``ToolMessage.artifact`` is a graph-side
      side-channel that the LLM cannot read. The shape strictly matches
      the existing ``react_graph.py:559-571`` deferred HumanMessage
      pattern (I-4.4c) so downstream consumers
      (``test_file_view_integration.py``, ``message_utils``,
      ``main_graph._compact_messages``) keep working.
    - ``events`` is the domain event list to emit via the bridge.

    **NOT done in Layer 3**: ``[TOOL_FAILED] / [TOOL_DENIED]`` error-prefix
    injection. That is the LLM adapter's job (Task 33/34, Commit 2b) and
    lives in ``ActusChatModel._messages_for_api()`` /
    ``ActusResponsesModel._messages_for_api()``. Layer 3 only sets
    ``ToolMessage.status`` correctly and records the typed artifact.
    """
    del session_ctx  # accepted in signature for consistency; not used yet
    # Runtime enforcement guard (Round 2f P1): refuse unknown/disabled variants.
    # enabled_outcome_variants=None → skip (backward-compat / dev envs).
    if enabled_outcome_variants is not None:
        variant_name = getattr(outcome, "variant", None)
        if variant_name is not None and variant_name not in enabled_outcome_variants:
            logger.error(
                "CS3 executable guard: outcome variant %r not in enabled_outcome_variants=%r. "
                "Refusing to emit — wrapper must respect runtime flag for coordinated rollout.",
                variant_name, enabled_outcome_variants,
            )
            raise ValueError(
                f"variant {variant_name!r} not enabled in runtime config "
                f"(enabled={enabled_outcome_variants})"
            )
    artifact = ToolArtifact(
        tool_call_id=tool_call["id"],
        tool_name=tool_call["name"],
        tool_source=tool_source,
        outcome=outcome,
    )
    artifact_json = artifact.model_dump(mode="json", by_alias=True)
    events: list[Any] = []
    deferred: list[HumanMessage] = []

    # Asked: interrupt path — no ToolMessage, caller routes to interrupt_helper.
    #
    # **_translate_outcome does NOT emit ToolConfirmationEvent** — that is
    # the dispatcher's job. Rationale:
    # - ``Asked.reason.type`` is a SOURCE taxonomy
    #   (approval_policy / smart_approve / risk_enforce / ast_validator),
    #   not a severity scale.
    # - ``ToolConfirmationEvent.risk_level`` is a SEVERITY scale
    #   (``high`` / ``medium`` / ``low``), and the existing legacy risk
    #   gate at ``tool_node`` line ~933 populates it from
    #   ``assessment.final_level.name.lower()`` alongside
    #   ``matched_patterns`` and ``suggested_alternative``.
    # - Frontend confirmation card styling and audit persistence depend on
    #   the severity axis + patterns + alternative, none of which live on
    #   ``Asked`` / ``ToolArtifact``.
    #
    # If ``_translate_outcome`` constructed the event itself, every future
    # caller would have to hand-patch ``risk_level`` / ``matched_patterns``
    # / ``suggested_alternative`` back in. Instead, return an empty event
    # list on Asked and let the dispatcher emit the confirmation event
    # with its own ``RiskAssessment`` (legacy gate) or its own
    # Layer-1-specific context (Task 21 / Commit 2a).
    if isinstance(outcome, Asked):
        return None, deferred, events

    # All other variants construct a ToolMessage. Step 1: content + guide.
    final_content = outcome.content
    success_variant = isinstance(outcome, (AllowSuccess, Passthrough))
    if success_variant and guide_injector is not None:
        guide = guide_injector(tool_call["name"])
        if guide:
            final_content = f"{final_content}\n\n---\n[Skill Guide]\n{guide}"

    # Step 2: truncate (guide injected BEFORE truncation matches existing
    # react_graph.py:794-799 order — I-4.4a / I-4.4b ordering invariant).
    final_content = truncate_tool_content(final_content, tool_result_max_chars)

    # Step 3: Variant → lc_status (side-channel to LLM adapter prefix logic)
    lc_status: Literal["success", "error"]
    if isinstance(outcome, (AllowSuccess, Passthrough)):
        lc_status = "success"
    elif isinstance(outcome, (AllowError, Denied)):
        lc_status = "error"
    else:
        raise AssertionError(f"Unreachable ToolOutcome variant: {type(outcome).__name__}")

    msg = ToolMessage(
        content=final_content,
        artifact=artifact_json,
        status=lc_status,
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )
    # R1/R2 convention: ToolEvent.tool_name stores the canonical CATEGORY
    # (browser / search / shell / file / ...), NOT the literal tool name.
    # AgentTaskRunner._handle_tool_event (agent_task_runner.py:2078) branches
    # on event.tool_name to enrich browser screenshots / search results /
    # etc., so emitting the actual tool_call name here would silently break
    # the enrichment path. The literal tool name lives in function_name.
    #
    # ``function_result`` MUST be populated even on the Denied / AllowError
    # paths — ``AgentTaskRunner._handle_tool_event`` reads
    # ``event.function_result.message`` / ``.success`` / ``.data`` to enrich
    # search / mcp / a2a / skill / file tool content. Without it, denied or
    # timed-out tools surface on the frontend as "(MCP工具无可用结果)" /
    # "(Skill工具无可用结果)" placeholders instead of the real rejection
    # reason. Message uses ``final_content`` (already truncated + guide
    # injected for success paths, already the Denied/AllowError text for
    # failure paths).
    _fn_result_success = isinstance(outcome, (AllowSuccess, Passthrough))
    events.append(
        ToolEvent(
            tool_call_id=tool_call["id"],
            tool_name=tool_source.category,
            function_name=tool_call["name"],
            function_args=tool_call["args"],
            function_result=ToolResult(
                success=_fn_result_success,
                message=final_content,
            ),
            status=ToolEventStatus.CALLED,
            # R4: artifact as dict (F2 fix). Projector 消费时用 TOOL_ARTIFACT_ADAPTER 懒校验.
            artifact=artifact_json,
            tool_source=tool_source,
        )
    )

    # Step 4: Passthrough → emit deferred HumanMessage so LLM "sees"
    # multimodal content. Strict 1:1 reuse of react_graph.py:810-823
    # existing structure and text so downstream test_file_view_integration.py
    # exact-match assertions still pass.
    if isinstance(outcome, Passthrough):
        blocks_capped = outcome.data.blocks[:_MAX_FILE_VIEW_IMAGES]
        omitted = len(outcome.data.blocks) - len(blocks_capped)
        human_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"[file_view: {tool_call['name']} — "
                    f"{len(blocks_capped)} image(s) loaded]"
                ),
            }
        ]
        for block in blocks_capped:
            human_content.append(block.model_dump(by_alias=True))
        if omitted > 0:
            human_content.append(
                {
                    "type": "text",
                    "text": f"[... {omitted} more images omitted]",
                }
            )
        deferred.append(HumanMessage(content=human_content))

    return msg, deferred, events


def _maybe_convert_shell_outcome_with_images(
    outcome: ToolOutcome,
    tool_name: str,
) -> ToolOutcome:
    """Legacy compat: shell tools sometimes embed base64 images in stdout.

    The current ``shell_execute`` / ``shell_read_output`` wrappers emit
    ``AllowSuccess(content=<raw stdout>)`` and leave image extraction to
    the dispatcher (historically inside the now-retired ``_run_tool``
    helper). This post-processor inspects ``AllowSuccess`` outcomes for
    shell tools, extracts ``data:image/...;base64,...`` payloads via
    ``_extract_shell_images``, and if any are found converts the
    outcome to a ``Passthrough`` carrying a typed
    ``MultimodalPayload``. Everything else passes through unchanged.

    This lets ``_translate_outcome``'s ``Passthrough`` branch attach
    the images as a deferred ``HumanMessage`` so the LLM actually
    sees them — which is what the legacy ``multimodal_blocks``
    pipeline used to do before R2.

    N.B. a cleaner fix would move this into the ``shell_execute``
    wrapper so the dispatcher doesn't know about tool-specific
    post-processing. Keeping it here for now minimizes Chunk 3
    blast radius; a follow-up can promote it into the wrapper.
    """
    if tool_name not in ("shell_execute", "shell_read_output"):
        return outcome
    if not isinstance(outcome, AllowSuccess):
        return outcome
    if not isinstance(outcome.content, str):
        return outcome
    if "data:image/" not in outcome.content:
        return outcome

    cleaned, image_dicts = _extract_shell_images(outcome.content)
    if not image_dicts:
        return outcome

    typed_blocks: list[ImageUrlBlock] = []
    for block_dict in image_dicts:
        image_url_dict = block_dict.get("image_url") or {}
        typed_blocks.append(
            ImageUrlBlock(
                image_url=ImageUrlPayload(
                    url=image_url_dict.get("url", ""),
                    detail=image_url_dict.get("detail", "auto"),
                )
            )
        )

    return Passthrough(
        content=cleaned,
        data=MultimodalPayload(blocks=typed_blocks),
    )


def _too_large_outcome(tool_name: str, byte_len: int, max_bytes: int) -> AllowError:
    return AllowError(
        content=(
            f"工具 '{tool_name}' 输出超长 "
            f"({byte_len} bytes > {max_bytes})"
        ),
        reason=DecisionReason(
            type="exception",
            code="wrapper_output_too_large",
            message=f"{byte_len} bytes",
        ),
        retryable=False,
    )


def _legacy_raw_to_outcome(raw: Any) -> ToolOutcome:
    """Coerce a pre-R2 wrapper return value (``response_format='content'``)
    into a typed ``ToolOutcome``.

    Branches:
    - ``FileProcessResult`` → ``Passthrough`` carrying a ``MultimodalPayload``
      with the tool's image / document blocks (restores pre-R2 file_view
      behavior that predates the ``content_and_artifact`` wrapper migration).
    - ``str`` → ``AllowSuccess(content=<str>)``.
    - anything else → ``AllowSuccess(content=str(raw))`` fallback.
    """
    if isinstance(raw, FileProcessResult):
        typed_blocks: list[Any] = []
        for block in list(raw.image_blocks):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "image_url":
                img = block.get("image_url") or {}
                typed_blocks.append(
                    ImageUrlBlock(
                        image_url=ImageUrlPayload(
                            url=img.get("url", ""),
                            detail=img.get("detail", "auto"),
                        )
                    )
                )
        for block in list(getattr(raw, "document_blocks", None) or []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "file":
                file = block.get("file") or {}
                typed_blocks.append(
                    FileBlock(
                        file=FilePayload(
                            filename=file.get("filename", "document"),
                            file_data=file.get("file_data", ""),
                        )
                    )
                )
        if typed_blocks:
            return Passthrough(
                content=raw.text,
                data=MultimodalPayload(blocks=typed_blocks),
            )
        return AllowSuccess(content=raw.text)
    if isinstance(raw, str):
        return AllowSuccess(content=raw)
    return AllowSuccess(content=str(raw))


async def _invoke_wrapper(
    tool: BaseTool,
    tool_call: ToolCall,
    tool_source: ToolSource,
    *,
    session_id: str = "",
    max_wrapper_output_bytes: int | None = None,
) -> ToolOutcome:
    """Layer 2: invoke wrapper via ``content_and_artifact`` and return typed outcome.

    NOTE: langchain-core 1.2.17 only returns ``ToolMessage`` when ``ainvoke()``
    receives a full ToolCall dict (``{"args", "id", "name", "type"}``). Passing
    only the plain args dict returns raw content and loses the artifact.

    ``max_wrapper_output_bytes`` comes from
    ``AppConfig.tool_runtime.max_wrapper_output_bytes`` via
    ``build_react_graph(tool_runtime_config=...)``. ``None`` falls back
    to the module constant ``_MAX_WRAPPER_OUTPUT_BYTES`` so legacy
    callers work unchanged.
    """
    max_bytes = (
        max_wrapper_output_bytes
        if max_wrapper_output_bytes is not None
        else _MAX_WRAPPER_OUTPUT_BYTES
    )
    del tool_source
    config = {"configurable": {"session_id": session_id}} if session_id else None

    legacy_response_format = (
        getattr(tool, "response_format", "content") != "content_and_artifact"
    )

    # Legacy branch: tool predates the CS2 wrapper migration. Call
    # ``ainvoke(args)`` to receive the raw Python object so we can
    # special-case ``FileProcessResult``, preserving the pre-R2 file_view
    # multimodal path under the new dispatcher.
    if legacy_response_format:
        try:
            raw = await tool.ainvoke(tool_call["args"], config=config)
        except asyncio.TimeoutError as exc:
            return AllowError(
                content=f"工具 '{tool.name}' 执行超时: {exc}",
                reason=DecisionReason(
                    type="timeout",
                    code="wrapper_ainvoke_timeout",
                    message=str(exc),
                ),
                retryable=True,
            )
        except Exception as exc:
            logger.exception("Unexpected wrapper exception for %s", tool.name)
            return AllowError(
                content=f"工具 '{tool.name}' 内部异常: {exc}",
                reason=DecisionReason(
                    type="exception",
                    code=type(exc).__name__,
                    message=str(exc),
                ),
                retryable=False,
            )

        outcome = _legacy_raw_to_outcome(raw)
        if isinstance(outcome.content, str) and len(outcome.content) > max_bytes:
            return _too_large_outcome(tool.name, len(outcome.content), max_bytes)
        return outcome

    # R2 CS2 typed path: tool opts into ``content_and_artifact``, so
    # ``ainvoke(ToolCall dict)`` returns a ``ToolMessage`` whose
    # ``artifact`` is the pre-built ``ToolOutcome``.
    try:
        tool_msg = await tool.ainvoke(
            {
                "args": tool_call["args"],
                "id": tool_call["id"],
                "name": tool_call["name"],
                "type": "tool_call",
            },
            config=config,
        )
    except asyncio.TimeoutError as exc:
        return AllowError(
            content=f"工具 '{tool.name}' 执行超时: {exc}",
            reason=DecisionReason(
                type="timeout",
                code="wrapper_ainvoke_timeout",
                message=str(exc),
            ),
            retryable=True,
        )
    except Exception as exc:
        logger.exception("Unexpected wrapper exception for %s", tool.name)
        return AllowError(
            content=f"工具 '{tool.name}' 内部异常: {exc}",
            reason=DecisionReason(
                type="exception",
                code=type(exc).__name__,
                message=str(exc),
            ),
            retryable=False,
        )

    if not isinstance(tool_msg, ToolMessage):
        return AllowError(
            content=f"工具 '{tool.name}' 返回非 ToolMessage 类型: {type(tool_msg).__name__}",
            reason=DecisionReason(
                type="exception",
                code="wrong_ainvoke_shape",
                message=(
                    "Expected ToolMessage from tool.ainvoke(ToolCall dict); "
                    f"got {type(tool_msg).__name__}"
                ),
            ),
            retryable=False,
        )

    if isinstance(tool_msg.content, str) and len(tool_msg.content) > max_bytes:
        return _too_large_outcome(tool.name, len(tool_msg.content), max_bytes)

    # Safety net: wrapper declared ``content_and_artifact`` but didn't
    # populate the artifact (misconfigured tool). Fall back to
    # ``AllowSuccess`` from the text content so the dispatcher doesn't
    # crash the whole step.
    if tool_msg.artifact is None:
        raw_content = tool_msg.content if isinstance(tool_msg.content, str) else str(
            tool_msg.content
        )
        return AllowSuccess(content=raw_content)

    try:
        return TOOL_OUTCOME_ADAPTER.validate_python(tool_msg.artifact)
    except Exception as exc:
        return AllowError(
            content=(
                f"工具 '{tool.name}' artifact 非 ToolOutcome variant: "
                f"{type(tool_msg.artifact).__name__}"
            ),
            reason=DecisionReason(
                type="exception",
                code="invalid_tool_outcome_artifact",
                message=str(exc),
            ),
            retryable=False,
        )


def build_react_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    agent_config: AgentConfig | None = None,
    tool_result_max_chars: int = 8000,
    assembler: ContextAssembler | None = None,
    checkpointer: Any = None,
    tool_runtime_config: "ToolRuntimeConfig | None" = None,
) -> CompiledStateGraph:
    """Build and compile the inner ReAct loop graph.

    Parameters
    ----------
    llm : LangChain BaseChatModel — must support bind_tools.
    tools : List of LangChain tools.
    agent_config : Optional AgentConfig for iteration limits etc.
    checkpointer : Optional LangGraph checkpointer (e.g. ``InMemorySaver``
        for tests, ``AsyncPostgresSaver`` for production). When provided,
        the graph supports ``interrupt()`` resume via
        ``Command(resume=...)``. R2 Day-4 hard gate tests rely on
        ``InMemorySaver`` to drive the interrupt_helper handshake.
    tool_runtime_config : R2 CS2 — ``AppConfig.tool_runtime`` plumbed in.
        Sets the SmartApprove timeout + wrapper output byte cap used by
        Layer 2 ``_invoke_wrapper``.
        Defaults to ``ToolRuntimeConfig()`` (1 MiB wrapper cap, 15s
        SmartApprove timeout). The values are captured in the closure
        so every future call site that wires
        ``_invoke_wrapper`` into the dispatcher can read them via
        ``_tool_runtime_cfg`` without re-plumbing ``build_react_graph``.
    """
    # R2 CS2: captured in closure so the dispatcher and helpers read
    # the same config instance regardless of call path. Deferred import
    # keeps ``react_graph.py`` from pulling the full app_config module
    # chain at import time.
    if tool_runtime_config is None:
        from app.domain.models.app_config import ToolRuntimeConfig as _TRC

        _tool_runtime_cfg = _TRC()
    else:
        _tool_runtime_cfg = tool_runtime_config

    # Build tool lookup
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    # ---- Nodes --------------------------------------------------------- #

    async def pre_llm_node(state: ReactGraphState, config: RunnableConfig) -> dict:
        """Trim messages for LLM input. state['messages'] is unchanged.

        [C2 PR-4 §8.4 #2 react_loop_entry] First cancel checkpoint of the
        ReAct loop iteration. Raising here aborts the iteration BEFORE
        ``context_assembler.assemble`` runs (which can do expensive
        trimming + LLM-summary work for long histories)."""
        if _should_cancel(config):
            raise CancelledByEventError("react_loop_entry")
        if assembler is None:
            return {"llm_input_messages": list(state["messages"])}
        result = assembler.assemble(list(state["messages"]))
        if result.actions:
            logger.info("context_assembler(in-step): %s", result.actions)
        return {"llm_input_messages": result.messages}

    async def llm_node(state: ReactGraphState, config: RunnableConfig) -> dict:
        """Call the LLM with current messages.

        [C2 PR-4 §8.4 #3 llm_node_entry] Cancel checkpoint at LLM call
        boundary. Raising here aborts BEFORE ``llm_with_tools.ainvoke``
        spends a token. The atomic ``ainvoke`` (NOT a streaming
        ``astream``) is why #4 llm_chunk_boundary is deferred until a
        future streaming refactor (see module top docstring)."""
        if _should_cancel(config):
            raise CancelledByEventError("llm_node_entry")
        import time as _time
        messages = list(state.get("llm_input_messages") or state["messages"])

        # D5: Inject recovery hint and blocked summary into messages (not state)
        _configurable = config.get("configurable", {})
        _control = _configurable.get("execution_control")
        _tracker = _configurable.get("tool_failure_tracker")
        _metrics = _configurable.get("execution_metrics")

        if _control and _control.idle_recovery_hint:
            messages = messages + [SystemMessage(content=_control.idle_recovery_hint)]
            _control.idle_recovery_hint = None  # consume once
        if _tracker:
            blocked = _tracker.get_blocked_summary()
            if blocked:
                messages = messages + [SystemMessage(content=blocked)]

        # 诊断日志：检查多模态内容是否到达 react_graph
        multimodal_msgs = [
            (i, [b.get("type") for b in m.content if isinstance(b, dict)])
            for i, m in enumerate(messages)
            if hasattr(m, "content") and isinstance(m.content, list)
        ]
        if multimodal_msgs:
            logger.info(
                "[MULTIMODAL] react llm_node: %d multimodal messages found: %s",
                len(multimodal_msgs), multimodal_msgs,
            )

        _llm_start = _time.monotonic()
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        # D5: Record LLM latency
        if _metrics:
            _metrics.record_llm_call((_time.monotonic() - _llm_start) * 1000)

        new_events = []

        # Emit ToolEvent(CALLING) for each tool call.
        #
        # ``resolve_tool_source`` raises on LLM-hallucinated names and on
        # dynamically-registered MCP/Skill tools that haven't flushed
        # their canonical identity yet. Falling through to the
        # ``unknown`` sentinel keeps the CALLING event path safe — and
        # more importantly prevents the fallback from using ``shell``
        # (which would make ``AgentTaskRunner._handle_tool_event`` read
        # the default shell session's console as the enrichment payload;
        # see the companion fix in ``tool_node``).
        if response.tool_calls:
            for tc in response.tool_calls:
                func_name = tc["name"]
                try:
                    _calling_category = resolve_tool_source(func_name).category
                except ToolSourceUnknownError:
                    _calling_category = "unknown"
                new_events.append(
                    ToolEvent(
                        tool_call_id=tc["id"],
                        tool_name=_calling_category,
                        function_name=func_name,
                        function_args=tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"]),
                        status=ToolEventStatus.CALLING,
                    )
                )

        # 最终回答（无 tool_calls 且有内容）发射 MessageEvent，使前端实时收到。
        # LLM 按 system prompt 要求返回 JSON 格式 {"success","result","attachments"}
        # （见 prompts/sections/output_format.py），需要 unwrap 成用户可读文本。
        # 使用 unwrap_message_envelope 做四级兜底解析，能容忍 LLM 在字符串值里
        # 塞真换行的常见错误；同时兼容 {"message","attachments"} 形状，与
        # SummarizerOutput 的 key 容忍度对齐。
        if not response.tool_calls and response.content:
            display_message = response.content
            if isinstance(display_message, str):
                display_message, _envelope_attachments = unwrap_message_envelope(
                    display_message
                )
            new_events.append(
                MessageEvent(role="assistant", message=display_message)
            )

        # [C2 PR-4 §8.4 #5 llm_return] Cancel checkpoint at llm_node exit.
        # If the parent cancelled DURING the ``ainvoke`` (possible because
        # ``ainvoke`` released the event loop), abort before the response
        # propagates to tool_node where it would consume more compute.
        if _should_cancel(config):
            raise CancelledByEventError("llm_return")

        return {
            "messages": [response],
            "events": new_events,
        }

    # Shared risk assessor instance (stateless, safe to reuse)
    _risk_assessor = RiskAssessor()

    # ======================================================================
    # PE-0 Phase 9: _pe_dispatch — full PermissionEngine evaluation path
    # ======================================================================
    # Called from tool_node when pe + ssm + flag_native are all set.
    # Handles the full per-tool-call PE loop:
    #   1. Resolve tool_source + run RiskAssessor (same as legacy gate)
    #   2. Build ToolCallSpec + EvaluationContext
    #   3. pe.evaluate → AllowSuccess / Denied / Asked / AllowError / Passthrough
    #   4. Translate outcome to ToolMessage Command or interrupt_helper Command
    #
    # Hard invariants enforced here:
    #   - NO writer.write / write_audit_only / delete_grant calls — PE owns those.
    #   - message_ask_user pseudo-tool still routes through legacy path (no ToolSource).
    #   - D5 cooperative termination check runs BEFORE this function is called
    #     (it lives in tool_node above the branch, so we skip it here).

    async def _pe_dispatch(
        state: ReactGraphState, config: RunnableConfig
    ) -> Command[Literal["pre_llm_node", "interrupt_helper", "__end__"]]:
        """PE-0 Phase 9: dispatch tool calls through PermissionEngine.evaluate.

        Called only when pe + ssm + the master ``enabled`` switch are all
        truthy (PE-4c: per-source flags retired). Falls back to an AllowError
        command on unexpected exceptions to prevent silent graph stalls.
        """
        import time as _time

        configurable = (config or {}).get("configurable", {}) if config else {}
        _pe = configurable.get("permission_engine")
        _ssm = configurable.get("session_state_machine")
        guide_injector = configurable.get("skill_guide_injector")
        event_queue = configurable.get("event_queue")
        confirmation_manager = configurable.get("confirmation_manager")
        _tracker = configurable.get("tool_failure_tracker")
        _metrics = configurable.get("execution_metrics")
        _session_id = configurable.get("session_id") or ""
        _user_id = configurable.get("user_id") or ""
        _runtime_max_bytes = _tool_runtime_cfg.max_wrapper_output_bytes

        # D5: re-check after we entered this path (same guard as legacy tool_node)
        _control = configurable.get("execution_control")
        if _control and _control.should_terminate:
            return Command(
                goto=END,
                update={"should_interrupt": True, "messages": [], "events": []},
            )

        # N1: AST validator settings (same as legacy path)
        _settings = get_settings()

        messages = state["messages"]

        # Find the AIMessage with the active tool_calls batch (same logic as legacy)
        tool_calls: list[dict] = []
        for _msg in reversed(messages):
            if isinstance(_msg, AIMessage) and _msg.tool_calls:
                tool_calls = _msg.tool_calls
                break

        has_prior_soft_hint = state.get("soft_hint_sent", False)
        already_done: set[str] = set(state.get("completed_tool_call_prefix", []) or [])
        pre_approved: set[str] = set(state.get("approved_tool_call_ids", []) or [])

        # PE-1 §2.5 (T15 P1#2 fix) + Round 2 P1#2: per-call gate. If ANY
        # pending tool_call in the batch is non-PE-eligible (e.g. skill
        # creator/guide, mcp discovery, an unsupported source, or the master
        # switch off), delegate the WHOLE batch to the legacy tool_node path which
        # preserves the fail-closed skill/mcp/a2a guards + skill creator/guide
        # confirmation pipelines. PE only handles batches where every pending call
        # qualifies for PE — otherwise we'd bypass legacy per-source
        # confirmation for mixed cases.
        #
        # ``is_pe_eligible_tool_source`` consumes the full ToolSource
        # (source + category) so that skill creator (``brainstorm_skill``,
        # ``generate_skill``, ``install_skill``) and skill guide
        # (``get_skill_guide``) tools — which share ``source="skill"`` but
        # are NOT in ``SkillTool._tool_bindings`` — also fall back to
        # legacy. ``build_skill_call_metadata`` would otherwise emit
        # ``AllowError(code="skill_metadata_unresolvable")`` for them.
        #
        # message_ask_user is a synthetic pseudo-tool with no ToolSource
        # and is handled inline within the PE loop (SOFT_HINT branch); it
        # does NOT count toward PE eligibility either way.
        _tc_for_gate = configurable.get("tool_confirmation_config")
        for _pre_tc in tool_calls:
            _pre_call_id = _pre_tc["id"]
            if _pre_call_id in already_done:
                continue
            _pre_name = _pre_tc["name"]
            if _pre_name == "message_ask_user":
                continue
            try:
                _pre_src = resolve_tool_source(_pre_name)
            except ToolSourceUnknownError:
                _pre_src = None
            if _tc_for_gate is None or not is_pe_eligible_tool_source(
                _pre_src, _tc_for_gate
            ):
                # Non-PE-eligible call detected (unknown source, unsupported
                # source, master switch off, or skill creator/guide) — fall back
                # to legacy tool_node for the whole batch.
                logger.debug(
                    "_pe_dispatch: non-PE-eligible tool '%s' (source=%s, "
                    "category=%s) in batch → falling back to legacy "
                    "tool_node path for the whole batch.",
                    _pre_name,
                    getattr(_pre_src, "source", None),
                    getattr(_pre_src, "category", None),
                )
                return None  # type: ignore[return-value]

        new_completed_ids: list[str] = []
        new_messages: list = []
        new_events: list = []
        new_deferred_human: list[HumanMessage] = []
        should_interrupt = False
        new_failures = 0

        session_ctx = _session_ctx_from(config)

        async def _finalize_pe_outcome(
            tc: dict,
            tc_args: dict,
            tool_source: ToolSource,
            outcome: ToolOutcome,
            tool_start_ts: float,
        ) -> None:
            """Same as _finalize_outcome but for PE path — no tracker/metrics changes."""
            nonlocal new_failures

            msg, deferred, events = await _translate_outcome(
                outcome,
                tc,
                tool_source,
                session_ctx,
                tool_result_max_chars=tool_result_max_chars,
                guide_injector=guide_injector,
                enabled_outcome_variants=_tool_runtime_cfg.enabled_outcome_variants,
            )
            if msg is not None:
                new_messages.append(msg)
            new_deferred_human.extend(deferred)
            for evt in events:
                new_events.append(evt)

            is_success = isinstance(outcome, (AllowSuccess, Passthrough))
            tc_name = tc["name"]
            if _tracker:
                if is_success:
                    _tracker.record_success(tc_name, tc_args)
                else:
                    _tracker.record_failure(tc_name, tc_args)
            if _metrics:
                _metrics.record_tool_call(
                    success=is_success,
                    latency_ms=(_time.monotonic() - tool_start_ts) * 1000,
                )
            if not is_success:
                new_failures += 1

            new_completed_ids.append(tc["id"])

        # PE-0 Phase 10: accumulate pe_resume_outcomes cleanup entries from
        # the replay path.  Merged into the batch-completion update after
        # the loop to clear consumed entries from state.
        _batch_pe_resume_consumed: dict[str, Any] = {}

        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            call_id = tc["id"]
            _tool_start = _time.monotonic()

            if call_id in already_done:
                continue

            _bypass_risk_gate = call_id in pre_approved

            # message_ask_user: no ToolSource — handle via legacy SOFT_HINT path
            if tool_name == "message_ask_user":
                suggest = str(args.get("suggest_user_takeover", "none")).strip().lower()
                if suggest in {"browser", "shell"}:
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                elif not has_prior_soft_hint:
                    result_str = "SOFT_HINT"
                    logger.info("message_ask_user (PE path): returning SOFT_HINT")
                else:
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True

                new_messages.append(
                    ToolMessage(content=result_str, tool_call_id=call_id, name=tool_name)
                )
                new_events.append(
                    ToolEvent(
                        tool_call_id=call_id,
                        tool_name=resolve_tool_source(tool_name).category,
                        function_name=tool_name,
                        function_args=args,
                        function_result=ToolResult(success=True, message=result_str),
                        status=ToolEventStatus.CALLED,
                    )
                )
                new_completed_ids.append(call_id)
                continue

            # Resolve ToolSource
            try:
                tool_source = resolve_tool_source(tool_name)
            except ToolSourceUnknownError:
                tool_source = ToolSource(
                    source="native",
                    category="unknown",
                    canonical_name=tool_name,
                )

            # PE-1 §2.5 (T15 P1#2 defensive) + Round 2 P1#2: pre-loop should
            # have routed any non-PE-eligible call to legacy. If we reach here
            # with a non-PE-eligible source (or skill creator/guide), it's a
            # caller-side invariant bug — PE-4c fails CLOSED rather than
            # executing unconfirmed.
            _is_pe_eligible_per_call = is_pe_eligible_tool_source(
                tool_source, _tc_for_gate
            )
            if not _is_pe_eligible_per_call:
                logger.error(
                    "_pe_dispatch: per-call non-PE-eligible reached PE loop "
                    "for tool '%s' (source=%s, category=%s) — pre-loop guard "
                    "should have prevented this; failing closed (PE-4c)",
                    tool_name,
                    tool_source.source,
                    tool_source.category,
                )
                # PE-4c: fail-CLOSED. Previously this executed via _invoke_wrapper
                # (fail-open). A non-PE-eligible call reaching the PE loop is a
                # caller-side invariant bug; running it unconfirmed is worse than
                # denying it. The agent re-issues the tool in its own batch, which
                # the pre-loop gate then routes to legacy passthrough correctly.
                _non_pe_denied = Denied(
                    content=(
                        f"工具 '{tool_name}' 因权限引擎资格判定不一致被拒绝执行；"
                        "请在单独的步骤中重新调用该工具。"
                    ),
                    reason=DecisionReason(
                        type="approval_policy",
                        code="pe_eligibility_invariant_violation",
                        message=(
                            "non-PE-eligible call reached the PE loop; "
                            "pre-loop gate should have routed it to legacy"
                        ),
                    ),
                )
                await _finalize_pe_outcome(
                    tc, args, tool_source, _non_pe_denied, _tool_start
                )
                new_completed_ids.append(call_id)
                continue

            # D5 tracker: block repeated failures
            if _tracker and _tracker.is_blocked(tool_name, args):
                blocked_outcome = AllowError(
                    content=(
                        f"[BLOCKED] 此工具调用模式（{tool_name}）因连续失败已被暂停，"
                        "请尝试不同的工具或参数"
                    ),
                    reason=DecisionReason(
                        type="exception",
                        code="tool_blocked_by_failure_tracker",
                        message="Tool signature hit the tracker blocklist threshold",
                    ),
                )
                await _finalize_pe_outcome(tc, args, tool_source, blocked_outcome, _tool_start)
                continue

            # Unknown tool
            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                unknown_outcome = AllowError(
                    content=f"Error: Unknown tool '{tool_name}'",
                    reason=DecisionReason(
                        type="exception",
                        code="unknown_tool",
                        message=f"Tool '{tool_name}' not in this graph's tool_map",
                    ),
                )
                await _finalize_pe_outcome(tc, args, tool_source, unknown_outcome, _tool_start)
                continue

            # Run RiskAssessor unconditionally for all native tools so that
            # arg_digest / primary_arg are always populated in ToolCallSpec.
            # Previously this was gated on risk_level_meta in ("high","medium"),
            # which left LOW-risk tools with empty arg_digest — meaning a single
            # user "session" or "always" approval would cover ANY args for that
            # tool (P2#2 fix: arg_digest must always be present for correct
            # cache-key scoping by ApprovalStateReader).
            assessment: Any = None
            if not _bypass_risk_gate and tool_source.source == "native":
                assessment = _risk_assessor.assess(tool_name, args)

            # PE-1 §3.2: for skill calls, build SkillCallMetadata from the
            # live ``skill_tool`` registration so PE's SkillSource sees the
            # canonical risk_level / runtime_type / trust_origin / skill_id /
            # content_hash. The metadata SUPERSEDES any caller-prefilled
            # ``risk_assessment`` (Risk #1 hard rule, spec §5.5).
            source_metadata = None
            if tool_source.source == "skill":
                from app.domain.services.permission.sources import (
                    build_skill_call_metadata,
                )
                _skill_tool = configurable.get("skill_tool")
                if _skill_tool is None or not _skill_tool.has_tool(tool_name):
                    _missing_outcome = AllowError(
                        content=(
                            f"PE-1 skill metadata unresolvable for '{tool_name}' "
                            f"(skill_tool missing or unregistered)"
                        ),
                        reason=DecisionReason(
                            type="exception",
                            code="skill_metadata_unresolvable",
                            message="skill_tool not wired or tool not registered",
                        ),
                        retryable=False,
                    )
                    await _finalize_pe_outcome(
                        tc, args, tool_source, _missing_outcome, _tool_start,
                    )
                    continue
                source_metadata = build_skill_call_metadata(
                    tool_name=tool_name,
                    tool_fn=tool_fn,
                    skill_tool=_skill_tool,
                )

            # Build ToolCallSpec for PE evaluation
            call_spec = _build_tool_call_spec_from_tc(
                tc, configurable, tool_source, assessment,
                source_metadata=source_metadata,
            )

            # Build EvaluationContext — read session mode + revision from SSM
            try:
                mode, rev = await _ssm.get_mode_with_revision(_session_id)
            except Exception:
                logger.warning(
                    "SSM.get_mode_with_revision failed for session %s (fail-closed)",
                    _session_id,
                )
                # P1#1 Fail-closed: DO NOT invoke the wrapper — surface an error
                # ToolMessage so the model sees a transient failure and can retry.
                ssm_error_outcome = AllowError(
                    content="[SSM_UNAVAILABLE] 会话状态暂时不可用，请重试（session state unavailable, please retry）",
                    reason=DecisionReason(
                        type="exception",
                        code="ssm_read_failure",
                        message="SSM.get_mode_with_revision raised an exception; failing closed",
                    ),
                    retryable=True,
                )
                await _finalize_pe_outcome(tc, args, tool_source, ssm_error_outcome, _tool_start)
                continue

            from app.domain.services.permission.context import EvaluationContext

            ctx = EvaluationContext(
                session_mode=mode,
                session_mode_revision=rev,
                retry_count=0,
                request_id=configurable.get("request_id", "") or "",
            )

            # ---- PE-0 Phase 10: replay path B (pe_resume_outcomes) ---- #
            # If interrupt_helper already committed a resume outcome for this
            # tool_call_id, skip pe.evaluate and use the cached typed outcome
            # directly (INV-5 path B).  This avoids double-evaluation on the
            # post-resume tool_node replay.
            replay_dict = state.get("pe_resume_outcomes") or {}
            if call_id in replay_dict:
                raw_cached = replay_dict[call_id]
                cached_outcome = TOOL_OUTCOME_ADAPTER.validate_python(raw_cached)
                # Clear the consumed entry from pe_resume_outcomes to keep
                # state small and prevent accidental double-execution.
                consumed_update = {
                    "pe_resume_outcomes": {
                        k: v for k, v in replay_dict.items() if k != call_id
                    },
                }
                # P1 (round-23): re-check session mode before replaying a cached
                # approved outcome.  The approval may have been granted while the
                # session was RUNNING, but by the time tool_node re-enters after the
                # interrupt resume the session could have switched to TAKEOVER or a
                # terminal state (FINISHING/COMPLETED).  Executing approved tools in
                # a non-live session is incorrect — deny instead.
                #
                # `mode` was already fetched by get_mode_with_revision above, so
                # reuse it here without an extra SSM round-trip.
                from app.domain.models.session import SessionStatus as _SessionStatus
                _live_modes = (_SessionStatus.RUNNING, _SessionStatus.WAITING)
                if mode not in _live_modes and isinstance(cached_outcome, (AllowSuccess, Passthrough)):
                    # Session left live mode after approval — convert to Denied.
                    logger.warning(
                        "_pe_dispatch replay: session %s is in non-live mode %s at replay time;"
                        " converting AllowSuccess/Passthrough to Denied for tool_call_id=%s",
                        _session_id,
                        mode.value,
                        call_id,
                    )
                    cached_outcome = Denied(
                        content=(
                            "[REPLAY_DENIED] 会话已切换至非活跃模式，已审批的工具调用被拒绝"
                            f"（session mode changed to {mode.value} before replay）"
                        ),
                        reason=DecisionReason(
                            type="approval_policy",
                            code="session_mode_changed_before_replay",
                            message=f"session is {mode.value} at replay time",
                        ),
                    )
                # For AllowSuccess/Passthrough → invoke the tool wrapper.
                if isinstance(cached_outcome, (AllowSuccess, Passthrough)):
                    result_data = await _invoke_wrapper(
                        tool_fn,
                        tc,
                        tool_source,
                        session_id=call_spec.session_id,
                        max_wrapper_output_bytes=_runtime_max_bytes,
                    )
                    result_data = _maybe_convert_shell_outcome_with_images(result_data, tool_name)
                    await _finalize_pe_outcome(tc, args, tool_source, result_data, _tool_start)
                else:
                    # Denied / AllowError / Asked — surface without invoking wrapper.
                    await _finalize_pe_outcome(tc, args, tool_source, cached_outcome, _tool_start)

                # Merge the state cleanup into the batch-final Command below
                # by continuing the loop (the consumed_update is collected after
                # the loop via a side-channel).  We accumulate it here so the
                # batch-completion update block can merge it.
                # Note: the per-tool continue applies to the normal batch path;
                # we stay in the loop and handle the pe_resume_outcomes cleanup
                # by accumulating into a mutable variable captured below.
                if not hasattr(_pe_dispatch, "_accumulated_consumed"):
                    pass  # consumed_update merged after the loop
                # Store consumed_update for post-loop merge.
                _batch_pe_resume_consumed.update(consumed_update)
                continue
            # ---- end replay path B ---- #

            # P1#4: Shell AST validator gate (N1) — mirrors the legacy tool_node
            # gate at ~line 1864+.  Must run BEFORE pe.evaluate so that shell
            # commands rejected by the AST validator are never presented to PE
            # for grant-based approval.  Without this gate, a malicious shell_execute
            # payload that would have been stopped by Stage S can pass through PE
            # via an existing 'session'/'always' grant or SmartApprove.
            if tool_source.category == "shell" and tool_name == "shell_execute":
                from app.domain.services.safety.shell_ast_validator import (
                    to_typed_denied,
                    validate,
                )
                from app.domain.services.safety.command_policy_evaluator import (
                    build_command_policy,
                    evaluate_command,
                )
                try:
                    _ast_result = validate(
                        command=args.get("command", ""),
                        effective_cwd=(
                            args.get("exec_dir", "") or _settings.sandbox_default_cwd
                        ),
                    )
                except Exception as _ast_exc:  # noqa: BLE001 — defensive
                    logger.exception(
                        "_pe_dispatch AST validator 兜底触发 (should not happen)"
                    )
                    if _metrics is not None:
                        _metrics.record_ast_validator_crash()
                    _ast_crash = AllowError(
                        content=(
                            f"[AST 拦截] validator 内部异常，出于安全原因拒绝本次调用\n"
                            f"命令: {args.get('command', '')[:200]}"
                        ),
                        reason=DecisionReason(
                            type="exception",
                            code="ast_validator_crash",
                            message=str(_ast_exc),
                        ),
                        retryable=False,
                    )
                    await _finalize_pe_outcome(tc, args, tool_source, _ast_crash, _tool_start)
                    continue

                if _metrics is not None:
                    _metrics.record_ast_validation(_ast_result.code)

                # C5a Seam B (PE path): best-effort tool_call policy snapshot emission
                # (still flag-gated + swallowed; the flag gates ONLY this emission, never
                # control flow). C5b: the snapshot reports enforcement_mode="enforce" and
                # the decision below is policy-driven (evaluate_command), not `_ast_result.allowed`.
                if _settings.sandbox_policy_compiler_enabled and (
                    _policy_sink := configurable.get("policy_snapshot_sink")
                ) is not None:
                    try:
                        from app.domain.models.sandbox_policy import (
                            ToolCallInput,
                            ValidationResultView,
                            build_settings_view,
                        )
                        from app.domain.services.safety.sandbox_policy_compiler import (
                            SandboxPolicyCompiler,
                        )

                        _pol_inp = ToolCallInput(
                            session_id=str(configurable.get("session_id") or ""),
                            sandbox_id=None,
                            sandbox_generation=0,
                            worker_type="unknown",
                            depth=0,
                            tool_call_id=str(call_id),
                            tool_name=tool_name,
                            tool_source=tool_source.source,
                            command=args.get("command", ""),
                            validation=ValidationResultView(
                                allowed=_ast_result.allowed,
                                code=_ast_result.code,
                                effective_cwd=_ast_result.effective_cwd,
                            ),
                            is_default_cwd=not bool(args.get("exec_dir")),
                            settings=build_settings_view(_settings),
                        )
                        await _policy_sink.record(
                            SandboxPolicyCompiler().compile_tool_call(_pol_inp)
                        )
                    except Exception as _pol_exc:  # noqa: BLE001 — observe must never alter flow
                        logger.warning(
                            "sandbox.policy observe failed surface=tool_call exc=%s",
                            type(_pol_exc).__name__,
                        )

                _cmd_decision = evaluate_command(
                    validation_code=_ast_result.code,
                    policy=build_command_policy(
                        effective_cwd=_ast_result.effective_cwd,
                        is_default_cwd=not bool(args.get("exec_dir")),
                    ),
                )
                if not _cmd_decision.allowed:
                    _ast_denied = to_typed_denied(
                        _ast_result, original_command=args.get("command", "")
                    )
                    await _finalize_pe_outcome(tc, args, tool_source, _ast_denied, _tool_start)
                    continue
            # — end N1 gate (PE path) —

            # Codex round-20 P2#1: Pre-approved tool calls (approved_tool_call_ids)
            # must bypass pe.evaluate entirely.  The legacy interrupt_helper writes
            # approved IDs when the PE task was unavailable (hot-switch / claim_nonce
            # missing) so the user's confirmation is captured in the legacy state
            # field.  If we still call pe.evaluate here, an ASK/DENY policy would
            # re-prompt the user or deny the call — the user has already approved it.
            #
            # Fix: when _bypass_risk_gate is set (call_id ∈ approved_tool_call_ids),
            # skip pe.evaluate and treat the call as pre-approved AllowSuccess, then
            # invoke the wrapper directly.
            if _bypass_risk_gate:
                logger.debug(
                    "_pe_dispatch: call_id=%s in approved_tool_call_ids — "
                    "skipping pe.evaluate and executing directly (legacy-approved fallback)",
                    call_id,
                )
                # P2 (round-25): re-check session mode for parity with the
                # pe_resume_outcomes replay path (round-23 P1#1).  The user approved
                # the tool while the session was RUNNING/WAITING, but by the time
                # tool_node replays the pre-approved call the session may have
                # entered TAKEOVER/FINISHING/COMPLETED.  Executing tools in a
                # non-live session is incorrect — surface a Denied instead.
                from app.domain.models.session import SessionStatus as _LegacyReplayStatus
                _legacy_live_modes = (_LegacyReplayStatus.RUNNING, _LegacyReplayStatus.WAITING)
                if mode not in _legacy_live_modes:
                    logger.warning(
                        "_pe_dispatch legacy-approved replay: session %s is in "
                        "non-live mode %s at replay time; converting pre-approval "
                        "to Denied for tool_call_id=%s",
                        _session_id,
                        mode.value,
                        call_id,
                    )
                    _mode_denied = Denied(
                        content=(
                            "[LEGACY_REPLAY_DENIED] 会话已切换至非活跃模式，"
                            "已审批的工具调用被拒绝"
                            f"（session mode changed to {mode.value} before replay）"
                        ),
                        reason=DecisionReason(
                            type="approval_policy",
                            code="session_mode_changed_before_replay",
                            message=f"session is {mode.value} at legacy-approved replay time",
                        ),
                    )
                    await _finalize_pe_outcome(tc, args, tool_source, _mode_denied, _tool_start)
                    continue
                _approved_result = await _invoke_wrapper(
                    tool_fn,
                    tc,
                    tool_source,
                    session_id=call_spec.session_id,
                    max_wrapper_output_bytes=_runtime_max_bytes,
                )
                _approved_result = _maybe_convert_shell_outcome_with_images(
                    _approved_result, tool_name
                )
                await _finalize_pe_outcome(tc, args, tool_source, _approved_result, _tool_start)
                continue

            # Evaluate through PE
            # PE-1 §2.7 / §5.1: explicit catches for UnsupportedSource and
            # PEInfrastructureUnavailable MUST come BEFORE the broad
            # ``except Exception`` block — otherwise these typed exceptions
            # would be swallowed by the catch-all and lose their structured
            # decision codes (Round 2 P1#10).
            try:
                pe_outcome = await _pe.evaluate(call_spec, ctx)
            except SessionModeViolation as exc:
                logger.warning(
                    "PE SessionModeViolation for tool '%s' session '%s': %s",
                    tool_name, _session_id, exc,
                )
                lifecycle_outcome = AllowError(
                    content=(
                        f"此操作无法在当前会话状态下执行（工具: {tool_name}）"
                    ),
                    reason=DecisionReason(
                        type="exception",
                        code="session_mode_violation",
                        message=str(exc),
                    ),
                    retryable=False,
                )
                await _finalize_pe_outcome(tc, args, tool_source, lifecycle_outcome, _tool_start)
                continue
            except PolicyConflict as exc:
                logger.warning(
                    "PE PolicyConflict for tool '%s' session '%s': %s",
                    tool_name, _session_id, exc,
                )
                conflict_outcome = AllowError(
                    content=(
                        f"策略冲突，工具调用被阻止（工具: {tool_name}）"
                    ),
                    reason=DecisionReason(
                        type="exception",
                        code="policy_conflict",
                        message=str(exc),
                    ),
                    retryable=False,
                )
                await _finalize_pe_outcome(tc, args, tool_source, conflict_outcome, _tool_start)
                continue
            except UnsupportedSource as exc:
                # PE-1 §2.7 + Round 2 P1#2: caller should have gated via
                # ``is_pe_eligible_tool_source`` (which delegates to
                # ``is_pe_enabled_for_source`` and adds the skill creator/
                # guide category check). Reaching PE with an unregistered
                # source means a caller bug — surface AllowError so the
                # model sees a structured failure and can decide whether to
                # retry/abort. retryable=False because the registry is
                # process-static and won't change mid-request.
                logger.error(
                    "PE UnsupportedSource for tool '%s' source=%r session '%s' "
                    "(caller bug — gate helper should have caught this)",
                    tool_name, exc.source, _session_id,
                )
                unsupported_outcome = AllowError(
                    content=(
                        f"权限引擎不支持此工具来源（工具: {tool_name}, "
                        f"来源: {exc.source}）"
                    ),
                    reason=DecisionReason(
                        type="exception",
                        code="unsupported_tool_source",
                        message=str(exc),
                    ),
                    retryable=False,
                )
                await _finalize_pe_outcome(
                    tc, args, tool_source, unsupported_outcome, _tool_start,
                )
                continue
            except PEInfrastructureUnavailable as exc:
                # PE-1 §5.1 / Round 2 P1#10: Redis / queue / writer crashed
                # mid-evaluate. retryable=True so the agent retry chain can
                # treat this as a transient failure (PolicyConflict above is
                # retryable=False because it indicates a deterministic state
                # mismatch).
                logger.warning(
                    "PE PEInfrastructureUnavailable for tool '%s' session '%s': %s",
                    tool_name, _session_id, exc,
                )
                infra_outcome = AllowError(
                    content=(
                        f"权限引擎基础设施暂时不可用（工具: {tool_name}），请重试"
                    ),
                    reason=DecisionReason(
                        type="exception",
                        code="pe_infrastructure_unavailable",
                        message=str(exc),
                    ),
                    retryable=True,
                )
                await _finalize_pe_outcome(
                    tc, args, tool_source, infra_outcome, _tool_start,
                )
                continue
            except ChildScopeViolation:
                # [C2 PR-2 §5.4] Child scope violations must propagate past this
                # catch-all so CoordinatorChildRunner finalizer (PR-4) can convert
                # to RESULT_READY(needs_authorization). NOT a crash — semantically a deny.
                raise
            except Exception as exc:
                logger.exception(
                    "PE.evaluate unexpected exception for tool '%s'", tool_name
                )
                error_outcome = AllowError(
                    content=f"权限引擎内部异常（工具: {tool_name}）: {exc}",
                    reason=DecisionReason(
                        type="exception",
                        code="pe_evaluate_crash",
                        message=str(exc),
                    ),
                    retryable=False,
                )
                await _finalize_pe_outcome(tc, args, tool_source, error_outcome, _tool_start)
                continue

            # Dispatch based on PE outcome
            if isinstance(pe_outcome, (AllowSuccess, Passthrough)):
                # P1#2 (round 34): re-check session mode before invoking wrapper.
                # pe.evaluate() awaits DB/SSM/policy reads above; the `mode` captured
                # earlier (line ~1389) is stale by the time we get here. If the
                # session switched to TAKEOVER or a terminal state during evaluate(),
                # we must NOT execute the wrapper.
                # Parity with replay path (round 23 P1#1) and the legacy
                # approved_tool_call_ids path (round 25 P2#1) which already do
                # this re-check using the single up-front fetch — here we need a
                # fresh fetch since evaluate() interleaved its own awaits.
                from app.domain.models.session import SessionStatus as _LiveCheckStatus
                _live_modes_for_wrapper = (
                    _LiveCheckStatus.RUNNING,
                    _LiveCheckStatus.WAITING,
                )
                try:
                    _current_mode, _ = await _ssm.get_mode_with_revision(_session_id)
                except Exception:
                    logger.warning(
                        "_pe_dispatch live-mode recheck: SSM.get_mode_with_revision "
                        "failed for session %s (fail-closed before wrapper)",
                        _session_id,
                    )
                    error_outcome = AllowError(
                        content=(
                            "[SSM_UNAVAILABLE] 会话状态暂时不可用，请重试"
                            "（session state unavailable before wrapper execution）"
                        ),
                        reason=DecisionReason(
                            type="exception",
                            code="ssm_read_failure",
                            message=(
                                "SSM.get_mode_with_revision failed during pre-wrapper "
                                "live-mode recheck; failing closed"
                            ),
                        ),
                        retryable=True,
                    )
                    await _finalize_pe_outcome(
                        tc, args, tool_source, error_outcome, _tool_start,
                    )
                    continue

                if _current_mode not in _live_modes_for_wrapper:
                    logger.warning(
                        "_pe_dispatch pre-wrapper: session %s switched to non-live "
                        "mode %s after pe.evaluate; converting AllowSuccess/Passthrough "
                        "to Denied for tool_call_id=%s",
                        _session_id,
                        _current_mode.value,
                        call_id,
                    )
                    denied_outcome = Denied(
                        content=(
                            "[MODE_DENIED] 会话已切换至非活跃模式，工具执行被拒绝"
                            f"（session mode changed to {_current_mode.value} "
                            "before wrapper execution）"
                        ),
                        reason=DecisionReason(
                            type="approval_policy",
                            code="session_mode_changed_before_invoke",
                            message=(
                                f"session is {_current_mode.value} at wrapper "
                                "execution time"
                            ),
                        ),
                    )
                    await _finalize_pe_outcome(
                        tc, args, tool_source, denied_outcome, _tool_start,
                    )
                    continue

                # Mode OK — execute the actual tool
                result_data = await _invoke_wrapper(
                    tool_fn,
                    tc,
                    tool_source,
                    session_id=call_spec.session_id,
                    max_wrapper_output_bytes=_runtime_max_bytes,
                )
                result_data = _maybe_convert_shell_outcome_with_images(result_data, tool_name)
                await _finalize_pe_outcome(tc, args, tool_source, result_data, _tool_start)

            elif isinstance(pe_outcome, (Denied, AllowError)):
                # No execution — emit ToolMessage directly
                await _finalize_pe_outcome(tc, args, tool_source, pe_outcome, _tool_start)

            elif isinstance(pe_outcome, Asked):
                # PE-1 §5.1: rebuild ToolConfirmationEvent from the
                # ConfirmationDetail PE just stored — PE owns the canonical
                # risk_level + matched_patterns (for skill calls these come
                # from PE's recomputed SkillCallMetadata, NOT the legacy
                # native-only RiskAssessment). Native calls fall back to the
                # local assessment when the detail is unavailable so we
                # never crash on a Redis miss / sweeper race.
                _timeout_seconds = configurable.get(
                    "tool_confirmation_timeout_seconds", 300
                )
                detail = None
                if confirmation_manager is not None:
                    try:
                        detail = await confirmation_manager.read(_session_id, call_id)
                    except Exception:
                        logger.warning(
                            "confirmation_manager.read failed for session=%s "
                            "tool_call_id=%s — falling back to assessment",
                            _session_id, call_id,
                            exc_info=True,
                        )
                        detail = None
                if detail is not None:
                    risk_level_str = detail.risk_level
                    matched_patterns = list(detail.matched_patterns)
                else:
                    risk_level_str = (
                        assessment.final_level.name.lower()
                        if assessment is not None else "medium"
                    )
                    matched_patterns = (
                        list(assessment.matched_patterns)
                        if assessment is not None else []
                    )
                # risk_reason: prefer PE's structured DecisionReason.message
                # (carries the canonical skill_risk_high / smart_approve_*
                # explanation); fall back to the local assessment.
                _pe_reason = getattr(pe_outcome, "reason", None)
                risk_reason = (
                    getattr(_pe_reason, "message", None)
                    or getattr(assessment, "risk_reason", "")
                    or ""
                )
                suggested_alternative = (
                    assessment.suggested_alternative
                    if assessment is not None else None
                )
                confirmation_event = ToolConfirmationEvent(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    tool_args=args,
                    risk_level=risk_level_str,
                    risk_reason=risk_reason,
                    matched_patterns=matched_patterns,
                    suggested_alternative=suggested_alternative,
                    timeout_seconds=_timeout_seconds,
                )
                if event_queue:
                    await event_queue.put(confirmation_event)

                # P2#5: Do NOT call confirmation_manager.store() here.
                # PE.evaluate() already called queue.store() (with status=pending +
                # no claim_nonce) before returning Asked.  A second store() here
                # would reset status=pending + clear any claim_nonce set by a
                # racing preflight_resume, causing commit_resume nonce mismatch.

                _pending_outcome = pe_outcome
                _pending_artifact = ToolArtifact(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    tool_source=tool_source,
                    outcome=_pending_outcome,
                )
                _update = {
                    "messages": new_messages + new_deferred_human,
                    "events": new_events,
                    "attempt_count": state["attempt_count"] + 1,
                    "failure_count": state["failure_count"] + new_failures,
                    "completed_tool_call_prefix": (
                        list(already_done) + new_completed_ids
                    ),
                    "pending_ask_outcome": _pending_outcome.model_dump(mode="json"),
                    "pending_ask_tool_call_id": call_id,
                    "pending_ask_artifact": _pending_artifact.model_dump(
                        mode="json", by_alias=True
                    ),
                    "pending_ask_tool_args": dict(args),
                }
                # P2#3: merge already-consumed pe_resume_outcomes cleanup into
                # this early-return Command so that replayed entries from earlier
                # tools in the same batch are not left in state.  Without this,
                # a second interrupt in the same batch would leave stale entries
                # that could match a future tool_call_id with the same name.
                if _batch_pe_resume_consumed:
                    _update.update(_batch_pe_resume_consumed)
                return Command(goto="interrupt_helper", update=_update)

            else:
                logger.error(
                    "PE returned unknown outcome type %s for tool '%s'",
                    type(pe_outcome).__name__, tool_name,
                )
                unknown_err = AllowError(
                    content=f"权限引擎返回未知结果类型（工具: {tool_name}）",
                    reason=DecisionReason(
                        type="exception",
                        code="pe_unknown_outcome",
                        message=f"unknown variant: {type(pe_outcome).__name__}",
                    ),
                    retryable=False,
                )
                await _finalize_pe_outcome(tc, args, tool_source, unknown_err, _tool_start)

        # Batch completed — same happy-path logic as legacy tool_node
        new_messages.extend(new_deferred_human)

        update: dict[str, Any] = {
            "messages": new_messages,
            "events": new_events,
            "attempt_count": state["attempt_count"] + 1,
            "failure_count": state["failure_count"] + new_failures,
            "completed_tool_call_prefix": [],
            "approved_tool_call_ids": [],
            "pending_ask_outcome": None,
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None,
            "pending_ask_tool_args": None,
        }
        # PE-0 Phase 10: clear consumed pe_resume_outcomes entries.
        # Replay path set _batch_pe_resume_consumed with the pruned dict;
        # merge it into the batch update to keep state clean.
        if _batch_pe_resume_consumed:
            update.update(_batch_pe_resume_consumed)
        if should_interrupt:
            update["should_interrupt"] = True
        if not has_prior_soft_hint and any(
            m.content == "SOFT_HINT" and m.name == "message_ask_user"
            for m in new_messages
        ):
            update["soft_hint_sent"] = True

        goto: str = (
            END
            if should_interrupt or update.get("attempt_count", 0) >= MAX_ITERATIONS
            else "pre_llm_node"
        )
        return Command(goto=goto, update=update)

    # ======================================================================
    # End PE-0 Phase 9: _pe_dispatch
    # ======================================================================

    async def tool_node(
        state: ReactGraphState, config: RunnableConfig
    ) -> Command[Literal["pre_llm_node", "interrupt_helper", "__end__"]]:
        """R2 CS2 dispatcher (Commit 1) — executes tool calls from the last AIMessage.

        ## Prefix-closure exactly-once (I-4.1)

        Every entry reads ``state.completed_tool_call_prefix`` and skips
        tool_calls whose ``id`` is already in that set. After a successful batch
        the prefix is reset to ``[]``. When the dispatcher routes to
        ``interrupt_helper`` for an ``Asked`` outcome, it writes the
        already-executed ids into ``completed_tool_call_prefix`` so the
        post-resume replay only runs the pending + remaining tool_calls.

        ## Pre-approved bypass (I-4.2 approve path)

        Tool call ids in ``state.approved_tool_call_ids`` — populated by
        ``interrupt_helper`` on the ``approve`` resume path — skip the
        per-tool risk assessment gate on replay and execute directly. This is
        the approve-resume bridge; the approval grant is persisted post-resume
        (via ``ApprovalStateWriter`` in ``agent_service._resume_tool_confirmation``),
        so the dispatcher cannot rely on a persisted record alone for the replay
        and needs this state flag.

        ## Special handling for ``message_ask_user``

        - ``suggest_user_takeover`` in {"browser", "shell"} → set
          ``should_interrupt`` and short-circuit.
        - Otherwise, first call returns ``SOFT_HINT`` (the agent should try
          autonomously first); second call → truly needs user input, set
          ``should_interrupt``.

        ## Risk assessment gate (Commit 1 transitional)

        For tools with ``risk_level`` in {"high", "medium"} the dispatcher
        runs ``RiskAssessor``. When the assessment resolves to Asked, the
        function **returns** ``Command(goto="interrupt_helper", update=...)``
        after writing the ``pending_ask_*`` state fields. The dispatcher
        itself **never** calls ``interrupt()``; that contract belongs
        exclusively to ``interrupt_helper`` (CS2.13 invariant). For PE-eligible
        batches this transitional gate is superseded by the PermissionEngine
        dispatch (``_pe_dispatch`` → ``_invoke_wrapper`` → ``_translate_outcome``).

        ## Return type

        Always returns ``Command`` — either ``Command(goto="pre_llm_node",
        update=...)`` on the happy path or ``Command(goto="interrupt_helper",
        update=...)`` when a tool call reaches an Asked outcome. This
        replaces the dict return + ``route_after_tool`` conditional edge.
        """
        # [C2 PR-4 §8.4 #6 tool_node_entry] Cancel checkpoint at tool dispatch
        # entry. Raising here aborts BEFORE any tool execution side effect
        # (file_write, shell_execute, ...) is attempted.
        if _should_cancel(config):
            raise CancelledByEventError("tool_node_entry")
        import time as _time
        configurable = (config or {}).get("configurable", {}) if config else {}
        guide_injector = configurable.get("skill_guide_injector")
        event_queue = configurable.get("event_queue")
        confirmation_manager = configurable.get("confirmation_manager")
        _tracker = configurable.get("tool_failure_tracker")
        _metrics = configurable.get("execution_metrics")

        # [C2b §4.3] Child-scope guard — enforce the coordinator child's manifest
        # allowlist + path lease + revision freshness BEFORE any tool side effect.
        # Runs ahead of the PE/legacy fork so it covers the whole pending batch
        # regardless of dispatch path. No-op for root sessions (no cpc in cfg).
        await _enforce_child_scope_or_raise(state, configurable)

        # ---- PE-0 Phase 9 + PE-1 §2.5: Permission Engine dispatch branch ---- #
        # When PE + SSM are wired (built per-task in _create_task), route
        # through DefaultPermissionEngine instead of the legacy
        # risk gate inline code below (the old _run_policy_chain stack was
        # deleted in PE-4a).
        #
        # PE-4c: per-source flags retired — routing is consulted PER CALL
        # inside ``_pe_dispatch`` via
        # ``is_pe_eligible_tool_source(tool_source, tool_confirmation_config)``
        # — which checks source membership + the master ``enabled`` switch AND
        # filters skill creator / skill guide tools (``source="skill"`` but
        # ``category != "skill"``). Calls whose source is not PE-eligible
        # short-circuit back to the legacy tool_node path (None sentinel).
        # Legacy path is preserved verbatim below as the fail-open fallback.
        _pe = configurable.get("permission_engine")
        _ssm = configurable.get("session_state_machine")

        if _pe is not None and _ssm is not None:
            _pe_result = await _pe_dispatch(state, config)
            if _pe_result is not None:
                return _pe_result
            # _pe_result is None: batch contains zero PE-eligible tool
            # calls (either non-PE source OR the master switch off; PE-4c
            # retired the per-source flags); fall through to legacy tool_node
            # path below.

        # ---- End PE-0 Phase 9 / PE-1 §2.5 branch ---- #

        # D5: Cooperative termination — set should_interrupt for routing
        _control = configurable.get("execution_control")
        if _control and _control.should_terminate:
            return Command(
                goto=END,
                update={
                    "should_interrupt": True,
                    "messages": [],
                    "events": [],
                },
            )

        # N1: read settings once for the shell AST validator gate below
        _settings = get_settings()

        messages = state["messages"]

        # R2 CS2 (I-4.1): find the AIMessage carrying the active tool_calls
        # batch. On the happy path ``state.messages[-1]`` is that AIMessage.
        # But on a replay following an ``interrupt_helper`` resume, the most
        # recent message is a ``ToolMessage`` written in the first pass —
        # the triggering AIMessage is further back. Search backward until
        # we hit an AIMessage with tool_calls (or fall through to an empty
        # batch if none is found, which triggers the happy-path return).
        tool_calls: list[dict] = []
        for _msg in reversed(messages):
            if isinstance(_msg, AIMessage) and _msg.tool_calls:
                tool_calls = _msg.tool_calls
                break

        # Check if a SOFT_HINT was already returned in this step
        has_prior_soft_hint = state.get("soft_hint_sent", False)

        # R2 CS2: prefix-closure + pre-approved bypass sets (I-4.1 / I-4.2)
        already_done: set[str] = set(
            state.get("completed_tool_call_prefix", []) or []
        )
        pre_approved: set[str] = set(
            state.get("approved_tool_call_ids", []) or []
        )
        new_completed_ids: list[str] = []

        new_messages: list = []
        new_events: list = []
        should_interrupt = False
        new_failures = 0
        # R2 CS2: Passthrough deferred HumanMessage list, appended AFTER all
        # ToolMessages so the AIMessage → ToolMessage* pairing survives for
        # group_messages() (context_assembler.py). _translate_outcome
        # appends to this; tool_node owns the final flush.
        new_deferred_human: list[HumanMessage] = []

        session_ctx = _session_ctx_from(config)

        async def _finalize_outcome(
            tc: dict,
            tc_args: dict,
            tool_source: ToolSource,
            outcome: ToolOutcome,
            tool_start_ts: float,
        ) -> None:
            """Layer 3 translate + tracker/metrics bookkeeping.

            Shared tail of the execution flow, used by:
            - tracker-blocked synthesis (AllowError)
            - unknown tool synthesis (AllowError)
            - cache deny / smart-approve deny synthesis (Denied)
            - legacy risk gate "allow" path via Layer 2 (_invoke_wrapper)
            - low-risk / no-risk path via Layer 2 (_invoke_wrapper)

            Every path that reaches this helper has a typed ``ToolOutcome``,
            so the ToolMessage that lands in ``new_messages`` carries a
            real typed ``artifact`` field. That is what makes Chunk 4's
            LLM adapter prefix injection (``[TOOL_FAILED: timeout]`` /
            ``[TOOL_DENIED: ast_validator]``) actually fire in
            production — the pre-fix dispatcher handed the adapter bare
            strings with ``artifact=None``, so the adapter always fell
            back to the generic ``[TOOL_ERROR]``.
            """
            nonlocal new_failures

            msg, deferred, events = await _translate_outcome(
                outcome,
                tc,
                tool_source,
                session_ctx,
                tool_result_max_chars=tool_result_max_chars,
                guide_injector=guide_injector,
                enabled_outcome_variants=_tool_runtime_cfg.enabled_outcome_variants,
            )
            if msg is not None:
                new_messages.append(msg)
            new_deferred_human.extend(deferred)
            for evt in events:
                new_events.append(evt)

            is_success = isinstance(outcome, (AllowSuccess, Passthrough))
            tc_name = tc["name"]
            if _tracker:
                if is_success:
                    _tracker.record_success(tc_name, tc_args)
                else:
                    _tracker.record_failure(tc_name, tc_args)
            if _metrics:
                _metrics.record_tool_call(
                    success=is_success,
                    latency_ms=(_time.monotonic() - tool_start_ts) * 1000,
                )
            if not is_success:
                new_failures += 1

            new_completed_ids.append(tc["id"])

        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            call_id = tc["id"]
            _tool_start = _time.monotonic()

            # R2 CS2 (I-4.1): skip tool_calls already executed in a prior
            # dispatcher entry — LangGraph replays the node on resume after
            # every interrupt, and the prefix must not re-run.
            if call_id in already_done:
                continue

            # R2 CS2 (I-4.2): pre-approved tool_calls bypass the risk gate
            # entirely on the replay following an approve resume.
            _bypass_risk_gate = call_id in pre_approved

            # ---- message_ask_user: SOFT_HINT gating ----
            # Pseudo-tool for user-input gating, not a real wrapper — it
            # stays on the legacy manual ToolMessage construction path
            # because it has no ``ToolSource`` entry and no typed artifact
            # semantics. SOFT_HINT / WAITING_FOR_USER strings are
            # consumed by agent_task_runner / interfaces layer, not the
            # LLM adapter's prefix logic.
            if tool_name == "message_ask_user":
                suggest = str(args.get("suggest_user_takeover", "none")).strip().lower()
                if suggest in {"browser", "shell"}:
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                elif not has_prior_soft_hint:
                    result_str = "SOFT_HINT"
                    logger.info("message_ask_user: returning SOFT_HINT (first attempt)")
                else:
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                    logger.info("message_ask_user: user input required (after SOFT_HINT)")

                new_messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )
                new_events.append(
                    ToolEvent(
                        tool_call_id=call_id,
                        tool_name=resolve_tool_source(tool_name).category,
                        function_name=tool_name,
                        function_args=args,
                        function_result=ToolResult(success=True, message=result_str),
                        status=ToolEventStatus.CALLED,
                    )
                )
                new_completed_ids.append(call_id)
                continue

            # Resolve the tool's ``ToolSource`` once — every downstream
            # branch (block, unknown, deny, execute) needs it for
            # ``_translate_outcome``. Fall through to the ``"unknown"``
            # sentinel ``ToolSource`` if R1 has never seen this name
            # (e.g. an LLM-hallucinated tool, or a dynamically-discovered
            # MCP tool whose factory hasn't registered a canonical
            # identity yet).
            #
            # ``"unknown"`` is a first-class category in
            # ``KNOWN_CATEGORIES`` (R2 CS2 addition) and lives under
            # ``IDENTITY_ONLY_CATEGORIES`` in
            # ``test_agent_task_runner_enrichment_contract.py``. Because
            # no ``_handle_tool_event`` branch matches ``"unknown"``,
            # ``tool_content`` stays ``None`` and the raw
            # ``Error: Unknown tool 'xxx'`` message surfaces to the UI —
            # no shell console leak, no file sync side effects.
            try:
                tool_source = resolve_tool_source(tool_name)
            except ToolSourceUnknownError:
                tool_source = ToolSource(
                    source="native",
                    category="unknown",
                    canonical_name=tool_name,
                )

            # N1 AST validator gate (shell_execute only, before legacy risk gate).
            #
            # Narrow scope — ``tool_source.category == "shell"`` also covers
            # shell_read_output / shell_wait_process / shell_write_input /
            # shell_kill_process, none of which carry a fresh ``command`` arg.
            # Running ``validate("")`` on them inflated ``parser_failure_rate``
            # denominator with empty-command successes and (for
            # shell_write_input) masked the actually-dangerous ``input_text``
            # payload as an AST pass. Scoping by tool_name keeps the metric
            # meaningful; ``shell_write_input.input_text`` validation is a
            # follow-up rather than being smuggled into this gate.
            if tool_source.category == "shell" and tool_name == "shell_execute":
                from app.domain.services.safety.shell_ast_validator import (
                    to_typed_denied,
                    validate,
                )
                from app.domain.services.safety.command_policy_evaluator import (
                    build_command_policy,
                    evaluate_command,
                )
                try:
                    ast_result = validate(
                        command=args.get("command", ""),
                        effective_cwd=(
                            args.get("exec_dir", "") or _settings.sandbox_default_cwd
                        ),
                    )
                except Exception as _ast_exc:  # noqa: BLE001 — defensive
                    logger.exception(
                        "tool_node AST validator 兜底触发 (should not happen)"
                    )
                    # P1-b: Layer-2 crash is a DISTINCT P0 signal (spec §6.5);
                    # do NOT fold into parser_failure_rate. Use dedicated counter.
                    if _metrics is not None:
                        _metrics.record_ast_validator_crash()
                    crash_outcome = AllowError(
                        content=(
                            f"[AST 拦截] validator 内部异常，出于安全原因拒绝本次调用\n"
                            f"命令: {args.get('command', '')[:200]}"
                        ),
                        reason=DecisionReason(
                            type="exception",
                            code="ast_validator_crash",
                            message=str(_ast_exc),
                        ),
                        retryable=False,
                    )
                    await _finalize_outcome(tc, args, tool_source, crash_outcome, _tool_start)
                    continue

                if _metrics is not None:
                    _metrics.record_ast_validation(ast_result.code)

                # C5a Seam B (legacy tool_node path): best-effort tool_call snapshot
                # emission (flag-gated + swallowed). C5b: the snapshot reports
                # enforcement_mode="enforce"; the decision below is policy-driven
                # (evaluate_command), not `ast_result.allowed`.
                if _settings.sandbox_policy_compiler_enabled and (
                    _policy_sink := configurable.get("policy_snapshot_sink")
                ) is not None:
                    try:
                        from app.domain.models.sandbox_policy import (
                            ToolCallInput,
                            ValidationResultView,
                            build_settings_view,
                        )
                        from app.domain.services.safety.sandbox_policy_compiler import (
                            SandboxPolicyCompiler,
                        )

                        _pol_inp = ToolCallInput(
                            session_id=str(configurable.get("session_id") or ""),
                            sandbox_id=None,
                            sandbox_generation=0,
                            worker_type="unknown",
                            depth=0,
                            tool_call_id=str(call_id),
                            tool_name=tool_name,
                            tool_source=tool_source.source,
                            command=args.get("command", ""),
                            validation=ValidationResultView(
                                allowed=ast_result.allowed,
                                code=ast_result.code,
                                effective_cwd=ast_result.effective_cwd,
                            ),
                            is_default_cwd=not bool(args.get("exec_dir")),
                            settings=build_settings_view(_settings),
                        )
                        await _policy_sink.record(
                            SandboxPolicyCompiler().compile_tool_call(_pol_inp)
                        )
                    except Exception as _pol_exc:  # noqa: BLE001 — observe must never alter flow
                        logger.warning(
                            "sandbox.policy observe failed surface=tool_call exc=%s",
                            type(_pol_exc).__name__,
                        )

                _cmd_decision = evaluate_command(
                    validation_code=ast_result.code,
                    policy=build_command_policy(
                        effective_cwd=ast_result.effective_cwd,
                        is_default_cwd=not bool(args.get("exec_dir")),
                    ),
                )
                if not _cmd_decision.allowed:
                    denied = to_typed_denied(ast_result, original_command=args.get("command", ""))
                    await _finalize_outcome(tc, args, tool_source, denied, _tool_start)
                    continue
            # — end N1 gate —

            # ---- D5 tracker: block signature with repeated failures ----
            if _tracker and _tracker.is_blocked(tool_name, args):
                blocked_outcome = AllowError(
                    content=(
                        f"[BLOCKED] 此工具调用模式（{tool_name}）因连续失败已被暂停，"
                        "请尝试不同的工具或参数"
                    ),
                    reason=DecisionReason(
                        type="exception",
                        code="tool_blocked_by_failure_tracker",
                        message="Tool signature hit the tracker blocklist threshold",
                    ),
                )
                await _finalize_outcome(
                    tc, args, tool_source, blocked_outcome, _tool_start
                )
                continue

            # ---- Unknown tool: synthesize AllowError, let Layer 3 translate ----
            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                unknown_outcome = AllowError(
                    content=f"Error: Unknown tool '{tool_name}'",
                    reason=DecisionReason(
                        type="exception",
                        code="unknown_tool",
                        message=f"Tool '{tool_name}' not in this graph's tool_map",
                    ),
                )
                await _finalize_outcome(
                    tc, args, tool_source, unknown_outcome, _tool_start
                )
                continue

            # PE-4c: per-call execution context (the legacy native risk gate is removed).
            _tc_enabled = configurable.get("tool_confirmation_enabled", True)
            _session_id = configurable.get("session_id") or ""
            _runtime_max_bytes = _tool_runtime_cfg.max_wrapper_output_bytes

            # PE-2 §6: full config + PE-present flags for the MCP mixed-batch guard.
            _tc_config = configurable.get("tool_confirmation_config")
            _pe_present = (
                configurable.get("permission_engine") is not None
                and configurable.get("session_state_machine") is not None
            )

            # PE-1b fail-closed guard: a dynamic SkillTool (source == category ==
            # "skill") only reaches the legacy path via a mixed-batch fallback — a
            # non-PE-eligible call (skill creator/guide, mcp discovery, an
            # unsupported source, or the master switch off) in the same batch
            # forced the whole batch off PE (per-batch gate → None). PE-1b deleted
            # the legacy R3 skill confirmation and the native gate below excludes
            # skills, so without this guard the skill would execute UNCONFIRMED. Deny
            # fail-closed; the agent re-issues the skill in its own batch, which
            # _pe_dispatch routes through PE + SkillSource. Keys on source/category
            # only (never skill risk metadata) so INV-6 stays satisfied; skill
            # creator/guide (category != "skill") are unaffected.
            if (
                not _bypass_risk_gate
                and _tc_enabled
                and tool_source
                and tool_source.source == "skill"
                and tool_source.category == "skill"
            ):
                _fail_closed = Denied(
                    content=(
                        f"Skill 工具 '{tool_name}' 无法与非权限引擎工具（MCP/A2A 等）"
                        "在同一批次中执行；请在单独的步骤中调用该 Skill。"
                    ),
                    reason=DecisionReason(
                        type="approval_policy",
                        code="skill_mixed_batch_fail_closed",
                        message=(
                            "dynamic skill reached legacy via mixed-batch fallback; "
                            "PE confirmation required"
                        ),
                    ),
                )
                await _finalize_outcome(
                    tc, args, tool_source, _fail_closed, _tool_start
                )
                continue

            # PE-2 §6: a PE-eligible MCP real tool must never execute via the
            # legacy fallback. A mixed batch (mcp + a2a / skill-creator / discovery)
            # forces the WHOLE batch to legacy (react_graph.py per-batch gate →
            # None); without this guard the MCP call reaches the direct-execute
            # point below and runs UNCONFIRMED, bypassing any user ASK/DENY policy
            # the PE path honors. MCP-specific (source/category) so native/skill/
            # a2a are untouched; gated on PE-present + master switch on (with mcp
            # ∈ PE_SUPPORTED_SOURCES; PE-4c retired the per-source flags) so
            # master-OFF / PE-absent fall through to legacy passthrough (§8 soft
            # rollback). INV-6 clean (source/category only). The agent
            # re-sends the MCP tool alone (driven by .content) → _pe_dispatch
            # routes it through PE + McpSource.
            if (
                not _bypass_risk_gate
                and tool_source
                and tool_source.source == "mcp"
                and tool_source.category == "mcp"
                and _pe_present
                and is_pe_enabled_for_source("mcp", _tc_config)
            ):
                _fail_closed = Denied(
                    content=(
                        f"MCP 工具 '{tool_name}' 无法与非权限引擎工具（A2A / skill creator 等）"
                        "在同一批次中执行；请在单独的步骤中调用该 MCP 工具。"
                    ),
                    reason=DecisionReason(
                        type="approval_policy",
                        code="mcp_mixed_batch_fail_closed",
                        message="MCP reached legacy via mixed-batch fallback; PE routing required",
                    ),
                )
                await _finalize_outcome(tc, args, tool_source, _fail_closed, _tool_start)
                continue

            # PE-3: a PE-eligible A2A tool must never execute via the legacy
            # fallback. A mixed batch (a2a + skill-creator/guide / mcp-discovery)
            # forces the WHOLE batch to legacy; without this guard the A2A call
            # reaches the direct-execute point below and runs UNCONFIRMED,
            # bypassing any user ASK/DENY policy the PE path honors. A2A-specific
            # (source/category) so native/skill/mcp are untouched; gated on
            # PE-present + master switch on (with a2a ∈ PE_SUPPORTED_SOURCES;
            # PE-4c retired the per-source flags) so master-OFF / PE-absent fall
            # through to legacy passthrough (§8 soft rollback). INV-6 clean
            # (source/category only). Agent re-sends A2A alone (driven by
            # .content) → _pe_dispatch routes it through PE + A2aSource.
            if (
                not _bypass_risk_gate
                and tool_source
                and tool_source.source == "a2a"
                and tool_source.category == "a2a"
                and _pe_present
                and is_pe_enabled_for_source("a2a", _tc_config)
            ):
                _fail_closed = Denied(
                    content=(
                        f"A2A 工具 '{tool_name}' 无法与非权限引擎工具（skill creator / MCP discovery 等）"
                        "在同一批次中执行；请在单独的步骤中调用该 A2A 工具。"
                    ),
                    reason=DecisionReason(
                        type="approval_policy",
                        code="a2a_mixed_batch_fail_closed",
                        message="A2A reached legacy via mixed-batch fallback; PE routing required",
                    ),
                )
                await _finalize_outcome(tc, args, tool_source, _fail_closed, _tool_start)
                continue

            # PE-4b §3: parity with the PE pre-approved replay recheck (:1762).
            # A native+meta batch can legitimately enter legacy pre-approved
            # replay: meta forces batch fallback → HTTP preflight routes to legacy
            # → interrupt_helper writes approved_tool_call_ids (no claim_nonce,
            # :3235) → replay falls back again → the native call must execute.
            # The mixed-batch guard below correctly excludes _bypass_risk_gate so
            # pre-approved native is NOT denied by it. BUT the legacy direct-execute
            # (:2964) does not recheck session live-mode, while the PE path does.
            # A native tool approved while the session was RUNNING/WAITING may, by
            # replay time, have entered TAKEOVER/FINISHING/COMPLETED — executing in
            # a non-live session is incorrect. Guard the legacy pre-approved
            # native direct-execute the same way. Native-specific (source) so the
            # skill/mcp/a2a pre-approved replays are unaffected.
            if _bypass_risk_gate and tool_source and tool_source.source == "native":
                from app.domain.models.session import SessionStatus as _NativeReplayStatus
                _native_live_modes = (
                    _NativeReplayStatus.RUNNING,
                    _NativeReplayStatus.WAITING,
                )
                try:
                    _native_mode, _ = await _ssm.get_mode_with_revision(_session_id)
                except Exception:
                    logger.warning(
                        "tool_node legacy pre-approved native recheck: "
                        "SSM.get_mode_with_revision failed for session %s "
                        "(fail-closed before wrapper)",
                        _session_id,
                    )
                    _native_ssm_err = AllowError(
                        content=(
                            "[SSM_UNAVAILABLE] 会话状态暂时不可用，请重试"
                            "（session state unavailable before pre-approved native replay）"
                        ),
                        reason=DecisionReason(
                            type="exception",
                            code="ssm_read_failure",
                            message=(
                                "SSM.get_mode_with_revision failed during legacy "
                                "pre-approved native live-mode recheck; failing closed"
                            ),
                        ),
                        retryable=True,
                    )
                    await _finalize_outcome(
                        tc, args, tool_source, _native_ssm_err, _tool_start
                    )
                    continue
                if _native_mode not in _native_live_modes:
                    logger.warning(
                        "tool_node legacy pre-approved native replay: session %s "
                        "is in non-live mode %s at replay time; converting "
                        "pre-approval to Denied for tool_call_id=%s",
                        _session_id,
                        _native_mode.value,
                        call_id,
                    )
                    _native_mode_denied = Denied(
                        content=(
                            "[LEGACY_REPLAY_DENIED] 会话已切换至非活跃模式，"
                            "已审批的原生工具调用被拒绝"
                            f"（session mode changed to {_native_mode.value} before replay）"
                        ),
                        reason=DecisionReason(
                            type="approval_policy",
                            code="session_mode_changed_before_replay",
                            message=f"session is {_native_mode.value} at legacy-approved native replay time",
                        ),
                    )
                    await _finalize_outcome(
                        tc, args, tool_source, _native_mode_denied, _tool_start
                    )
                    continue

            # PE-4b §3: a PE-eligible REAL native tool must never execute via the
            # legacy fallback. A mixed batch (native + skill-creator/guide /
            # mcp-discovery / unknown) forces the WHOLE batch to legacy (per-batch
            # gate → None); without this guard the native call reaches the
            # direct-execute point below and runs UNCONFIRMED, bypassing the PE
            # risk gate. Native-specific (source) so skill/mcp/a2a are untouched.
            # category != "unknown" excludes the :2438 unresolvable sentinel
            # (ToolSource(source="native", category="unknown")) — a permanent
            # passthrough, NOT a real native tool. Gated on _tc_enabled (master
            # switch) + _pe_present, so master-OFF / PE-absent leave the guard inert
            # and the native call direct-executes below. NB (PE-4c): master-ON +
            # PE-absent is UNREACHABLE — _create_task (agent_service) fails CLOSED
            # when confirmation is required but PE cannot be built, so PE-absent
            # here always implies master-OFF (confirmation disabled by explicit
            # operator choice). The legacy native risk gate that used to backstop
            # the PE-absent case was deleted in PE-4c. INV-6 clean (source + flag
            # only, never reads skill risk metadata). _bypass_risk_gate (pre-approved
            # replay) is intentionally excluded — see the pre-approved native
            # recheck above. The agent re-sends the native tool alone (driven by
            # .content) → _pe_dispatch routes it through PE.
            if (
                not _bypass_risk_gate
                and _tc_enabled
                and tool_source
                and tool_source.source == "native"
                and tool_source.category != "unknown"
                and _pe_present
                and is_pe_enabled_for_source("native", _tc_config)
            ):
                _fail_closed = Denied(
                    content=(
                        f"原生工具 '{tool_name}' 无法与非权限引擎工具（skill creator / "
                        "MCP discovery 等）在同一批次中执行；请在单独的步骤中调用该工具。"
                    ),
                    reason=DecisionReason(
                        type="approval_policy",
                        code="native_mixed_batch_fail_closed",
                        message="native reached legacy via mixed-batch fallback; PE routing required",
                    ),
                )
                await _finalize_outcome(tc, args, tool_source, _fail_closed, _tool_start)
                continue

            # No risk metadata (or bypass via pre-approved) → execute directly
            outcome = await _invoke_wrapper(
                tool_fn,
                tc,
                tool_source,
                session_id=_session_id,
                max_wrapper_output_bytes=_runtime_max_bytes,
            )
            outcome = _maybe_convert_shell_outcome_with_images(
                outcome, tool_name
            )
            await _finalize_outcome(
                tc, args, tool_source, outcome, _tool_start
            )

        # Append deferred HumanMessages AFTER all ToolMessages.
        # Preserves AIMessage → ToolMessage* pairing for group_messages().
        new_messages.extend(new_deferred_human)

        # R2 CS2 happy path: the whole batch completed without hitting an
        # Asked outcome. Reset prefix + pre-approved set + pending state
        # and hand control back to pre_llm_node for the next LLM turn.
        update: dict[str, Any] = {
            "messages": new_messages,
            "events": new_events,
            "attempt_count": state["attempt_count"] + 1,
            "failure_count": state["failure_count"] + new_failures,
            "completed_tool_call_prefix": [],
            "approved_tool_call_ids": [],
            "pending_ask_outcome": None,
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None,
            "pending_ask_tool_args": None,
        }
        if should_interrupt:
            update["should_interrupt"] = True
        if not has_prior_soft_hint and any(
            m.content == "SOFT_HINT" and m.name == "message_ask_user"
            for m in new_messages
        ):
            update["soft_hint_sent"] = True

        # R2 CS2: tool_node routes itself via Command; no conditional edge.
        goto: str = (
            END
            if should_interrupt or update.get("attempt_count", 0) >= MAX_ITERATIONS
            else "pre_llm_node"
        )
        # [C2 PR-4 §8.4 #7 tool_node_return] Cancel checkpoint at tool_node
        # exit. By here the tool already executed; raising abandons the
        # tool's just-built ToolMessage rather than feeding it to the next
        # llm_node iteration. The cancel finalizer doesn't care about the
        # residual message — it builds CANCEL_ACK from the existing state.
        if _should_cancel(config):
            raise CancelledByEventError("tool_node_return")
        return Command(goto=goto, update=update)

    # ---- Routing ------------------------------------------------------- #

    def route_after_llm(state: ReactGraphState) -> str:
        """Route after LLM call: tool calls → tool_node, else END."""
        if state.get("should_interrupt"):
            return END

        messages = state["messages"]
        if not messages:
            return END

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tool_node"

        return END

    async def _legacy_interrupt_helper_resume(
        state: ReactGraphState,
        user_response: dict,
        pending_id: str,
        pending_artifact_dict: dict,
        config: RunnableConfig,
    ) -> Command:  # type: ignore[type-arg]
        """Legacy (fail-open) resume path for interrupt_helper.

        Preserves the original approve/deny logic that writes to
        ``approved_tool_call_ids``.  Used when PE is unwired or the feature
        flag is off.  PE-3 cleanup will remove this once all 4 tool sources
        (native/skill/MCP/A2A) have migrated to the ``pe_resume_outcomes``
        path.
        """
        action = (
            user_response.get("action", "deny")
            if isinstance(user_response, dict)
            else "deny"
        )

        if action == "approve":
            logger.info(
                "interrupt_helper (legacy): user approved tool_call %s", pending_id
            )
            return Command(
                goto="tool_node",
                update={
                    "approved_tool_call_ids": (
                        list(state.get("approved_tool_call_ids", []) or [])
                        + [pending_id]
                    ),
                    "pending_ask_outcome": None,
                    "pending_ask_tool_call_id": None,
                    "pending_ask_artifact": None,
                    "pending_ask_tool_args": None,
                },
            )

        # deny / timeout_fallback: synthesize a Denied outcome via Layer 3,
        # emit the ToolMessage, and mark the id completed so the batch
        # moves on to the next tool_call on replay.
        deny_reason_code = action  # "deny" | "timeout_fallback"
        tool_name = pending_artifact_dict["tool_name"]
        tool_source_dict = pending_artifact_dict["tool_source"]
        # R2 CS2: use the original args persisted by tool_node when the
        # Asked was raised (pending_ask_tool_args). Fallback to empty dict
        # only if the state carries no args (older checkpoint).
        pending_args = state.get("pending_ask_tool_args") or {}
        logger.info(
            "interrupt_helper (legacy): user %s tool_call %s (tool=%s)",
            action,
            pending_id,
            tool_name,
        )

        denied_outcome = Denied(
            content=(
                "用户拒绝了此操作"
                if action == "deny"
                else "操作因超时被跳过"
            ),
            reason=DecisionReason(
                type="approval_policy",
                code=deny_reason_code,
                message=f"interrupt_helper action={action}",
            ),
        )
        tool_source_obj = ToolSource.model_validate(tool_source_dict)
        fake_tool_call: ToolCall = {
            "id": pending_id,
            "name": tool_name,
            "args": dict(pending_args),
            "type": "tool_call",
        }
        _configurable = (config or {}).get("configurable", {}) if config else {}
        tool_result_max_chars = _configurable.get("tool_result_max_chars", 8000)
        guide_injector = _configurable.get("skill_guide_injector")

        msg, deferred, deny_events = await _translate_outcome(
            denied_outcome,
            fake_tool_call,
            tool_source_obj,
            _session_ctx_from(config),
            tool_result_max_chars=tool_result_max_chars,
            guide_injector=guide_injector,
            enabled_outcome_variants=_tool_runtime_cfg.enabled_outcome_variants,
        )

        # NOTE: deny_events are regular ToolEvents (not ToolConfirmationEvents).
        # Per the event_bridge.py:103-107 contract, regular events travel
        # through the state-update path and the bridge forwards them to
        # the SSE queue automatically. Pushing them to event_queue here as
        # well would cause double-emission. Only urgent live events
        # (ToolConfirmationEvent in tool_node) use event_queue.put().
        new_messages: list = []
        if msg is not None:
            new_messages.append(msg)
        new_messages.extend(deferred)

        return Command(
            goto="tool_node",
            update={
                "messages": new_messages,
                "events": deny_events,
                "completed_tool_call_prefix": (
                    list(state.get("completed_tool_call_prefix", []) or [])
                    + [pending_id]
                ),
                "pending_ask_outcome": None,
                "pending_ask_tool_call_id": None,
                "pending_ask_artifact": None,
                "pending_ask_tool_args": None,
            },
        )

    async def interrupt_helper(
        state: ReactGraphState, config: RunnableConfig
    ) -> Command[Literal["tool_node"]]:
        """R2 CS2 — the only node allowed to call ``interrupt()``.

        Receives control from ``tool_node`` whenever a tool_call reaches an
        ``Asked`` outcome (PE evaluate via ``_pe_dispatch`` or the legacy
        risk gate / wrapper). Reads
        the ``pending_ask_*`` state written by ``tool_node``, calls
        ``interrupt(...)`` to pause the graph, and on resume dispatches:

        **PE path (when permission_engine + session_state_machine are wired):**

        - Calls ``pe.commit_resume`` with the resume payload.
        - Writes typed ``ToolOutcome`` into ``pe_resume_outcomes[tool_call_id]``.
        - ``tool_node`` reads this on replay (INV-5 path B) and skips
          ``pe.evaluate``.

        **Legacy fail-open path (PE unwired or feature flag off):**

        - ``approve`` → add ``pending_id`` to ``approved_tool_call_ids`` and
          return to ``tool_node``. On the replay, ``tool_node`` sees the id
          in its pre-approved set and skips the risk gate entirely.
        - ``deny`` / ``timeout_fallback`` → construct a typed ``Denied``
          outcome, translate it into a ``ToolMessage`` via
          ``_translate_outcome``, mark the id as completed in the prefix,
          and return to ``tool_node`` so the next tool_call in the batch
          can proceed.

        **CS2.13 invariant**: this function must NOT read
        ``state.messages[-1].tool_calls``, must NOT call
        ``_invoke_wrapper`` (or any tool-dispatch helper), and must NOT execute
        wrappers. Its sole job is the interrupt handshake and routing.
        """
        _early = _interrupt_helper_early_return(state)
        if _early is not None:
            return _early
        pending_id = state["pending_ask_tool_call_id"]
        pending_artifact_dict = state["pending_ask_artifact"]
        assert pending_id is not None  # narrowed by _interrupt_helper_early_return
        assert pending_artifact_dict is not None

        # interrupt() here. On the first invocation LangGraph raises
        # GraphInterrupt, the client surfaces the confirmation card, the
        # user sends a resume command, and LangGraph replays the node —
        # the second invocation of interrupt() returns the resume value
        # without raising. The body must be idempotent across replays.
        user_response = interrupt(
            {
                "type": "tool_confirmation",
                "tool_call_id": pending_id,
                "ask": state.get("pending_ask_outcome"),
                "artifact": pending_artifact_dict,
                "completed_prefix": state.get(
                    "completed_tool_call_prefix", []
                ),
            }
        )

        # ---- PE-0 Phase 10: PE path vs legacy fail-open path ---- #
        configurable = (config or {}).get("configurable", {}) if config else {}
        pe = configurable.get("permission_engine")
        ssm = configurable.get("session_state_machine")

        if pe is None or ssm is None:
            # Fail-open: PE unwired or feature flag off. Delegate to the
            # legacy resume helper which preserves all existing behavior
            # (writes to approved_tool_call_ids). Both PE and legacy paths
            # coexist via state.approved_tool_call_ids (kept in PE-0) and
            # state.pe_resume_outcomes (new PE path). PE-3 cleanup removes
            # the legacy path once all 4 sources have migrated.
            return await _legacy_interrupt_helper_resume(
                state, user_response, pending_id, pending_artifact_dict, config
            )

        # PE path: commit_resume + write pe_resume_outcomes
        user_response_dict = user_response if isinstance(user_response, dict) else {}
        action = user_response_dict.get("action", "deny")
        scope = user_response_dict.get("scope", "once")
        claim_nonce = user_response_dict.get("claim_nonce")
        tool_call_id = user_response_dict.get("tool_call_id") or pending_id

        # If no claim_nonce in payload, the PE path cannot validate the resume
        # claim — fall back to legacy path to avoid a hard failure.
        if claim_nonce is None:
            logger.warning(
                "interrupt_helper: no claim_nonce in resume payload for tool_call %s "
                "(falling back to legacy path)",
                tool_call_id,
            )
            return await _legacy_interrupt_helper_resume(
                state, user_response, pending_id, pending_artifact_dict, config
            )

        # Rehydrate ToolCallSpec from pending_ask_* state, then fill in
        # user_id / session_id from configurable (not stored in state).
        from app.domain.services.permission.tool_call_spec import ToolCallSpec  # local import
        call_spec = _rehydrate_call_spec(state, tool_call_id)
        _user_id = configurable.get("user_id") or ""
        _session_id = configurable.get("session_id") or ""

        # P1#2: _rehydrate_call_spec reads tool_name/tool_source/tool_args from
        # pending_ask_artifact but does NOT restore arg_digest/primary_arg/dir_arg
        # because ToolArtifact schema does not carry those fields.  pe.commit_resume
        # compares call.arg_digest against detail.arg_digest — mismatch → PolicyConflict.
        # Fix: read the ConfirmationDetail from the queue (which was written by
        # _pe_dispatch at evaluate time with the canonical arg_digest) and use its
        # values as the authoritative source.  This also covers the case where the
        # queue was not yet read in the current interrupt_helper invocation.
        _rehydrate_primary_arg = call_spec.primary_arg
        _rehydrate_dir_arg = call_spec.dir_arg
        _rehydrate_arg_digest = call_spec.arg_digest
        try:
            _queue_detail = await pe._queue.read(_session_id, tool_call_id)  # type: ignore[attr-defined]
            if _queue_detail is not None:
                _rehydrate_primary_arg = _queue_detail.primary_arg or None
                _rehydrate_dir_arg = _queue_detail.dir_arg or None
                _rehydrate_arg_digest = _queue_detail.arg_digest or None
        except Exception:
            logger.warning(
                "interrupt_helper: queue.read failed for %s:%s; "
                "using state-derived arg_digest (may trigger arg_digest_mismatch)",
                _session_id, tool_call_id,
            )

        # Replace the empty user_id / session_id stubs with the real values.
        call_spec = ToolCallSpec(
            tool_name=call_spec.tool_name,
            tool_args=call_spec.tool_args,
            tool_source=call_spec.tool_source,
            user_id=_user_id,
            session_id=_session_id,
            tool_call_id=call_spec.tool_call_id,
            primary_arg=_rehydrate_primary_arg,
            dir_arg=_rehydrate_dir_arg,
            arg_digest=_rehydrate_arg_digest,
            risk_assessment=call_spec.risk_assessment,
        )

        try:
            mode, rev = await ssm.get_mode_with_revision(_session_id)
        except Exception as _ssm_exc:
            # P1#1 (Codex round-10): if PE preflight already wrote a claim_nonce
            # (HTTP preflight success → PE path), we must NOT fall back to legacy
            # on SSM failure.  Legacy approve writes approved_tool_call_ids which
            # bypasses commit_resume's nonce/mode validation, executes the tool
            # without writing a grant/audit, and leaves the Redis confirmation
            # stuck in 'processing' while the sweeper may reopen it.
            # Fail closed instead: return a resume error command.
            if claim_nonce is not None:
                logger.warning(
                    "interrupt_helper: SSM.get_mode_with_revision failed for session %s "
                    "with active PE claim_nonce — fail closed to protect nonce integrity",
                    _session_id,
                )
                # P2 (round-18): PE preflight already marked the queue entry
                # 'processing'.  SSM failure means commit_resume will never run,
                # so we must cleanup the entry here to prevent it from being stuck
                # in 'processing' and blocking future /resume attempts.
                try:
                    await pe.cleanup_pending_confirmation(_session_id, tool_call_id)
                except Exception:
                    logger.exception(
                        "interrupt_helper: cleanup_pending_confirmation failed on "
                        "SSM error path for %s:%s", _session_id, tool_call_id,
                    )
                return _build_resume_error_command(state, _ssm_exc)
            logger.warning(
                "interrupt_helper: SSM.get_mode_with_revision failed for session %s "
                "(no PE claim, falling back to legacy path)",
                _session_id,
            )
            return await _legacy_interrupt_helper_resume(
                state, user_response, pending_id, pending_artifact_dict, config
            )

        from app.domain.services.permission.context import EvaluationContext, ResumeSignal
        from app.domain.services.permission.errors import (
            PolicyConflict,
            SessionModeViolation,
            WriterIntegrityError,
        )

        ctx = EvaluationContext(
            session_mode=mode,
            session_mode_revision=rev,
            retry_count=0,
            request_id=configurable.get("request_id", "") or "",
        )
        signal = ResumeSignal(
            confirmation_id=f"{_session_id}:{tool_call_id}",
            action=action,
            grant_scope=scope,
            actor="user_click",
        )

        try:
            outcome = await pe.commit_resume(
                call_spec, ctx, signal, claim_nonce=claim_nonce,
            )
        except (PolicyConflict, WriterIntegrityError, SessionModeViolation) as exc:
            logger.warning(
                "interrupt_helper: pe.commit_resume raised %s for tool_call %s: %s",
                type(exc).__name__, tool_call_id, exc,
            )
            # P1 (round-25): claim_nonce_mismatch means the current call does NOT
            # own the queue entry — another resume (new owner) has already claimed
            # it.  Calling cleanup_pending_confirmation here would delete the new
            # owner's state and corrupt their in-flight confirmation.  Skip cleanup
            # and let the new owner's commit_resume handle it.
            #
            # For all other PolicyConflict / WriterIntegrityError /
            # SessionModeViolation variants, the queue entry either belongs to us
            # (commit_resume may have exited before its own cleanup) or is safe to
            # purge because the session is in a terminal state.  Cleanup is
            # idempotent so re-calling it after commit_resume's own cleanup is safe.
            if isinstance(exc, PolicyConflict) and "claim_nonce_mismatch" in str(exc):
                logger.warning(
                    "interrupt_helper: nonce mismatch for %s:%s — leaving queue "
                    "entry for new owner, no cleanup performed",
                    _session_id, tool_call_id,
                )
                # TODO (codex round-35 P1, BLOCKED — accepted race window):
                # Returning _build_resume_error_command below clears pending_ask_*
                # AND adds tool_call_id to completed_tool_call_prefix → the graph
                # advances past interrupt_helper.  Meanwhile, the new owner (B)
                # already called task.resume(Command_B) which is sitting in the
                # checkpointer waiting to be consumed by interrupt_helper.  But the
                # graph has already advanced past interrupt → Command_B is consumed
                # as a no-op → B's queue entry stays in 'processing' until B's
                # deadline_ts expires and the sweeper find_expired() cleans it up.
                #
                # Fix Option A (re-raise GraphInterrupt) requires deep manipulation
                # of LangGraph internals: scratchpad.resume is already populated
                # with A's resume value, the runner re-persists RESUME writes at
                # _runner.py:440-441, and on the next /resume call the OLD A value
                # would be returned by interrupt() again instead of waiting for B.
                # There is no documented LangGraph primitive for "consume the
                # current resume but pause for a new one".
                #
                # Fix Option C (sweeper detects graph already advanced past
                # interrupt while queue entry processing → cleanup) requires
                # reading graph state from the sweeper — feasible but adds another
                # round-trip per sweep cycle and is non-trivial.
                #
                # Current behavior: B's resume is lost; B's entry stays processing
                # until deadline_ts expiry → sweeper find_expired() → submit
                # timeout_fallback (which is a no-op since graph already advanced)
                # → cleanup() in the resume_ok branch of sweep Phase 1.  B's user
                # sees a confirmation timeout instead of the action they actually
                # requested.  This is the accepted trade-off until Option A or C
                # is properly designed.
            else:
                try:
                    await pe.cleanup_pending_confirmation(_session_id, tool_call_id)
                except Exception:
                    logger.exception(
                        "interrupt_helper: cleanup_pending_confirmation failed on "
                        "commit_resume error path for %s:%s", _session_id, tool_call_id,
                    )
            return _build_resume_error_command(state, exc)
        except Exception as exc:
            # Infrastructure error (Redis/DB/SSM connection issue) — convert to
            # an error ToolMessage so the caller sees the failure via SSE rather
            # than the confirmation hanging in 'processing' forever.  The
            # pending_ask_* fields are cleared by _build_resume_error_command so
            # the next interrupt_helper replay does not re-enter the PE path with
            # stale data.
            # P2 (round-18): Cleanup the queue entry to prevent stale 'processing'
            # state in case commit_resume raised before reaching its own cleanup.
            logger.exception(
                "interrupt_helper: pe.commit_resume failed with infrastructure error "
                "for tool_call %s — returning error Command to prevent hang: %s",
                tool_call_id,
                exc,
            )
            try:
                await pe.cleanup_pending_confirmation(_session_id, tool_call_id)
            except Exception:
                logger.exception(
                    "interrupt_helper: cleanup_pending_confirmation failed on "
                    "generic exception path for %s:%s", _session_id, tool_call_id,
                )
            return _build_resume_error_command(state, exc)

        update: dict[str, Any] = {
            "pe_resume_outcomes": {
                **(state.get("pe_resume_outcomes") or {}),
                tool_call_id: outcome.model_dump(mode="json"),
            },
            # Clear pending_ask fields so the next interrupt_helper replay
            # doesn't re-enter the PE path with stale data.
            "pending_ask_outcome": None,
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None,
            "pending_ask_tool_args": None,
        }
        logger.info(
            "interrupt_helper (PE): commit_resume done for tool_call %s action=%s scope=%s",
            tool_call_id, action, scope,
        )
        return Command(goto="tool_node", update=update)

    # ---- Build Graph --------------------------------------------------- #

    g: StateGraph = StateGraph(ReactGraphState)

    # RetryPolicy for transient LLM errors (ServerRequestsError → RuntimeError)
    llm_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=2.0,
        backoff_factor=2.0,
        retry_on=ServerRequestsError,
    )

    g.add_node("pre_llm_node", pre_llm_node)
    g.add_node("llm_node", llm_node, retry_policy=llm_retry)
    g.add_node("tool_node", tool_node)
    # R2 CS2: interrupt_helper is the only node that calls interrupt().
    # Registered so tool_node's Command(goto="interrupt_helper") resolves.
    g.add_node("interrupt_helper", interrupt_helper)

    g.add_edge(START, "pre_llm_node")
    g.add_edge("pre_llm_node", "llm_node")
    g.add_conditional_edges("llm_node", route_after_llm)
    # tool_node routes itself via Command(goto=...); interrupt_helper too.
    # No conditional edge needed — replaces the former route_after_tool.

    return g.compile(checkpointer=checkpointer)
