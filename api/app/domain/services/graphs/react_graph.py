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
from app.domain.services.risk_assessor import RiskAssessor, RiskLevel
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
# Layer 1 dependency slots are Commit 1 stubs. Task 13 (tool_node dispatcher
# rewrite) wires the real services from ``configurable`` (approval_cache,
# summary_llm) into the policy chain. Task 44 replaces the timeout constants
# with injected ``ToolRuntimeConfig`` values.

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


def _smart_approve_applies(tool_source: ToolSource, tool_call: ToolCall) -> bool:
    """Whether Stage P.2 (SmartApprove) should run for this tool.

    Commit 1 stub always returns True so the happy-path test exercises
    the full S → P.1 → P.2 → None chain. Task 13 replaces this with the
    real risk-level check (mirrors the existing ``risk_level`` metadata
    gate at react_graph.py:443-448).
    """
    return True


async def _stage_s_ast_validate(
    tool_call: ToolCall,
) -> ToolOutcome | None:
    """Stage S: shell AST validator (N1).

    Calls into the pure-sync ``shell_ast_validator.validate()``.
    Returns ``Denied(reason.type="ast_validator")`` when validate() rejects,
    ``None`` when allowed.
    """
    from app.domain.services.safety.shell_ast_validator import (
        to_typed_denied,
        validate,
    )
    from core.config import get_settings

    args = tool_call.get("args", {}) or {}
    command = args.get("command", "") or ""
    exec_dir = args.get("exec_dir", "") or get_settings().sandbox_default_cwd
    result = validate(command, effective_cwd=exec_dir)
    if result.allowed:
        return None
    return to_typed_denied(result, original_command=command)


async def _stage_p1_approval_cache_check(
    session_ctx: _SessionContext,
    tool_call: ToolCall,
) -> ToolOutcome | Literal["policy_allow"] | None:
    """Stage P.1: ApprovalCache (Redis).

    Commit 1 stub returns None (no cached decision). Task 13 replaces
    with ``configurable['approval_cache'].check(...)``.

    Returns:
      - ``Denied | Asked`` — user has already decided, short-circuit
      - ``"policy_allow"`` — user pre-allowed, skip SmartApprove
      - ``None`` — no cached decision, continue chain
    """
    return None


async def _stage_p2_smart_approve(
    tool_call: ToolCall,
    session_ctx: _SessionContext,
) -> ToolOutcome | None:
    """Stage P.2: SmartApprove LLM evaluation.

    Commit 1 stub returns None. Task 13 replaces with
    ``SmartApprove(llm=configurable['summary_llm']).evaluate(...)``.
    """
    return None


async def _run_policy_chain(
    tool_call: ToolCall,
    tool: BaseTool,
    tool_source: ToolSource,
    session_ctx: _SessionContext,
    *,
    smart_approve_timeout_seconds: int | None = None,
) -> ToolOutcome | None:
    """Stage-based Layer 1 policy evaluation.

    Returns None iff all stages allow (wrapper should run). Otherwise
    returns a ``Denied`` / ``Asked`` / ``AllowError`` that short-circuits
    Layer 2.

    Stage-level exception handling:
    - **Stage S crash → AllowError (fail-closed)**. Shell AST is a safety
      gate; crashing it must NOT fall through to wrapper execution.
    - **Stage P.1 crash (e.g., Redis down) → fail-open**. Redis outages
      shouldn't block every tool call. Log + continue.
    - **Stage P.2 timeout/crash → fail-open**. SmartApprove is optional
      reinforcement; its failure mustn't hold up the happy path.

    Layer 1 only produces ``Denied/Asked/AllowError``; ``AllowSuccess``
    and ``Passthrough`` come from Layer 2 (wrapper).

    ``smart_approve_timeout_seconds`` comes from
    ``AppConfig.tool_runtime.smart_approve_timeout_seconds`` via
    ``build_react_graph(tool_runtime_config=...)``. ``None`` falls back to
    the module constant ``_SMART_APPROVE_TIMEOUT_SECONDS`` so legacy
    callers and existing tests keep working without plumbing the config
    through.
    """
    p2_timeout = (
        smart_approve_timeout_seconds
        if smart_approve_timeout_seconds is not None
        else _SMART_APPROVE_TIMEOUT_SECONDS
    )
    # Stage S: Safety (shell AST validator, native shell only)
    if _is_shell_category(tool_source):
        try:
            ast_outcome = await _stage_s_ast_validate(tool_call)
        except Exception as exc:
            logger.exception("Stage S AST validator crashed for %s", tool_call["name"])
            return AllowError(
                content=f"AST validator 内部异常: {exc}",
                reason=DecisionReason(
                    type="exception",
                    code="layer1_ast_crash",
                    message=str(exc),
                ),
                retryable=False,
            )
        if ast_outcome is not None:
            return ast_outcome

    # Stage P.1: ApprovalCache (Redis)
    try:
        cached = await _stage_p1_approval_cache_check(session_ctx, tool_call)
    except Exception:
        logger.exception(
            "Stage P.1 ApprovalCache crashed for %s (fail-open)", tool_call["name"]
        )
        cached = None
    if isinstance(cached, (Denied, Asked)):
        return cached
    if cached == "policy_allow":
        return None  # User pre-allowed, skip Stage P.2

    # Stage P.2: SmartApprove (LLM)
    if _smart_approve_applies(tool_source, tool_call):
        try:
            smart = await asyncio.wait_for(
                _stage_p2_smart_approve(tool_call, session_ctx),
                timeout=p2_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Stage P.2 SmartApprove timeout for %s (fail-open)", tool_call["name"]
            )
            smart = None
        except Exception:
            logger.exception(
                "Stage P.2 SmartApprove crashed for %s (fail-open)", tool_call["name"]
            )
            smart = None
        if isinstance(smart, (Denied, Asked)):
            return smart

    return None


GuideInjector = Callable[[str], str | None]


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

    legacy_response_format = (
        getattr(tool, "response_format", "content") != "content_and_artifact"
    )

    # Legacy branch: tool predates the CS2 wrapper migration. Call
    # ``ainvoke(args)`` to receive the raw Python object so we can
    # special-case ``FileProcessResult``, preserving the pre-R2 file_view
    # multimodal path under the new dispatcher.
    if legacy_response_format:
        try:
            raw = await tool.ainvoke(tool_call["args"])
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
            }
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
        Layer 1 ``_run_policy_chain`` and Layer 2 ``_invoke_wrapper``.
        Defaults to ``ToolRuntimeConfig()`` (1 MiB wrapper cap, 15s
        SmartApprove timeout). The values are captured in the closure
        so every future call site that wires ``_run_policy_chain`` /
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

    async def pre_llm_node(state: ReactGraphState) -> dict:
        """Trim messages for LLM input. state['messages'] is unchanged."""
        if assembler is None:
            return {"llm_input_messages": list(state["messages"])}
        result = assembler.assemble(list(state["messages"]))
        if result.actions:
            logger.info("context_assembler(in-step): %s", result.actions)
        return {"llm_input_messages": result.messages}

    async def llm_node(state: ReactGraphState, config: RunnableConfig) -> dict:
        """Call the LLM with current messages."""
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

        return {
            "messages": [response],
            "events": new_events,
        }

    # Shared risk assessor instance (stateless, safe to reuse)
    _risk_assessor = RiskAssessor()

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
        the approve-resume bridge; ``ApprovalCache`` write is post-resume in
        ``agent_service._resume_tool_confirmation``, so the dispatcher cannot
        rely on the cache alone for the replay and needs this state flag.

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
        exclusively to ``interrupt_helper`` (CS2.13 invariant). Commit 2a
        (Task 21) will replace this gate with the full Layer 1/2/3 pipeline
        via ``_run_policy_chain`` / ``_invoke_wrapper`` / ``_translate_outcome``.

        ## Return type

        Always returns ``Command`` — either ``Command(goto="pre_llm_node",
        update=...)`` on the happy path or ``Command(goto="interrupt_helper",
        update=...)`` when a tool call reaches an Asked outcome. This
        replaces the dict return + ``route_after_tool`` conditional edge.
        """
        import time as _time
        configurable = (config or {}).get("configurable", {}) if config else {}
        guide_injector = configurable.get("skill_guide_injector")
        event_queue = configurable.get("event_queue")
        confirmation_manager = configurable.get("confirmation_manager")
        _tracker = configurable.get("tool_failure_tracker")
        _metrics = configurable.get("execution_metrics")

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

                if not ast_result.allowed:
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

            # ---- Legacy risk gate (real ApprovalCache / SmartApprove) ----
            #
            # The Layer 1 ``_run_policy_chain`` helper exists as a typed
            # facade but its internal stage stubs are not yet wired to
            # real services (``_stage_s_ast_validate`` awaits N1,
            # ``_stage_p1_approval_cache_check`` / ``_stage_p2_smart_approve``
            # would need ``configurable`` threading). Rather than gate
            # production on half-wired stubs, R2 keeps the legacy gate
            # as the policy layer and only swaps the **execution + output
            # construction** onto Layer 2 (``_invoke_wrapper``) + Layer 3
            # (``_translate_outcome``). Result: every path that used to
            # call ``_run_tool`` + build a raw ToolMessage now produces
            # a typed ``ToolOutcome`` whose artifact flows through to
            # Chunk 4's LLM adapter prefix injection.
            risk_level_meta = (getattr(tool_fn, "metadata", None) or {}).get("risk_level")
            _tc_enabled = configurable.get("tool_confirmation_enabled", True)
            _runtime_max_bytes = _tool_runtime_cfg.max_wrapper_output_bytes

            # R3: Pre-Stage-P risk refresh for skill tools.
            # If the bundle changed since init, rescan and use the fresh risk level
            # so approval decisions are never based on stale metadata.
            if (
                tool_source
                and tool_source.source == "skill"
                and _tc_enabled
            ):
                _r3_skill_tool = configurable.get("skill_tool")
                if _r3_skill_tool:
                    _refreshed_risk = _r3_skill_tool.refresh_risk_if_stale(tool_name)
                    if _refreshed_risk:
                        risk_level_meta = _refreshed_risk
                        # Also update tool_fn.metadata so the rest of the pipeline sees it
                        _meta = getattr(tool_fn, "metadata", None) or {}
                        _meta["risk_level"] = _refreshed_risk
                        tool_fn.metadata = _meta

            # ========== R3: Skill Stage P branch ==========
            if (
                not _bypass_risk_gate
                and _tc_enabled
                and tool_source
                and tool_source.source == "skill"
                and risk_level_meta
                and risk_level_meta in ("high", "medium")
            ):
                from app.domain.services.skill_risk_assessor import SkillRiskAssessor
                from app.domain.models.skill import SkillRuntimeType

                _skill_meta = getattr(tool_fn, "metadata", None) or {}
                _skill_risk_level = RiskLevel[risk_level_meta.upper()]
                _skill_assessor = SkillRiskAssessor()
                assessment = _skill_assessor.assess(
                    tool_name=tool_name,
                    tool_args=args,
                    risk_level=_skill_risk_level,
                    runtime_type=SkillRuntimeType(
                        _skill_meta.get("runtime_type", "native")
                    ),
                    trust_origin=_skill_meta.get("trust_origin", "user_installed"),
                )

                # P.1: ApprovalStateReader (R5b-2: 取代 ApprovalCache 读路径)
                cache_decision = "no_match"
                approval_state_reader = configurable.get("approval_state_reader")
                _user_id = configurable.get("user_id") or ""
                _session_id = configurable.get("session_id") or ""
                if approval_state_reader and _user_id and _session_id:
                    try:
                        cache_decision = await approval_state_reader.check(
                            user_id=_user_id,
                            session_id=_session_id,
                            tool_name=tool_name,
                            arg_digest=assessment.arg_digest,
                            primary_arg=assessment.primary_arg,
                            dir_arg=assessment.dir_arg,
                        )
                    except Exception:
                        logger.warning(
                            "Stage P.1 ApprovalStateReader crashed for skill %s (fail-open)",
                            tool_name,
                        )
                        cache_decision = "no_match"

                if cache_decision == "allow":
                    pass  # fall through to execution

                elif cache_decision == "deny":
                    _denied = Denied(
                        content=f"Skill '{tool_name}' 被审批策略拒绝",
                        reason=DecisionReason(
                            type="approval_policy",
                            code="cache_deny",
                            message=f"risk_level={risk_level_meta}",
                        ),
                    )
                    await _finalize_outcome(tc, args, tool_source, _denied, _tool_start)
                    continue

                else:
                    # P.3: risk_enforce backstop (>= MEDIUM → user confirmation)
                    _timeout_seconds = configurable.get(
                        "tool_confirmation_timeout_seconds", 300
                    )
                    confirmation_event = ToolConfirmationEvent(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        tool_args=args,
                        risk_level=assessment.final_level.name.lower(),
                        risk_reason=assessment.risk_reason,
                        matched_patterns=assessment.matched_patterns,
                        suggested_alternative=assessment.suggested_alternative,
                        timeout_seconds=_timeout_seconds,
                    )
                    if event_queue:
                        await event_queue.put(confirmation_event)

                    if confirmation_manager:
                        from app.domain.services.confirmation_manager import (
                            ConfirmationDetail,
                        )
                        _detail = ConfirmationDetail(
                            session_id=_session_id,
                            tool_call_id=call_id,
                            user_id=_user_id,
                            tool_name=tool_name,
                            tool_args=args,
                            risk_level=assessment.final_level.name.lower(),
                            arg_digest=assessment.arg_digest,
                            primary_arg=assessment.primary_arg,
                            dir_arg=assessment.dir_arg,
                            matched_patterns=assessment.matched_patterns,
                            deadline_ts=_time.time() + confirmation_event.timeout_seconds,
                        )
                        await confirmation_manager.store(_detail)

                    _pending_outcome = Asked(
                        content="等待用户确认 Skill 工具执行",
                        reason=DecisionReason(
                            type="risk_enforce",
                            code=assessment.final_level.name.lower(),
                            message=assessment.risk_reason or "",
                        ),
                    )
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
                    return Command(goto="interrupt_helper", update=_update)

            # ========== End R3 Skill Stage P ==========
            # Original native tool gate — skip for skill tools (handled above)
            if (
                not _bypass_risk_gate
                and _tc_enabled
                and risk_level_meta
                and risk_level_meta in ("high", "medium")
                and (not tool_source or tool_source.source != "skill")
            ):
                assessment = _risk_assessor.assess(tool_name, args)

                if assessment.final_level >= RiskLevel.MEDIUM:
                    # R5b-2 Reader 接入 + R5b-4 cleanup：ApprovalCache 已移除，
                    # SmartApprove 写路径走 configurable["approval_state_writer"]（R5b-3）
                    approval_state_reader = configurable.get("approval_state_reader")
                    _user_id = configurable.get("user_id") or ""
                    _session_id = configurable.get("session_id") or ""

                    cache_decision = "no_match"
                    if approval_state_reader and _user_id and _session_id:
                        try:
                            cache_decision = await approval_state_reader.check(
                                user_id=_user_id,
                                session_id=_session_id,
                                tool_name=tool_name,
                                arg_digest=assessment.arg_digest,
                                primary_arg=assessment.primary_arg,
                                dir_arg=assessment.dir_arg,
                            )
                        except Exception:
                            logger.warning(
                                "ApprovalStateReader.check failed for tool '%s', "
                                "defaulting to no_match",
                                tool_name,
                            )
                            cache_decision = "no_match"

                    if cache_decision == "allow":
                        outcome = await _invoke_wrapper(
                            tool_fn,
                            tc,
                            tool_source,
                            max_wrapper_output_bytes=_runtime_max_bytes,
                        )
                        outcome = _maybe_convert_shell_outcome_with_images(
                            outcome, tool_name
                        )
                        await _finalize_outcome(
                            tc, args, tool_source, outcome, _tool_start
                        )
                        continue

                    if cache_decision == "deny":
                        denied_outcome = Denied(
                            content="此操作已被永久规则拒绝",
                            reason=DecisionReason(
                                type="approval_policy",
                                code="cache_always_deny",
                                message=(
                                    "User-configured always-deny rule matched "
                                    "for this tool+arg pattern"
                                ),
                            ),
                        )
                        await _finalize_outcome(
                            tc, args, tool_source, denied_outcome, _tool_start
                        )
                        continue

                    # cache miss — try SmartApprove before interrupting
                    _sa_resolved = False
                    _smart_approve_enabled = configurable.get(
                        "smart_approve_enabled", False
                    )
                    _sa_medium_only = configurable.get(
                        "smart_approve_medium_only", False
                    )
                    if _smart_approve_enabled and not (
                        _sa_medium_only
                        and assessment.final_level > RiskLevel.MEDIUM
                    ):
                        from app.domain.services.smart_approve import (
                            SmartApprove,
                        )

                        _summary_llm = configurable.get("summary_llm")
                        if _summary_llm:
                            # B5 PR-S3-2 reviewer round-2 P3: thread
                            # the decision recorder from configurable
                            # so SmartApprove (domain) doesn't import
                            # the OTel helper directly. Composition
                            # (planner_react._build_config) injects
                            # the OTel-backed callable; tests pass
                            # ``None`` and SmartApprove silently
                            # skips the emit.
                            _decision_recorder = configurable.get(
                                "decision_recorder"
                            )
                            _smart = SmartApprove(
                                llm=_summary_llm,
                                decision_recorder=_decision_recorder,
                            )
                            try:
                                _sa_decision = await asyncio.wait_for(
                                    _smart.evaluate(
                                        tool_name=tool_name,
                                        tool_args=args,
                                        risk_level=assessment.final_level.name.lower(),
                                        matched_patterns=assessment.matched_patterns,
                                        task_context="",
                                    ),
                                    timeout=(
                                        _tool_runtime_cfg.smart_approve_timeout_seconds
                                    ),
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    "SmartApprove timeout (%.1fs) for tool '%s' — "
                                    "falling through to user-confirmation interrupt",
                                    _tool_runtime_cfg.smart_approve_timeout_seconds,
                                    tool_name,
                                )
                                _sa_decision = None
                            if _sa_decision == "approve":
                                logger.info(
                                    "SmartApprove: auto-approved tool '%s', "
                                    "granting session scope",
                                    tool_name,
                                )
                                # R5b-3: SmartApprove 写路径从 ApprovalCache.write_session
                                # 切到 ApprovalStateWriter（CS4 单一 Writer）。confirmation_id=None
                                # 命中 partial UNIQUE (user, session, tool, arg_digest, effect)
                                # WHERE confirmation_id IS NULL 做 SmartApprove 去重。
                                _asw = configurable.get("approval_state_writer")
                                if _asw and _session_id and _user_id:
                                    try:
                                        from app.domain.models.approval_grant import (
                                            ApprovalDecision,
                                        )
                                        from app.domain.services.approval_grant_policy import (
                                            session_grant_expires_at,
                                        )
                                        _sa_tool_source = (
                                            tool_source.source
                                            if tool_source is not None
                                            else "native"
                                        )
                                        _sa_decision_obj = ApprovalDecision(
                                            user_id=_user_id,
                                            session_id=_session_id,
                                            tool_name=tool_name,
                                            tool_source=_sa_tool_source,
                                            arg_digest=assessment.arg_digest,
                                            primary_arg=assessment.primary_arg,
                                            dir_arg=assessment.dir_arg or "",
                                            scope="session",
                                            effect="approve",
                                            source_type="smart_approve",
                                            confirmation_id=None,
                                            expires_at=session_grant_expires_at(),
                                            risk_level=assessment.final_level.name.lower(),
                                        )
                                        await _asw.write(_sa_decision_obj)
                                    except Exception as _sa_write_err:
                                        logger.warning(
                                            "SmartApprove grant 写入失败 tool=%s: %s "
                                            "(fail-open: 本次执行不受影响，下次同 arg 仍会走 SmartApprove)",
                                            tool_name, _sa_write_err,
                                        )
                                outcome = await _invoke_wrapper(
                                    tool_fn,
                                    tc,
                                    tool_source,
                                    max_wrapper_output_bytes=_runtime_max_bytes,
                                )
                                outcome = _maybe_convert_shell_outcome_with_images(
                                    outcome, tool_name
                                )
                                await _finalize_outcome(
                                    tc, args, tool_source, outcome, _tool_start
                                )
                                _sa_resolved = True
                            elif _sa_decision == "deny":
                                logger.info(
                                    "SmartApprove: auto-denied tool '%s'",
                                    tool_name,
                                )
                                denied_outcome = Denied(
                                    content="此操作已被自动安全策略拒绝",
                                    reason=DecisionReason(
                                        type="smart_approve",
                                        code="smart_approve_deny",
                                        message=(
                                            "SmartApprove LLM evaluation rejected "
                                            "this tool call"
                                        ),
                                    ),
                                )
                                await _finalize_outcome(
                                    tc,
                                    args,
                                    tool_source,
                                    denied_outcome,
                                    _tool_start,
                                )
                                _sa_resolved = True
                            # _sa_decision == "escalate" → fall through to
                            # interrupt_helper routing below

                    if _sa_resolved:
                        continue

                    # Escalate to user: route to interrupt_helper with
                    # typed pending_ask_* state.
                    _timeout_seconds = configurable.get(
                        "tool_confirmation_timeout_seconds", 300
                    )
                    confirmation_event = ToolConfirmationEvent(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        tool_args=args,
                        risk_level=assessment.final_level.name.lower(),
                        risk_reason=assessment.risk_reason,
                        matched_patterns=assessment.matched_patterns,
                        suggested_alternative=assessment.suggested_alternative,
                        timeout_seconds=_timeout_seconds,
                    )
                    if event_queue:
                        await event_queue.put(confirmation_event)

                    if confirmation_manager:
                        from app.domain.services.confirmation_manager import (
                            ConfirmationDetail,
                        )

                        _detail = ConfirmationDetail(
                            session_id=_session_id,
                            tool_call_id=call_id,
                            user_id=_user_id,
                            tool_name=tool_name,
                            tool_args=args,
                            risk_level=assessment.final_level.name.lower(),
                            arg_digest=assessment.arg_digest,
                            primary_arg=assessment.primary_arg,
                            dir_arg=assessment.dir_arg,
                            matched_patterns=assessment.matched_patterns,
                            deadline_ts=_time.time()
                            + confirmation_event.timeout_seconds,
                        )
                        await confirmation_manager.store(_detail)

                    _pending_outcome = Asked(
                        content="等待用户确认工具执行",
                        reason=DecisionReason(
                            type="risk_enforce",
                            code=assessment.final_level.name.lower(),
                            message=assessment.risk_reason or "",
                        ),
                    )
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
                        "pending_ask_outcome": _pending_outcome.model_dump(
                            mode="json"
                        ),
                        "pending_ask_tool_call_id": call_id,
                        "pending_ask_artifact": _pending_artifact.model_dump(
                            mode="json", by_alias=True
                        ),
                        "pending_ask_tool_args": dict(args),
                    }
                    return Command(
                        goto="interrupt_helper",
                        update=_update,
                    )

                # assessment < MEDIUM → execute directly
                outcome = await _invoke_wrapper(
                    tool_fn,
                    tc,
                    tool_source,
                    max_wrapper_output_bytes=_runtime_max_bytes,
                )
                outcome = _maybe_convert_shell_outcome_with_images(
                    outcome, tool_name
                )
                await _finalize_outcome(
                    tc, args, tool_source, outcome, _tool_start
                )
                continue

            # No risk metadata (or bypass via pre-approved) → execute directly
            outcome = await _invoke_wrapper(
                tool_fn,
                tc,
                tool_source,
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

    async def interrupt_helper(
        state: ReactGraphState, config: RunnableConfig
    ) -> Command[Literal["tool_node"]]:
        """R2 CS2 — the only node allowed to call ``interrupt()``.

        Receives control from ``tool_node`` whenever a tool_call reaches an
        ``Asked`` outcome (Layer 1 policy chain or Layer 2 wrapper). Reads
        the ``pending_ask_*`` state written by ``tool_node``, calls
        ``interrupt(...)`` to pause the graph, and on resume dispatches:

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
        ``_invoke_wrapper`` / ``_run_policy_chain``, and must NOT execute
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

        action = (
            user_response.get("action", "deny")
            if isinstance(user_response, dict)
            else "deny"
        )

        if action == "approve":
            logger.info(
                "interrupt_helper: user approved tool_call %s", pending_id
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
            "interrupt_helper: user %s tool_call %s (tool=%s)",
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
        configurable = (config or {}).get("configurable", {}) if config else {}
        tool_result_max_chars = configurable.get("tool_result_max_chars", 8000)
        guide_injector = configurable.get("skill_guide_injector")

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
