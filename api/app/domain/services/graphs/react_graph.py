"""react_graph — inner ReAct loop as a LangGraph StateGraph.

Replaces BaseAgent.invoke() and ReActAgent.execute_step().
Nodes: pre_llm_node, llm_node, tool_node
Edges: START → pre_llm_node → llm_node → route_after_llm → (tool_node → pre_llm_node) | END

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.3-4.4
"""

from __future__ import annotations

import base64 as _b64
import json
import logging
from typing import Any, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, interrupt

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.file_processor import FileProcessResult
from app.domain.models.app_config import AgentConfig
from app.domain.models.event import (
    MessageEvent,
    ToolConfirmationEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import ToolResult
from app.domain.services.json_envelope import unwrap_message_envelope
from app.domain.services.risk_assessor import RiskAssessor, RiskLevel
from app.domain.services.tools.tool_source_resolver import resolve_tool_source

from .message_utils import truncate_tool_content
from .state import ReactGraphState

if TYPE_CHECKING:
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


def build_react_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    agent_config: AgentConfig | None = None,
    tool_result_max_chars: int = 8000,
    assembler: ContextAssembler | None = None,
) -> CompiledStateGraph:
    """Build and compile the inner ReAct loop graph.

    Parameters
    ----------
    llm : LangChain BaseChatModel — must support bind_tools.
    tools : List of LangChain tools.
    agent_config : Optional AgentConfig for iteration limits etc.
    """
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

        # Emit ToolEvent(CALLING) for each tool call
        if response.tool_calls:
            for tc in response.tool_calls:
                func_name = tc["name"]
                new_events.append(
                    ToolEvent(
                        tool_call_id=tc["id"],
                        tool_name=resolve_tool_source(func_name).category,
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

    async def tool_node(state: ReactGraphState, config: RunnableConfig) -> dict:
        """Execute tool calls from the last assistant message.

        Special handling for ``message_ask_user``:
        - If ``suggest_user_takeover`` is "browser"/"shell" → set should_interrupt
          (handled by confirmation_check, but also guard here).
        - Otherwise, first call returns SOFT_HINT (agent should try to solve
          autonomously). If a SOFT_HINT was already returned in this step
          and the LLM calls again, it truly needs user input → interrupt.

        Risk assessment gate for tools with ``risk_level`` metadata (high/medium):
        - Runs RiskAssessor to evaluate dynamic risk.
        - If final_level >= MEDIUM, emits ToolConfirmationEvent and calls
          ``interrupt()`` to pause the graph until the user responds.
        - Approved calls proceed to normal execution; denied/timed-out calls
          return an error string without executing.
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
            return {
                "should_interrupt": True,
                "messages": [],
                "events": [],
            }

        messages = state["messages"]
        last_msg = messages[-1]

        # AIMessage.tool_calls is a list of dicts with id/name/args
        tool_calls = last_msg.tool_calls if isinstance(last_msg, AIMessage) else []

        # Check if a SOFT_HINT was already returned in this step
        has_prior_soft_hint = state.get("soft_hint_sent", False)

        new_messages = []
        new_events = []
        should_interrupt = False
        new_failures = 0
        # Collect multimodal HumanMessages from file_view results.
        # Appended AFTER all ToolMessages to preserve AIMessage → ToolMessage*
        # pairing for group_messages() (context_assembler.py).
        deferred_human_messages: list[HumanMessage] = []
        deferred_document_messages: list[HumanMessage] = []

        async def _run_tool(
            tool_fn: BaseTool,
            tool_name: str,
            args: dict,
        ) -> tuple[str, bool, list[dict]]:
            """Execute a tool and handle result type coercion.

            Returns (result_str, success, multimodal_blocks).
            Side effect: may append to ``deferred_document_messages``.
            """
            mm_blocks: list[dict] = []
            try:
                raw_result = await tool_fn.ainvoke(args)
                if isinstance(raw_result, FileProcessResult):
                    r_str = raw_result.text
                    mm_blocks = list(raw_result.image_blocks)
                    if raw_result.document_blocks:
                        doc_blocks: list[dict] = list(raw_result.document_blocks)
                        doc_blocks.insert(0, {"type": "text", "text": "[file_view: PDF document attached]"})
                        deferred_document_messages.append(HumanMessage(content=doc_blocks))
                elif isinstance(raw_result, str):
                    r_str = raw_result
                else:
                    r_str = str(raw_result)
                # Shell image detection (M1d)
                if tool_name in ("shell_execute", "shell_read_output"):
                    r_str, shell_images = _extract_shell_images(r_str)
                    mm_blocks.extend(shell_images)
                return r_str, True, mm_blocks
            except Exception as exc:
                return f"Error executing {tool_name}: {exc}", False, []

        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            call_id = tc["id"]
            _tool_start = _time.monotonic()

            # ---- message_ask_user: SOFT_HINT gating ---- #
            tool_success = True
            multimodal_blocks: list[dict] = []
            if tool_name == "message_ask_user":
                suggest = str(args.get("suggest_user_takeover", "none")).strip().lower()
                if suggest in {"browser", "shell"}:
                    # Takeover request → always interrupt
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                elif not has_prior_soft_hint:
                    # First non-takeover ask → return SOFT_HINT
                    result_str = "SOFT_HINT"
                    logger.info("message_ask_user: returning SOFT_HINT (first attempt)")
                else:
                    # Second call after SOFT_HINT → truly needs user input
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                    logger.info("message_ask_user: user input required (after SOFT_HINT)")
            else:
                # ---- Normal tool execution (with risk assessment gate) ---- #
                # D5: Check if this tool+args signature is blocked by tracker
                if _tracker and _tracker.is_blocked(tool_name, args):
                    result_str = f"[BLOCKED] 此工具调用模式（{tool_name}）因连续失败已被暂停，请尝试不同的工具或参数"
                    tool_success = False
                    new_messages.append(ToolMessage(content=f"[TOOL_ERROR] {result_str}", tool_call_id=call_id, name=tool_name))
                    new_events.append(ToolEvent(
                        tool_call_id=call_id, tool_name=resolve_tool_source(tool_name).category,
                        function_name=tool_name, function_args=args,
                        function_result=ToolResult(success=False, message=result_str),
                        status=ToolEventStatus.CALLED,
                    ))
                    new_failures += 1
                    if _metrics:
                        _metrics.record_tool_call(
                            success=False,
                            latency_ms=(_time.monotonic() - _tool_start) * 1000,
                        )
                    continue

                tool_fn = tool_map.get(tool_name)
                if tool_fn is None:
                    result_str = f"Error: Unknown tool '{tool_name}'"
                    tool_success = False
                else:
                    # Check tool risk_level metadata for confirmation gating
                    risk_level_meta = (getattr(tool_fn, "metadata", None) or {}).get("risk_level")
                    _tc_enabled = configurable.get("tool_confirmation_enabled", True)
                    if _tc_enabled and risk_level_meta and risk_level_meta in ("high", "medium"):
                        assessment = _risk_assessor.assess(tool_name, args)

                        if assessment.final_level >= RiskLevel.MEDIUM:
                            # Check ApprovalCache (session + always-rules)
                            approval_cache = configurable.get("approval_cache")
                            _user_id = configurable.get("user_id") or ""
                            _session_id = configurable.get("session_id") or ""

                            if approval_cache and _user_id and _session_id:
                                try:
                                    cache_decision = await approval_cache.check(
                                        user_id=_user_id,
                                        session_id=_session_id,
                                        tool_name=tool_name,
                                        arg_digest=assessment.arg_digest,
                                        primary_arg=assessment.primary_arg,
                                        dir_arg=assessment.dir_arg,
                                    )
                                except Exception:
                                    logger.warning(
                                        "ApprovalCache.check failed for tool '%s', defaulting to no_match",
                                        tool_name,
                                    )
                                    cache_decision = "no_match"
                            else:
                                cache_decision = "no_match"

                            if cache_decision == "allow":
                                # Cached approval — execute directly
                                result_str, tool_success, multimodal_blocks = await _run_tool(tool_fn, tool_name, args)
                            elif cache_decision == "deny":
                                result_str = "此操作已被永久规则拒绝"
                                tool_success = False
                            else:
                                # SmartApprove: LLM-assisted auto-approval before interrupting
                                _sa_resolved = False
                                _smart_approve_enabled = configurable.get("smart_approve_enabled", False)
                                _sa_medium_only = configurable.get("smart_approve_medium_only", False)
                                # Skip SmartApprove if medium_only is set and tool is HIGH
                                if _smart_approve_enabled and not (_sa_medium_only and assessment.final_level > RiskLevel.MEDIUM):
                                    from app.domain.services.smart_approve import SmartApprove
                                    _summary_llm = configurable.get("summary_llm")
                                    if _summary_llm:
                                        _smart = SmartApprove(llm=_summary_llm)
                                        _sa_decision = await _smart.evaluate(
                                            tool_name=tool_name,
                                            tool_args=args,
                                            risk_level=assessment.final_level.name.lower(),
                                            matched_patterns=assessment.matched_patterns,
                                            task_context="",
                                        )
                                        if _sa_decision == "approve":
                                            logger.info(
                                                "SmartApprove: auto-approved tool '%s', granting session scope",
                                                tool_name,
                                            )
                                            if approval_cache and _session_id:
                                                await approval_cache.write_session(
                                                    _session_id, tool_name, assessment.arg_digest
                                                )
                                            result_str, tool_success, multimodal_blocks = await _run_tool(
                                                tool_fn, tool_name, args
                                            )
                                            _sa_resolved = True
                                        elif _sa_decision == "deny":
                                            logger.info(
                                                "SmartApprove: auto-denied tool '%s'", tool_name
                                            )
                                            result_str = "此操作已被自动安全策略拒绝"
                                            tool_success = False
                                            _sa_resolved = True
                                        # else: "escalate" — fall through to interrupt path below

                                if not _sa_resolved:
                                    # Emit confirmation event via event_queue
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

                                    # Persist confirmation detail to Redis so the
                                    # resume path (_resume_tool_confirmation) can
                                    # read it back after the graph is interrupted.
                                    if confirmation_manager:
                                        import time as _time
                                        from app.domain.services.confirmation_manager import ConfirmationDetail
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

                                    # Interrupt — graph pauses here, resumes with user response
                                    user_response = interrupt({
                                        "type": "tool_confirmation",
                                        "tool_call_id": call_id,
                                    })

                                    action = user_response.get("action", "deny") if isinstance(user_response, dict) else "deny"
                                    if action == "approve":
                                        logger.info("tool_confirmation: user approved tool '%s'", tool_name)
                                        result_str, tool_success, multimodal_blocks = await _run_tool(tool_fn, tool_name, args)
                                    elif action == "timeout_fallback":
                                        result_str = "操作因超时被跳过。请尝试安全替代方案，或告知用户。"
                                        tool_success = False
                                        logger.info("tool_confirmation: timeout for tool '%s'", tool_name)
                                    else:
                                        result_str = "用户拒绝了此操作"
                                        tool_success = False
                                        logger.info("tool_confirmation: user denied tool '%s'", tool_name)
                        else:
                            # Assessment resolved to none/low risk — execute directly
                            result_str, tool_success, multimodal_blocks = await _run_tool(tool_fn, tool_name, args)
                    else:
                        # No risk metadata — execute directly (original path)
                        result_str, tool_success, multimodal_blocks = await _run_tool(tool_fn, tool_name, args)

            # D5: Record tool outcome in tracker + metrics
            if _tracker:
                if tool_success:
                    _tracker.record_success(tool_name, args)
                else:
                    _tracker.record_failure(tool_name, args)
            if _metrics and tool_name != "message_ask_user":
                # Exclude message_ask_user from latency stats (it's a gating mechanism,
                # not a real tool execution).
                _metrics.record_tool_call(
                    success=tool_success,
                    latency_ms=(_time.monotonic() - _tool_start) * 1000,
                )

            if not tool_success:
                new_failures += 1
                multimodal_blocks = []

            # Prefix error messages so the LLM can clearly identify failures
            content = f"[TOOL_ERROR] {result_str}" if not tool_success else result_str

            # Phase 2: 首次调用 Tier 1 skill 时注入 guide
            if tool_success and guide_injector:
                guide = guide_injector(tool_name)
                if guide:
                    content = f"{content}\n\n---\n[Skill Guide]\n{guide}"

            # Tier 1: 截断超长工具结果，保护 ReAct 循环期间上下文窗口
            content = truncate_tool_content(content, tool_result_max_chars)
            result_str = truncate_tool_content(result_str, tool_result_max_chars)

            new_messages.append(ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=tool_name,
            ))

            # Collect deferred HumanMessage for file_view multimodal results
            if multimodal_blocks:
                blocks: list[dict] = list(multimodal_blocks[:_MAX_FILE_VIEW_IMAGES])
                omitted = len(multimodal_blocks) - len(blocks)
                # Must include a text block so _compact_messages() and
                # _flatten_multimodal_content() can extract a meaningful summary
                # instead of falling back to str(content) JSON garbage.
                blocks.insert(0, {
                    "type": "text",
                    "text": f"[file_view: {tool_name} — {len(blocks)} image(s) loaded]",
                })
                if omitted > 0:
                    blocks.append({"type": "text", "text": f"[... {omitted} more images omitted]"})
                deferred_human_messages.append(HumanMessage(content=blocks))

            # Emit ToolEvent(CALLED) with correct success status
            new_events.append(
                ToolEvent(
                    tool_call_id=call_id,
                    tool_name=resolve_tool_source(tool_name).category,
                    function_name=tool_name,
                    function_args=args,
                    function_result=ToolResult(success=tool_success, message=result_str),
                    status=ToolEventStatus.CALLED,
                )
            )

        # Append deferred HumanMessages AFTER all ToolMessages.
        # Preserves AIMessage → ToolMessage* pairing for group_messages().
        new_messages.extend(deferred_human_messages)
        new_messages.extend(deferred_document_messages)

        result: dict = {
            "messages": new_messages,
            "events": new_events,
            "attempt_count": state["attempt_count"] + 1,
            "failure_count": state["failure_count"] + new_failures,
        }
        if should_interrupt:
            result["should_interrupt"] = True
        if not has_prior_soft_hint and any(
            m.content == "SOFT_HINT" and m.name == "message_ask_user"
            for m in new_messages
        ):
            result["soft_hint_sent"] = True
        return result

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

    def route_after_tool(state: ReactGraphState) -> str:
        """Route after tool execution: back to LLM."""
        if state.get("should_interrupt"):
            return END
        if state.get("attempt_count", 0) >= MAX_ITERATIONS:
            return END
        return "pre_llm_node"

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

    g.add_edge(START, "pre_llm_node")
    g.add_edge("pre_llm_node", "llm_node")
    g.add_conditional_edges("llm_node", route_after_llm)
    g.add_conditional_edges("tool_node", route_after_tool)

    return g.compile()
