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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.application.errors.exceptions import ServerRequestsError
from app.domain.external.file_processor import FileProcessResult
from app.domain.models.app_config import AgentConfig
from app.domain.models.event import (
    MessageEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import ToolResult

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


# Tool name → category mapping (mirrors agent_task_runner._classify_tool_name)
_TOOL_CATEGORY_PREFIXES = {
    "browser_": "browser",
    "shell_": "shell",
    "file_": "file",
    "search_": "search",
    "message_": "message",
}
_KNOWN_CATEGORIES = frozenset(
    {"browser", "shell", "file", "search", "message", "mcp", "a2a", "skill"}
)


def _classify_tool_name(tool_name: str) -> str:
    """Extract tool category from LangChain tool name.

    e.g., "browser_navigate" → "browser", "shell_execute" → "shell".
    Keeps names like "mcp", "browser" as-is if already a category.
    """
    if tool_name in _KNOWN_CATEGORIES:
        return tool_name
    for prefix, category in _TOOL_CATEGORY_PREFIXES.items():
        if tool_name.startswith(prefix):
            return category
    return tool_name


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

    async def llm_node(state: ReactGraphState) -> dict:
        """Call the LLM with current messages."""
        messages = state.get("llm_input_messages") or state["messages"]

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

        response: AIMessage = await llm_with_tools.ainvoke(messages)

        new_events = []

        # Emit ToolEvent(CALLING) for each tool call
        if response.tool_calls:
            for tc in response.tool_calls:
                func_name = tc["name"]
                new_events.append(
                    ToolEvent(
                        tool_call_id=tc["id"],
                        tool_name=_classify_tool_name(func_name),
                        function_name=func_name,
                        function_args=tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"]),
                        status=ToolEventStatus.CALLING,
                    )
                )

        # 最终回答（无 tool_calls 且有内容）发射 MessageEvent，使前端实时收到
        # LLM 按 system prompt 要求返回 JSON 格式 {"success","result","attachments"}，
        # 需要提取 result 字段作为用户可读消息，避免前端显示原始 JSON。
        if not response.tool_calls and response.content:
            display_message = response.content
            if isinstance(display_message, str):
                try:
                    parsed = json.loads(display_message)
                    if isinstance(parsed, dict) and "result" in parsed:
                        extracted = parsed["result"]
                        if isinstance(extracted, str) and extracted.strip():
                            display_message = extracted
                except (json.JSONDecodeError, ValueError):
                    pass
            new_events.append(
                MessageEvent(role="assistant", message=display_message)
            )

        return {
            "messages": [response],
            "events": new_events,
        }

    async def tool_node(state: ReactGraphState, config: RunnableConfig) -> dict:
        """Execute tool calls from the last assistant message.

        Special handling for ``message_ask_user``:
        - If ``suggest_user_takeover`` is "browser"/"shell" → set should_interrupt
          (handled by confirmation_check, but also guard here).
        - Otherwise, first call returns SOFT_HINT (agent should try to solve
          autonomously). If a SOFT_HINT was already returned in this step
          and the LLM calls again, it truly needs user input → interrupt.
        """
        guide_injector = (config or {}).get("configurable", {}).get("skill_guide_injector") if config else None

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

        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            call_id = tc["id"]

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
                # ---- Normal tool execution ---- #
                tool_fn = tool_map.get(tool_name)
                if tool_fn is None:
                    result_str = f"Error: Unknown tool '{tool_name}'"
                    tool_success = False
                elif getattr(tool_fn, "metadata", {}).get("require_confirmation"):
                    # Runtime enforcement: dangerous tools must go through
                    # user confirmation via interrupt, not just prompt hints.
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                    logger.info(f"require_confirmation: interrupting for tool '{tool_name}'")
                else:
                    try:
                        raw_result = await tool_fn.ainvoke(args)
                        if isinstance(raw_result, FileProcessResult):
                            result_str = raw_result.text
                            multimodal_blocks = list(raw_result.image_blocks)
                            if raw_result.document_blocks:
                                doc_blocks: list[dict] = list(raw_result.document_blocks)
                                doc_blocks.insert(0, {"type": "text", "text": "[file_view: PDF document attached]"})
                                deferred_document_messages.append(HumanMessage(content=doc_blocks))
                        elif isinstance(raw_result, str):
                            result_str = raw_result
                        else:
                            result_str = str(raw_result)
                        # Shell image detection (M1d)
                        if tool_name in ("shell_execute", "shell_read_output"):
                            result_str, shell_images = _extract_shell_images(result_str)
                            multimodal_blocks.extend(shell_images)
                    except Exception as exc:
                        result_str = f"Error executing {tool_name}: {exc}"
                        tool_success = False

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
                    tool_name=_classify_tool_name(tool_name),
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
