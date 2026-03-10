"""react_graph — inner ReAct loop as a LangGraph StateGraph.

Replaces BaseAgent.invoke() and ReActAgent.execute_step().
Nodes: llm_node, tool_node
Edges: START → llm_node → route_after_llm → (tool_node → llm_node) | END

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.3-4.4
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.domain.models.event import (
    MessageEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import ToolResult

from .state import ReactGraphState

logger = logging.getLogger(__name__)

# Max ReAct iterations to prevent infinite loops
MAX_ITERATIONS = 30


def build_react_graph(llm: Any, tools: list, agent_config: Any = None) -> Any:
    """Build and compile the inner ReAct loop graph.

    Parameters
    ----------
    llm : LangChain BaseChatModel (or LLMAdapter) — must support bind_tools.
    tools : List of LangChain tools.
    agent_config : Optional AgentConfig for iteration limits etc.
    """
    # Build tool lookup
    tool_map: dict[str, Any] = {t.name: t for t in tools}

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    # ---- Nodes --------------------------------------------------------- #

    async def llm_node(state: ReactGraphState) -> dict:
        """Call the LLM with current messages."""
        messages = state["messages"]
        response: AIMessage = await llm_with_tools.ainvoke(messages)

        new_events = []

        # Convert response to dict message for storage
        msg_dict: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }
        if response.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"])
                        if isinstance(tc["args"], dict)
                        else tc["args"],
                    },
                }
                for tc in response.tool_calls
            ]
            # Emit ToolEvent(CALLING) for each tool call
            for tc in response.tool_calls:
                new_events.append(
                    ToolEvent(
                        tool_call_id=tc["id"],
                        tool_name=tc["name"],
                        function_name=tc["name"],
                        function_args=tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"]),
                        status=ToolEventStatus.CALLING,
                    )
                )

        # 最终回答（无 tool_calls 且有内容）发射 MessageEvent，使前端实时收到
        if not response.tool_calls and response.content:
            new_events.append(
                MessageEvent(role="assistant", message=response.content)
            )

        return {
            "messages": state["messages"] + [msg_dict],
            "events": new_events,
        }

    async def tool_node(state: ReactGraphState) -> dict:
        """Execute tool calls from the last assistant message.

        Special handling for ``message_ask_user``:
        - If ``suggest_user_takeover`` is "browser"/"shell" → set should_interrupt
          (handled by confirmation_check, but also guard here).
        - Otherwise, first call returns SOFT_HINT (agent should try to solve
          autonomously). If a SOFT_HINT was already returned in this step
          and the LLM calls again, it truly needs user input → interrupt.
        """
        messages = state["messages"]
        last_msg = messages[-1]
        tool_calls = last_msg.get("tool_calls") or []

        # Check if there was already a SOFT_HINT in this step's history
        has_prior_soft_hint = any(
            m.get("content") == "SOFT_HINT"
            and m.get("function_name") == "message_ask_user"
            for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
        )

        new_messages = []
        new_events = []
        should_interrupt = False

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            call_id = tc.get("id", "")

            # ---- message_ask_user: SOFT_HINT gating ---- #
            tool_success = True  # default; overridden in normal-tool branch on failure
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
                else:
                    try:
                        result_str = await tool_fn.ainvoke(args)
                        if not isinstance(result_str, str):
                            result_str = str(result_str)
                        # Detect failure indicators from upstream ToolResult
                        tool_success = "success=False" not in result_str
                    except Exception as exc:
                        result_str = f"Error executing {tool_name}: {exc}"
                        tool_success = False

            # Prefix error messages so the LLM can clearly identify failures
            content = f"[TOOL_ERROR] {result_str}" if not tool_success else result_str

            new_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
                "function_name": tool_name,
            })

            # Emit ToolEvent(CALLED) with correct success status
            new_events.append(
                ToolEvent(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    function_name=tool_name,
                    function_args=args,
                    function_result=ToolResult(success=tool_success, message=result_str),
                    status=ToolEventStatus.CALLED,
                )
            )

        result: dict = {
            "messages": state["messages"] + new_messages,
            "events": new_events,
            "attempt_count": state["attempt_count"] + 1,
        }
        if should_interrupt:
            result["should_interrupt"] = True
        return result

    async def confirmation_check(state: ReactGraphState) -> dict:
        """Check if any tool call requires user confirmation.

        Note: takeover requests (message_ask_user with suggest_user_takeover)
        are handled by tool_node, which executes the tool first to generate
        a proper tool response (WAITING_FOR_USER), then sets should_interrupt.
        This ensures complete tool_call→tool_response pairs in message history,
        which is required by LLMs on conversation resume.
        """
        # Currently a pass-through — all tool calls are routed to tool_node.
        # Future: add confirmation logic for dangerous tools here.
        return {}

    # ---- Routing ------------------------------------------------------- #

    def route_after_llm(state: ReactGraphState) -> str:
        """Route after LLM call: tool calls → confirmation_check, else END."""
        if state.get("should_interrupt"):
            return END

        messages = state["messages"]
        if not messages:
            return END

        last_msg = messages[-1]
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            return "confirmation_check"

        return END

    def route_after_confirmation(state: ReactGraphState) -> str:
        """Route after confirmation check."""
        if state.get("should_interrupt"):
            return END
        return "tool_node"

    def route_after_tool(state: ReactGraphState) -> str:
        """Route after tool execution: back to LLM."""
        if state.get("should_interrupt"):
            return END
        if state.get("attempt_count", 0) >= MAX_ITERATIONS:
            return END
        return "llm_node"

    # ---- Build Graph --------------------------------------------------- #

    g: StateGraph = StateGraph(ReactGraphState)

    g.add_node("llm_node", llm_node)
    g.add_node("tool_node", tool_node)
    g.add_node("confirmation_check", confirmation_check)

    g.add_edge(START, "llm_node")
    g.add_conditional_edges("llm_node", route_after_llm)
    g.add_conditional_edges("confirmation_check", route_after_confirmation)
    g.add_conditional_edges("tool_node", route_after_tool)

    return g.compile()
