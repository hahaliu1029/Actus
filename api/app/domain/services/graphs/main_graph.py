"""main_graph — outer orchestration as a LangGraph StateGraph.

Replaces PlannerReActFlow.invoke() while-loop.
Nodes: planner_node, executor_node, updater_node, summarizer_node
Edges: See design doc §4.2

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.1-4.2
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.models.event import (
    DoneEvent,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    TitleEvent,
)
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.repositories.uow import IUnitOfWork

from .state import MainGraphState

logger = logging.getLogger(__name__)


def build_main_graph(
    planner_llm: LLM,
    react_graph: Any,  # compiled react_graph
    json_parser: JSONParser,
    summary_llm: LLM,
    uow_factory: Callable[[], IUnitOfWork],
    session_id: str,
    agent_config: Any = None,
    prompts: Any = None,
) -> Any:
    """Build and compile the main orchestration graph.

    Parameters
    ----------
    planner_llm : LLM for plan generation/update.
    react_graph : Compiled react_graph for step execution.
    json_parser : JSON parser for extracting plan from LLM response.
    summary_llm : LLM for summary generation.
    uow_factory : Factory for UoW instances.
    session_id : Current session ID.
    """
    from app.domain.services.prompts.planner import PLANNER_SYSTEM_PROMPT, CREATE_PLAN_PROMPT

    # ---- Nodes --------------------------------------------------------- #

    async def planner_node(state: MainGraphState) -> dict:
        """Call planner LLM to create a plan from user message."""
        attachments = state.get("attachments", [])
        prompt = CREATE_PLAN_PROMPT.format(
            message=state["message"],
            attachments=", ".join(attachments) if attachments else "无",
        )

        # Build system prompt with optional conversation summaries
        system_content = PLANNER_SYSTEM_PROMPT
        conversation_summaries = state.get("conversation_summaries") or []
        if conversation_summaries:
            system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(conversation_summaries)

        response = await planner_llm.invoke(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.get("content", "")
        parsed = await json_parser.invoke(content)

        if not isinstance(parsed, dict):
            # Fallback: single-step plan
            parsed = {
                "title": "Task",
                "goal": state["message"],
                "language": state.get("language", "zh"),
                "steps": [{"description": state["message"]}],
                "message": "I'll help you with that.",
            }

        steps = [
            Step(description=s.get("description", ""))
            for s in parsed.get("steps", [])
        ]
        plan = Plan(
            title=parsed.get("title", "Task"),
            goal=parsed.get("goal", state["message"]),
            language=parsed.get("language", state.get("language", "zh")),
            steps=steps,
            message=parsed.get("message", ""),
            status=ExecutionStatus.RUNNING,
        )

        events = [
            TitleEvent(title=plan.title),
            MessageEvent(role="assistant", message=plan.message),
            PlanEvent(plan=plan, status=PlanEventStatus.CREATED),
        ]

        return {
            "plan": plan,
            "current_step": plan.get_next_step(),
            "flow_status": "executing",
            "original_request": plan.goal,
            "events": events,
        }

    async def executor_node(state: MainGraphState, config: RunnableConfig) -> dict:
        """Execute current step via react_graph sub-graph.

        Streams react events to the event_queue in real-time so the frontend
        sees tool calls / results as they happen, rather than after the entire
        step completes.
        """
        from app.domain.services.prompts.react import REACT_SYSTEM_PROMPT, EXECUTION_PROMPT

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        async def _emit(evt: Any) -> None:
            if event_queue is not None:
                await event_queue.put(evt)

        step = state["current_step"]
        if not step:
            return {"flow_status": "summarizing", "events": []}

        # Emit StepEvent(STARTED) immediately
        await _emit(StepEvent(step=step, status=StepEventStatus.STARTED))

        # Build initial messages with system prompt + execution prompt
        attachments = state.get("attachments", [])
        language = state.get("language", "zh")
        skill_context = state.get("skill_context", "")

        system_content = REACT_SYSTEM_PROMPT
        if skill_context:
            system_content += f"\n\n{skill_context}"

        # 注入历史对话摘要
        conversation_summaries = state.get("conversation_summaries") or []
        if conversation_summaries:
            system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(conversation_summaries)

        # 三分支 messages 构建逻辑
        is_resuming = state.get("is_resuming", False)
        saved_messages = state.get("messages") or []

        if is_resuming and saved_messages:
            # 中断恢复：保留消息 + 追加恢复提示
            initial_messages = saved_messages + [
                {"role": "user", "content": f"用户已完成接管并交还控制。请继续执行当前步骤：{step.description}\n用户消息：{state['message']}"},
            ]
        elif saved_messages:
            # 非首步/有历史：更新 system prompt 为最新版本，追加新 execution prompt
            saved_messages[0]["content"] = system_content
            initial_messages = saved_messages + [
                {"role": "user", "content": EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=", ".join(attachments) if attachments else "无",
                    language=language,
                    step=step.description,
                )},
            ]
        else:
            # 首步/无历史：干净的 system + execution prompt
            initial_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=", ".join(attachments) if attachments else "无",
                    language=language,
                    step=step.description,
                )},
            ]

        # Build react_graph input
        react_input = {
            "messages": initial_messages,
            "step_description": step.description,
            "original_request": state.get("original_request", ""),
            "language": language,
            "attachments": attachments,
            "events": [],
            "should_interrupt": False,
            "attempt_count": 0,
            "failure_count": 0,
        }

        # Stream react_graph — emit events in real-time
        react_final: dict[str, Any] = {}
        async for chunk in react_graph.astream(react_input):
            for _node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                react_final.update(node_output)
                # Push react events to frontend immediately
                for evt in node_output.get("events") or []:
                    await _emit(evt)

        # Extract execution summary and detect step success from LLM response JSON
        react_messages = react_final.get("messages", [])
        step_success = True
        summary = ""
        for msg in reversed(react_messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                summary = msg["content"][:500]
                try:
                    result_json = json.loads(msg["content"])
                    step_success = result_json.get("success", True)
                except (json.JSONDecodeError, TypeError):
                    pass  # Non-JSON response treated as success
                break

        step.status = ExecutionStatus.COMPLETED
        step.success = step_success
        await _emit(StepEvent(step=step, status=StepEventStatus.COMPLETED))

        # Check for interrupt
        should_interrupt = react_final.get("should_interrupt", False)
        if should_interrupt:
            return {
                "messages": react_final.get("messages", state["messages"]),
                "should_interrupt": True,
                "events": [],  # already emitted via queue
            }

        return {
            "messages": react_final.get("messages", state["messages"]),
            "execution_summary": summary,
            "flow_status": "updating",
            "events": [],  # already emitted via queue
        }

    async def updater_node(state: MainGraphState) -> dict:
        """Update the plan after step execution — mark step done, get next."""
        plan = state["plan"]
        if not plan:
            return {"flow_status": "summarizing", "events": []}

        # Find next step
        next_step = plan.get_next_step()
        if not next_step:
            return {
                "current_step": None,
                "flow_status": "summarizing",
                "events": [],
            }

        return {
            "current_step": next_step,
            "flow_status": "executing",
            "events": [],
        }

    async def summarizer_node(state: MainGraphState) -> dict:
        """Generate final summary."""
        plan = state["plan"]
        events = []

        if plan:
            plan.status = ExecutionStatus.COMPLETED
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED))

        events.append(DoneEvent())

        return {
            "flow_status": "completed",
            "events": events,
        }

    # ---- Routing ------------------------------------------------------- #

    def route_entry(state: MainGraphState) -> str:
        """Route from START based on flow_status."""
        status = state.get("flow_status", "idle")

        if status in ("idle", "planning"):
            return "planner_node"
        if status == "executing":
            return "executor_node"
        if status == "updating":
            return "updater_node"
        if status == "summarizing":
            return "summarizer_node"
        return "summarizer_node"

    def route_after_executor(state: MainGraphState) -> str:
        """Route after step execution."""
        if state.get("should_interrupt"):
            return END
        status = state.get("flow_status", "")
        if status == "updating":
            return "updater_node"
        if status == "summarizing":
            return "summarizer_node"
        return END

    def route_after_updater(state: MainGraphState) -> str:
        """Route after plan update."""
        status = state.get("flow_status", "")
        if status == "executing":
            return "executor_node"
        if status == "summarizing":
            return "summarizer_node"
        return END

    # ---- Build Graph --------------------------------------------------- #

    g: StateGraph = StateGraph(MainGraphState)

    g.add_node("planner_node", planner_node)
    g.add_node("executor_node", executor_node)
    g.add_node("updater_node", updater_node)
    g.add_node("summarizer_node", summarizer_node)

    g.add_conditional_edges(START, route_entry)
    g.add_edge("planner_node", "executor_node")
    g.add_conditional_edges("executor_node", route_after_executor)
    g.add_conditional_edges("updater_node", route_after_updater)
    g.add_edge("summarizer_node", END)

    return g.compile()
