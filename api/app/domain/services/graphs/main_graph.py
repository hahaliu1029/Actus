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
import re
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.application.errors.exceptions import ServerRequestsError
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
    WaitEvent,
)
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.flows.base import FlowStatus

from .message_utils import messages_to_dicts
from .state import MainGraphState

logger = logging.getLogger(__name__)

# Browser tool names whose content can be compacted
_BROWSER_COMPACT_TOOLS = frozenset(["browser_view", "browser_navigate"])


def _compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Compact step messages to reduce context size between steps.

    Mirrors the main-branch Memory.compact() behaviour:
    - Replace browser tool results with short summaries
    - Truncate very long tool results
    """
    compacted: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.name in _BROWSER_COMPACT_TOOLS:
            # Extract title and short preview from HTML content
            content = msg.content
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL
            )
            title = title_match.group(1).strip() if title_match else ""
            plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content)).strip()
            preview = plain[:200]
            parts = [f"[已执行] {msg.name}:"]
            if title:
                parts.append(f"页面标题: {title},")
            parts.append(f"内容摘要: {preview}")
            compacted.append(ToolMessage(
                content=" ".join(parts),
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        elif isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > 2000:
            # Truncate very long tool results to keep context manageable
            compacted.append(ToolMessage(
                content=msg.content[:2000] + "\n...(已截断)",
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        else:
            compacted.append(msg)
    return compacted


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
    from app.domain.services.prompts.planner import PLANNER_SYSTEM_PROMPT, CREATE_PLAN_PROMPT, UPDATE_PLAN_PROMPT

    # ---- Nodes --------------------------------------------------------- #

    async def planner_node(state: MainGraphState) -> dict:
        """Call planner LLM to create a plan from user message."""
        attachments = state.get("attachments", [])
        prompt = CREATE_PLAN_PROMPT.format(
            message=state["message"],
            attachments=", ".join(attachments) if attachments else "无",
        )

        # Build system prompt with optional tool summary and conversation summaries
        system_content = PLANNER_SYSTEM_PROMPT

        # Inject available tool summary so planner knows about dedicated tools
        # (e.g. brainstorm_skill, generate_skill) and can plan accordingly
        skill_context = state.get("skill_context") or ""
        tool_summary_marker = "## Available Tool Summary"
        if tool_summary_marker in skill_context:
            tool_summary = skill_context[skill_context.index(tool_summary_marker):]
            system_content += f"\n\n{tool_summary}"

        conversation_summaries = state.get("conversation_summaries") or []
        if conversation_summaries:
            system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(conversation_summaries)

        # planner_llm is the raw Actus LLM (not LangChain adapter) — uses dict messages
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
                "message": "好的，我来帮你处理。",
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
            "flow_status": FlowStatus.EXECUTING.value,
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
            return {
                "flow_status": FlowStatus.SUMMARIZING.value,
                "events": [],
                "messages": state.get("messages", []),
            }

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

        # 三分支 messages 构建逻辑（使用 LangChain 消息类型）
        is_resuming = state.get("is_resuming", False)
        saved_messages: list[BaseMessage] = state.get("messages") or []

        if is_resuming and saved_messages:
            # 中断恢复：保留消息 + 追加恢复提示
            # 检查中断类型：如果最后一条 ToolMessage 是 message_ask_user 的回复，
            # 说明是用户回答问题（非浏览器/终端接管），使用不同的提示措辞
            last_tool = next(
                (m for m in reversed(saved_messages)
                 if isinstance(m, ToolMessage) and m.name == "message_ask_user"),
                None,
            )
            if last_tool is not None:
                resume_hint = (
                    f"用户已回复你之前的提问。请根据用户回复继续执行当前步骤：{step.description}\n"
                    f"用户回复：{state['message']}"
                )
            else:
                resume_hint = (
                    f"用户已完成接管并交还控制。请继续执行当前步骤：{step.description}\n"
                    f"用户消息：{state['message']}"
                )
            initial_messages = saved_messages + [HumanMessage(content=resume_hint)]
        elif saved_messages:
            # 非首步/有历史：更新 system prompt 为最新版本，追加新 execution prompt
            first = saved_messages[0]
            updated_first = SystemMessage(content=system_content) if isinstance(first, SystemMessage) else first
            initial_messages = [
                updated_first,
                *saved_messages[1:],
                HumanMessage(content=EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=", ".join(attachments) if attachments else "无",
                    language=language,
                    step=step.description,
                )),
            ]
        else:
            # 首步/无历史：干净的 system + execution prompt
            initial_messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=", ".join(attachments) if attachments else "无",
                    language=language,
                    step=step.description,
                )),
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
        # 注意：should_interrupt 需要单独跟踪，避免后续 chunk 覆盖
        react_final: dict[str, Any] = {}
        seen_interrupt = False
        async for chunk in react_graph.astream(react_input):
            for _node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                if node_output.get("should_interrupt"):
                    seen_interrupt = True
                react_final.update(node_output)
                # Push react events to frontend immediately
                for evt in node_output.get("events") or []:
                    await _emit(evt)
        if seen_interrupt:
            react_final["should_interrupt"] = True

        # Extract execution summary and detect step success from LLM response
        react_messages: list = react_final.get("messages", [])
        step_success = True
        summary = ""
        for msg in reversed(react_messages):
            if isinstance(msg, AIMessage) and msg.content:
                summary = msg.content[:500]
                try:
                    result_json = json.loads(msg.content)
                    step_success = result_json.get("success", True)
                except (json.JSONDecodeError, TypeError):
                    pass  # Non-JSON response treated as success
                break

        # Compact messages to reduce context size for next step
        # (mirrors main-branch compact_memory behaviour)
        final_messages = _compact_messages(
            react_final.get("messages", state.get("messages", []))
        )

        # Check for interrupt (user takeover request) BEFORE marking step as completed.
        # 中断时步骤尚未完成，保持 RUNNING 状态以便恢复后继续执行。
        should_interrupt = react_final.get("should_interrupt", False)
        if should_interrupt:
            # 中断的步骤不标记为 COMPLETED，保留 RUNNING 状态
            await _emit(WaitEvent())
            return {
                "messages": final_messages,
                "current_step": step,
                "execution_summary": summary,
                "should_interrupt": True,
                "flow_status": FlowStatus.EXECUTING.value,
                "original_request": state.get("original_request", ""),
                "events": [],  # WaitEvent 已通过 _emit 发送，避免重复
            }

        # 通过 model_copy 创建新对象避免直接变异 state 对象
        # （LangGraph 要求节点返回 partial update dict，不可直接修改 state）
        step = step.model_copy(update={
            "status": ExecutionStatus.COMPLETED,
            "success": step_success,
            "result": summary,
        })
        await _emit(StepEvent(step=step, status=StepEventStatus.COMPLETED))

        return {
            "messages": final_messages,
            "current_step": step,
            "execution_summary": summary,
            "flow_status": FlowStatus.UPDATING.value,
            "events": [],  # already emitted via queue
        }

    async def updater_node(state: MainGraphState, config: RunnableConfig) -> dict:
        """Update the plan after step execution — mark step done, call planner
        to update remaining steps with execution context, then get next step.

        This mirrors the main-branch PlannerReActFlow UPDATING phase:
        1. Sync completed step back into plan
        2. Call planner LLM with UPDATE_PLAN_PROMPT (execution_summary)
        3. Replace pending steps with planner's updated steps
        4. Emit PlanEvent(UPDATED)
        """
        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        plan = state["plan"]
        if not plan:
            return {"flow_status": FlowStatus.SUMMARIZING.value, "events": []}

        # 1. Sync completed step back into plan.steps
        completed_step = state.get("current_step")
        if completed_step and completed_step.done:
            updated_steps = [
                completed_step if s.id == completed_step.id else s
                for s in plan.steps
            ]
            plan = plan.model_copy(update={"steps": updated_steps})

        # 2. Call planner LLM to update remaining steps based on execution results
        execution_summary = state.get("execution_summary", "")
        plan_updated = False
        if completed_step and execution_summary:
            try:
                query = UPDATE_PLAN_PROMPT.format(
                    plan=plan.model_dump_json(),
                    step=completed_step.model_dump_json(),
                    execution_summary=execution_summary or "无额外执行详情",
                )

                system_content = PLANNER_SYSTEM_PROMPT
                # Inject tool summary so planner can reference available tools
                skill_context = state.get("skill_context") or ""
                tool_summary_marker = "## Available Tool Summary"
                if tool_summary_marker in skill_context:
                    tool_summary = skill_context[skill_context.index(tool_summary_marker):]
                    system_content += f"\n\n{tool_summary}"

                response = await planner_llm.invoke(
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": query},
                    ],
                    response_format={"type": "json_object"},
                )

                content = response.get("content", "")
                parsed_obj = await json_parser.invoke(content)

                if isinstance(parsed_obj, dict):
                    new_steps_raw = parsed_obj.get("steps", [])
                    new_steps = [
                        Step(
                            description=s.get("description", ""),
                            id=s.get("id", ""),
                        )
                        for s in new_steps_raw
                        if isinstance(s, dict) and s.get("description")
                    ]

                    if new_steps:
                        # Find first pending step index
                        first_pending_index = None
                        for idx, s in enumerate(plan.steps):
                            if not s.done:
                                first_pending_index = idx
                                break

                        if first_pending_index is not None:
                            # Preserve completed steps, replace pending with new
                            merged = list(plan.steps[:first_pending_index]) + new_steps
                            plan = plan.model_copy(update={"steps": merged})

                        plan_updated = True
                        logger.info(
                            "Planner 更新计划成功，新步骤数: %d", len(new_steps)
                        )
            except Exception as exc:
                # Plan update failure is non-fatal — continue with original plan
                logger.warning("Planner 更新计划失败，沿用原计划继续: %s", exc)

        events: list = []
        if plan_updated:
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.UPDATED))
            if event_queue is not None:
                for evt in events:
                    await event_queue.put(evt)

        # 3. Find next step
        next_step = plan.get_next_step()
        if not next_step:
            return {
                "plan": plan,
                "current_step": None,
                "flow_status": FlowStatus.SUMMARIZING.value,
                "events": events,
            }

        return {
            "plan": plan,
            "current_step": next_step,
            "flow_status": FlowStatus.EXECUTING.value,
            "events": events,
        }

    async def summarizer_node(state: MainGraphState) -> dict:
        """Generate final summary and emit completion events."""
        from app.domain.services.prompts.react import SUMMARIZE_PROMPT

        plan = state["plan"]
        events = []

        if plan:
            # 通过 model_copy 避免直接变异 state 对象
            plan = plan.model_copy(update={"status": ExecutionStatus.COMPLETED})
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED))

        # Call LLM to generate a user-facing summary
        # summary_llm is raw Actus LLM — needs dict messages
        react_messages: list = state.get("messages", [])
        if react_messages:
            try:
                dict_messages = messages_to_dicts(react_messages) if react_messages else []
                response = await summary_llm.invoke(
                    messages=dict_messages + [
                        {"role": "user", "content": SUMMARIZE_PROMPT},
                    ],
                )
                summary_content = response.get("content", "")
                if summary_content:
                    try:
                        parsed = json.loads(summary_content)
                        if isinstance(parsed, dict) and parsed.get("message"):
                            events.append(MessageEvent(
                                role="assistant",
                                message=parsed["message"],
                            ))
                        else:
                            events.append(MessageEvent(role="assistant", message=summary_content))
                    except (json.JSONDecodeError, TypeError):
                        events.append(MessageEvent(role="assistant", message=summary_content))
            except Exception as exc:
                logger.warning(f"Summarizer LLM call failed: {exc}")

        events.append(DoneEvent())

        result: dict = {
            "flow_status": FlowStatus.COMPLETED.value,
            "events": events,
        }
        if plan:
            result["plan"] = plan
        return result

    # ---- Routing ------------------------------------------------------- #

    def route_entry(state: MainGraphState) -> Literal[
        "planner_node", "executor_node", "updater_node", "summarizer_node",
    ]:
        """Route from START based on flow_status."""
        status = state.get("flow_status", FlowStatus.IDLE.value)

        if status in (FlowStatus.IDLE.value, FlowStatus.PLANNING.value):
            return "planner_node"
        if status == FlowStatus.EXECUTING.value:
            return "executor_node"
        if status == FlowStatus.UPDATING.value:
            return "updater_node"
        return "summarizer_node"

    def route_after_executor(state: MainGraphState) -> str:
        """Route after step execution."""
        if state.get("should_interrupt"):
            return END
        status = state.get("flow_status", "")
        if status == FlowStatus.UPDATING.value:
            return "updater_node"
        if status == FlowStatus.SUMMARIZING.value:
            return "summarizer_node"
        return END

    def route_after_updater(state: MainGraphState) -> str:
        """Route after plan update."""
        status = state.get("flow_status", "")
        if status == FlowStatus.EXECUTING.value:
            return "executor_node"
        if status == FlowStatus.SUMMARIZING.value:
            return "summarizer_node"
        return END

    # ---- Build Graph --------------------------------------------------- #

    g: StateGraph = StateGraph(MainGraphState)

    # RetryPolicy for transient planner LLM errors
    planner_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=2.0,
        backoff_factor=2.0,
        retry_on=ServerRequestsError,
    )

    g.add_node("planner_node", planner_node, retry_policy=planner_retry)
    g.add_node("executor_node", executor_node)
    g.add_node("updater_node", updater_node)
    g.add_node("summarizer_node", summarizer_node)

    g.add_conditional_edges(START, route_entry)
    g.add_edge("planner_node", "executor_node")
    g.add_conditional_edges("executor_node", route_after_executor)
    g.add_conditional_edges("updater_node", route_after_updater)
    g.add_edge("summarizer_node", END)

    return g.compile()
