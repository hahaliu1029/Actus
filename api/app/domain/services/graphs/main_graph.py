"""main_graph — outer orchestration as a LangGraph StateGraph.

Replaces PlannerReActFlow.invoke() while-loop.
Nodes: planner_node, executor_node, updater_node, summarizer_node
Edges: See design doc §4.2

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.1-4.2
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any, Callable, Literal, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .message_utils import truncate_tool_content
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from app.domain.models.app_config import AgentConfig

from app.application.errors.exceptions import ServerRequestsError
from app.domain.models.file import File
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef, SummarizerOutput
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

from .message_utils import build_multimodal_content, dedup_messages, format_attachments_text
from .state import MainGraphState

if TYPE_CHECKING:
    from .context_assembler import ContextAssembler

logger = logging.getLogger(__name__)

# Browser tool names whose content can be compacted
_BROWSER_COMPACT_TOOLS = frozenset(["browser_view", "browser_navigate"])


def _compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Compact step messages to reduce context size between steps.

    Mirrors the main-branch Memory.compact() behaviour:
    - Replace browser tool results with short summaries
    - Truncate very long tool results
    - Strip multimodal image blocks from HumanMessage to avoid re-sending
      large base64 payloads on every subsequent step
    """
    compacted: list[BaseMessage] = []
    for msg in messages:
        # Strip multimodal image content blocks — images were already "seen" in step 1,
        # subsequent steps only need the text portion to avoid bloating context with base64.
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            text_parts = [
                block.get("text", "")
                for block in msg.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text_only = "\n".join(text_parts) if text_parts else str(msg.content)
            compacted.append(HumanMessage(content=text_only))
            continue
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
            # Tier 2: 截断超长工具结果，压缩跨 step 存储
            compacted.append(ToolMessage(
                content=truncate_tool_content(msg.content, 2000),
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        else:
            compacted.append(msg)
    return compacted


def build_main_graph(
    planner_llm: BaseChatModel,
    react_graph: CompiledStateGraph,
    summary_llm: BaseChatModel,
    uow_factory: Callable[[], IUnitOfWork],
    session_id: str,
    agent_config: AgentConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    assembler: ContextAssembler | None = None,
    supports_vision: bool = True,
) -> CompiledStateGraph:
    """Build and compile the main orchestration graph.

    Parameters
    ----------
    planner_llm : BaseChatModel for plan generation/update (structured output).
    react_graph : Compiled react_graph for step execution.
    summary_llm : BaseChatModel for summary generation (streaming via astream).
    uow_factory : Factory for UoW instances.
    session_id : Current session ID.
    checkpointer : LangGraph checkpointer for interrupt/resume support.
    """
    from app.domain.services.prompts.planner import PLANNER_SYSTEM_PROMPT, CREATE_PLAN_PROMPT, UPDATE_PLAN_PROMPT

    # ---- Nodes --------------------------------------------------------- #

    async def planner_node(state: MainGraphState) -> dict:
        """Call planner LLM to create a plan from user message."""
        attachments = state.get("attachments", [])
        image_blocks = state.get("image_content_blocks", [])
        # Planner 不传图片但需要知道附件包含图片，使用 planner 专用提示
        prompt = CREATE_PLAN_PROMPT.format(
            message=state["message"],
            attachments=format_attachments_text(
                attachments, has_image_blocks=bool(image_blocks), for_planner=True,
                supports_vision=supports_vision,
            ),
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

        # Planner 不传图片：planner 识图不可靠，容易幻觉图片内容并写入 step description，
        # 导致 executor 被错误的描述误导。图片分析留给 executor 通过 MCP 工具完成。
        # 附件文本信息（路径、URL）仍保留，让 planner 知道有附件存在。
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]
        structured_llm = planner_llm.with_structured_output(PlanResponse)

        try:
            parsed: PlanResponse = await structured_llm.ainvoke(messages)
        except Exception:
            logger.warning("Planner structured output failed, using fallback plan")
            parsed = PlanResponse(
                title="Task",
                goal=state["message"],
                language=state.get("language", "zh"),
                steps=[StepDef(description=state["message"])],
                message="好的，我来帮你处理。",
            )

        steps = [
            Step(description=s.description)
            for s in parsed.steps
        ]
        plan = Plan(
            title=parsed.title or "Task",
            goal=parsed.goal or state["message"],
            language=parsed.language or state.get("language", "zh"),
            steps=steps,
            message=parsed.message or "",
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

    async def executor_node(
        state: MainGraphState, config: RunnableConfig,
    ) -> Command[Literal["updater_node", "interrupt_node", "__end__"]]:
        """Execute current step via react_graph sub-graph.

        Streams react events to the event_queue in real-time so the frontend
        sees tool calls / results as they happen, rather than after the entire
        step completes.
        """
        from app.domain.services.prompts.react import REACT_SYSTEM_PROMPT, EXECUTION_PROMPT
        from app.domain.services.execution_watchdog import _should_terminate

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        async def _emit(evt: Any) -> None:
            if event_queue is not None:
                await event_queue.put(evt)

        # D5: Cooperative termination check
        if _should_terminate(config):
            logger.info("executor_node: should_terminate=True, routing to END")
            return Command(
                update={
                    "flow_status": FlowStatus.COMPLETED.value,
                    "events": [],
                    "messages": state.get("messages", []),
                },
                goto=END,
            )

        # D5: Reset tracker blocked set at step boundary
        tracker = config.get("configurable", {}).get("tool_failure_tracker")
        if tracker is not None:
            tracker.reset_blocked()

        step = state["current_step"]
        if not step:
            logger.warning("executor_node: current_step is None, routing to END")
            return Command(
                update={
                    "flow_status": FlowStatus.COMPLETED.value,
                    "events": [],
                    "messages": state.get("messages", []),
                },
                goto=END,
            )

        # Phase 3: 获取当前 step 的编译后 react_graph（渐进式 Skill 加载）
        # 传递步骤描述，使 provider 能按步骤选择相关 Skill 工具
        react_graph_provider = (config.get("configurable") or {}).get("react_graph_provider")
        if react_graph_provider:
            try:
                step_react = await react_graph_provider(step.description)
            except Exception:
                logger.warning("react_graph_provider 失败，使用默认（无动态Skill工具）")
                step_react = react_graph
        else:
            step_react = react_graph

        resume_value = state.get("resume_value")

        # Emit StepEvent(STARTED) — skip on resume to avoid duplicate events
        if resume_value is None:
            await _emit(StepEvent(step=step, status=StepEventStatus.STARTED))

        # Build initial messages with system prompt + execution prompt
        attachments = state.get("attachments", [])
        image_blocks = state.get("image_content_blocks", [])
        language = state.get("language", "zh")
        skill_context = state.get("skill_context", "")

        system_content = REACT_SYSTEM_PROMPT

        # Inject file_view hint when the tool is available
        has_file_view = (config.get("configurable") or {}).get("has_file_view", False)
        if has_file_view:
            from app.domain.services.prompts.react import FILE_VIEW_HINT
            system_content += FILE_VIEW_HINT

        # Inject memory tools hint when available (C6)
        has_memory_tools = (config.get("configurable") or {}).get("has_memory_tools", False)
        if has_memory_tools:
            from app.domain.services.prompts.react import MEMORY_TOOLS_HINT
            system_content += MEMORY_TOOLS_HINT

        if skill_context:
            system_content += f"\n\n{skill_context}"

        # 注入历史对话摘要
        conversation_summaries = state.get("conversation_summaries") or []
        if conversation_summaries:
            system_content += "\n\n## 历史对话摘要\n" + "\n\n".join(conversation_summaries)

        # 两分支 messages 构建逻辑（使用 LangChain 消息类型）
        # resume_value 由 interrupt_node 在恢复时设置，或由 DB fallback 路径直接注入
        saved_messages: list[BaseMessage] = state.get("messages") or []

        if resume_value is not None:
            # Resume path: messages 已由 checkpointer 保存，追加用户回复
            last_tool = next(
                (m for m in reversed(saved_messages)
                 if isinstance(m, ToolMessage) and m.name == "message_ask_user"),
                None,
            )
            if last_tool is not None:
                resume_hint = (
                    f"用户已回复你之前的提问。请根据用户回复继续执行当前步骤：{step.description}\n"
                    f"用户回复：{resume_value}"
                )
            else:
                resume_hint = (
                    f"用户已完成接管并交还控制。请继续执行当前步骤：{step.description}\n"
                    f"用户消息：{resume_value}"
                )
            initial_messages = saved_messages + [HumanMessage(content=resume_hint)]
        elif saved_messages:
            # 非首步/有历史：更新 system prompt 为最新版本，追加新 execution prompt
            # 图片已在首步消息中（已被 compact 剥离），无需重复添加
            first = saved_messages[0]
            updated_first = SystemMessage(content=system_content) if isinstance(first, SystemMessage) else first
            initial_messages = [
                updated_first,
                *saved_messages[1:],
                HumanMessage(content=EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=format_attachments_text(attachments),
                    language=language,
                    step=step.description,
                )),
            ]
        else:
            # 首步/无历史：干净的 system + execution prompt（含图片多模态内容）
            attachments_text = format_attachments_text(
                attachments, has_image_blocks=bool(image_blocks),
                supports_vision=supports_vision,
            )
            execution_text = EXECUTION_PROMPT.format(
                message=state["message"],
                attachments=attachments_text,
                language=language,
                step=step.description,
            )
            logger.debug("[IMG] executor_node: image_blocks=%d, attachments=%s", len(image_blocks), attachments)
            prompt_content = build_multimodal_content(execution_text, image_blocks)
            logger.debug("[IMG] executor_node: prompt_content type=%s, is_list=%s", type(prompt_content).__name__, isinstance(prompt_content, list))
            initial_messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=prompt_content),
            ]

        # Build react_graph input
        react_input = {
            "messages": initial_messages,
            "step_description": step.description,
            "original_request": state.get("original_request", ""),
            "language": language,
            "attachments": attachments,
            "image_content_blocks": image_blocks,
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        }

        # ── Cross-step context assembly ──
        if assembler is not None and resume_value is None:
            asm_result = assembler.assemble(react_input["messages"])
            react_input["messages"] = asm_result.messages
            if asm_result.actions:
                logger.info("context_assembler(cross-step): %s", asm_result.actions)

        # Stream react_graph — emit events in real-time.
        # IMPORTANT: Must use stream_mode="updates" explicitly.
        # LangGraph 1.0.x defaults to "values" (full state per chunk), but we
        # need "updates" ({node_name: state_update} per chunk) so that we can
        # accumulate only NEW messages from each node and emit events in order.
        react_final: dict[str, Any] = {}
        all_react_messages: list = []
        seen_interrupt = False
        async for chunk in step_react.astream(
            react_input, config=config, stream_mode="updates",
        ):
            for _node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                # Accumulate messages separately — pop before update to
                # prevent react_final["messages"] from being overwritten
                all_react_messages.extend(node_output.pop("messages", []))
                if node_output.get("should_interrupt"):
                    seen_interrupt = True
                react_final.update(node_output)
                # Push react events to frontend immediately
                for evt in node_output.get("events") or []:
                    await _emit(evt)
        # Reconstruct full message list: initial input + all new messages,
        # deduplicating by ID to replicate add_messages semantics.
        react_final["messages"] = dedup_messages(initial_messages + all_react_messages)
        if seen_interrupt:
            react_final["should_interrupt"] = True

        # Extract execution summary from last AI message
        react_messages: list = react_final.get("messages", [])
        summary = ""
        for msg in reversed(react_messages):
            if isinstance(msg, AIMessage) and msg.content:
                summary = msg.content[:500]
                break

        # Detect step success from react_graph's accumulated failure_count
        step_success = react_final.get("failure_count", 0) == 0

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
            return Command(
                update={
                    "messages": final_messages,
                    "current_step": step,
                    "execution_summary": summary,
                    "should_interrupt": True,
                    "resume_value": None,
                    "flow_status": FlowStatus.EXECUTING.value,
                    "original_request": state.get("original_request", ""),
                    "events": [],  # WaitEvent 已通过 _emit 发送，避免重复
                },
                goto="interrupt_node",
            )

        # 通过 model_copy 创建新对象避免直接变异 state 对象
        # （LangGraph 要求节点返回 partial update dict，不可直接修改 state）
        step = step.model_copy(update={
            "status": ExecutionStatus.COMPLETED,
            "success": step_success,
            "result": summary,
        })
        await _emit(StepEvent(step=step, status=StepEventStatus.COMPLETED))

        # D5: Record step outcome in metrics
        _metrics = config.get("configurable", {}).get("execution_metrics")
        if _metrics is not None:
            if step_success:
                _metrics.steps_completed += 1
            else:
                _metrics.steps_failed += 1

        return Command(
            update={
                "messages": final_messages,
                "current_step": step,
                "execution_summary": summary,
                "resume_value": None,
                "flow_status": FlowStatus.UPDATING.value,
                "events": [],  # already emitted via queue
            },
            goto="updater_node",
        )

    async def updater_node(
        state: MainGraphState, config: RunnableConfig,
    ) -> Command[Literal["executor_node", "__end__"]]:
        """Update the plan after step execution — mark step done, call planner
        to update remaining steps with execution context, then get next step.

        This mirrors the main-branch PlannerReActFlow UPDATING phase:
        1. Sync completed step back into plan
        2. Call planner LLM with UPDATE_PLAN_PROMPT (execution_summary)
        3. Replace pending steps with planner's updated steps
        4. Emit PlanEvent(UPDATED)
        """
        from app.domain.services.execution_watchdog import _should_terminate

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        # D5: Cooperative termination check
        if _should_terminate(config):
            logger.info("updater_node: should_terminate=True, routing to END")
            return Command(
                update={
                    "flow_status": FlowStatus.COMPLETED.value,
                    "events": [],
                },
                goto=END,
            )

        plan = state["plan"]
        if not plan:
            return Command(
                update={"flow_status": FlowStatus.COMPLETED.value, "events": []},
                goto=END,
            )

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

                update_messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=query),
                ]
                structured_llm = planner_llm.with_structured_output(PlanUpdateResponse)
                parsed_obj: PlanUpdateResponse = await structured_llm.ainvoke(update_messages)

                if parsed_obj and parsed_obj.steps:
                    new_steps = [
                        Step(
                            description=s.description,
                            id=s.id,
                        )
                        for s in parsed_obj.steps
                        if s.description
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

        # Phase 3: 根据下一步描述刷新 skill context
        new_skill_context = state.get("skill_context", "")
        if next_step:
            refresher = (config.get("configurable") or {}).get("skill_context_refresher")
            if refresher:
                try:
                    new_skill_context = await refresher(next_step.description)
                except Exception as exc:
                    logger.warning("Step-level skill refresh 失败: %s", exc)

        if not next_step:
            plan.status = ExecutionStatus.COMPLETED
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED))
            if event_queue is not None:
                for evt in events:
                    await event_queue.put(evt)
            return Command(
                update={
                    "plan": plan,
                    "current_step": None,
                    "flow_status": FlowStatus.COMPLETED.value,
                    "skill_context": new_skill_context,
                    "events": [],  # already emitted via queue
                },
                goto=END,
            )

        return Command(
            update={
                "plan": plan,
                "current_step": next_step,
                "flow_status": FlowStatus.EXECUTING.value,
                "skill_context": new_skill_context,
                "events": events,
            },
            goto="executor_node",
        )

    async def summarizer_node(state: MainGraphState, config: RunnableConfig) -> dict:
        """Generate final summary with streaming and emit completion events.

        Streams partial MessageEvent chunks via event_queue for real-time
        frontend updates. The final MessageEvent carries partial=False,
        the same stream_id, and parsed attachments.
        """
        from app.domain.services.prompts.react import SUMMARIZE_PROMPT

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        async def _emit(evt: Any) -> None:
            if event_queue is not None:
                await event_queue.put(evt)

        plan = state["plan"]
        events: list = []

        if plan:
            # 通过 model_copy 避免直接变异 state 对象
            plan = plan.model_copy(update={"status": ExecutionStatus.COMPLETED})
            plan_evt = PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED)
            events.append(plan_evt)
            await _emit(plan_evt)

        # Stream LLM summary to frontend in real-time
        react_messages: list[BaseMessage] = state.get("messages", [])
        stream_id = str(uuid.uuid4())

        if react_messages:
            try:
                # summary_llm is BaseChatModel — pass LangChain messages directly
                messages: list[BaseMessage] = list(react_messages) + [
                    HumanMessage(content=SUMMARIZE_PROMPT),
                ]

                # Stream with cumulative prefix for frontend
                chunks: list[str] = []
                async for chunk in summary_llm.astream(messages):
                    if chunk.content:
                        chunks.append(chunk.content)
                        cumulative_text = "".join(chunks)
                        await _emit(MessageEvent(
                            role="assistant",
                            message=cumulative_text,
                            stream_id=stream_id,
                            partial=True,
                        ))

                full_text = "".join(chunks)

                # Parse {message, attachments} JSON from LLM output
                summary_text = full_text
                summary_attachments: list[str] = []
                if full_text:
                    try:
                        parsed = SummarizerOutput.model_validate_json(full_text)
                        if parsed.text:
                            summary_text = parsed.text
                        summary_attachments = [
                            a for a in parsed.attachments
                            if isinstance(a, str) and a.strip()
                        ]
                    except (ValueError, Exception):
                        # LLM returned non-JSON or malformed JSON — use raw text
                        summary_text = full_text
                        summary_attachments = []

                # Build File attachments
                file_attachments: list[File] = []
                for path in summary_attachments:
                    filename = path.rsplit("/", 1)[-1]
                    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
                    file_attachments.append(File(
                        filename=filename,
                        filepath=path,
                        extension=ext,
                    ))

                # Final MessageEvent (partial=False, same stream_id, with attachments)
                final_msg_evt = MessageEvent(
                    role="assistant",
                    message=summary_text,
                    attachments=file_attachments,
                    stream_id=stream_id,
                    partial=False,
                )
                events.append(final_msg_evt)
                await _emit(final_msg_evt)
            except Exception as exc:
                logger.warning("Summarizer LLM call failed: %s", exc)

        done_evt = DoneEvent()
        events.append(done_evt)
        await _emit(done_evt)

        result: dict = {
            "flow_status": FlowStatus.COMPLETED.value,
            "events": [],  # events already emitted via queue
        }
        if plan:
            result["plan"] = plan
        return result

    async def interrupt_node(state: MainGraphState) -> dict:
        """Pause graph execution for user input via LangGraph native interrupt().

        This node is only reached when executor_node sets should_interrupt=True.
        On resume, interrupt() returns the user's response, which is stored in
        resume_value for executor_node to consume.
        """
        user_response = interrupt({
            "reason": "user_input_required",
            "step": state["current_step"].description if state.get("current_step") else "",
        })
        return {
            "resume_value": user_response,
            "should_interrupt": False,
        }

    # ---- Routing ------------------------------------------------------- #

    def route_entry(state: MainGraphState) -> Literal[
        "planner_node", "executor_node", "updater_node",
    ]:
        """Route from START based on flow_status."""
        status = state.get("flow_status", FlowStatus.IDLE.value)

        if status in (FlowStatus.IDLE.value, FlowStatus.PLANNING.value):
            return "planner_node"
        if status == FlowStatus.EXECUTING.value:
            return "executor_node"
        return "updater_node"

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
    g.add_node("interrupt_node", interrupt_node)

    g.add_conditional_edges(START, route_entry)
    g.add_edge("planner_node", "executor_node")
    # executor_node and updater_node use Command(goto=...) for routing —
    # no conditional edges needed. Command handles all outgoing transitions.
    g.add_edge("interrupt_node", "executor_node")

    return g.compile(checkpointer=checkpointer)
