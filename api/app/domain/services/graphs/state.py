"""LangGraph state models for main_graph and react_graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from app.domain.models.plan import Plan, Step
from app.domain.services.flows.base import FlowStatus


class MainGraphState(TypedDict):
    """State for the outer orchestration graph (planner → executor → updater → summarizer).

    ``messages`` stores LangChain BaseMessage objects (overwrite semantics — each
    executor_node returns the full compacted conversation so far).
    """

    # Input
    message: str
    language: str
    attachments: list[str]
    image_content_blocks: list[dict]  # 图片附件的多模态内容块 (OpenAI image_url format)

    # Planning
    plan: Plan | None
    current_step: Step | None

    # Execution — LangChain BaseMessage list (overwrite semantics, not append).
    # 不使用 add_messages reducer：executor_node 需要通过 _compact_messages() 修改已有消息内容，
    # add_messages 的 ID 去重机制会导致压缩后的消息被当作新消息追加而非替换。
    # 且只有 executor_node 写此字段，不存在多节点拼接遗漏的风险。
    messages: list[BaseMessage]
    execution_summary: str

    # Events produced by nodes (accumulated across nodes)
    events: Annotated[list, operator.add]

    # Control
    flow_status: str  # FlowStatus.value — typed routing via FlowStatus enum
    session_id: str
    should_interrupt: bool
    resume_value: str | None  # set by interrupt_node on resume; None otherwise

    # Context
    original_request: str
    skill_context: str
    conversation_summaries: list[str]  # 历史对话摘要文本（to_prompt_text() 输出）

    # B5 C5a: per-step prompt assembly metadata (optional — older checkpoints
    # won't have these; ``state.get(key)`` returns None for missing keys).
    # - ``skill_names_in_context``: authoritative skill ids that produced the
    #   current step's skill_context (written by executor_node or falls back
    #   to updater_node's skill_context_refresher).
    # - ``system_prompt_version_hash`` / ``system_prompt_tokens``: observability
    #   only, written by executor_node via the PromptAssembler path.
    #   See B5 design doc "Two-clock architecture".
    skill_names_in_context: NotRequired[list[str]]
    system_prompt_version_hash: NotRequired[str]
    system_prompt_tokens: NotRequired[int]


class ReactGraphState(TypedDict):
    """State for the inner ReAct loop graph (LLM → tool → LLM → ...).

    ``messages`` uses ``add_messages`` reducer — nodes return only new messages
    and the reducer appends them to the accumulated list.
    """

    # Conversation — LangChain BaseMessage list (append via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Context assembler output — trimmed messages for LLM input.
    # No reducer (overwrite semantics). Written by pre_llm_node, read by llm_node.
    # react_graph has no checkpointer, so this is not persisted within the loop.
    llm_input_messages: list[BaseMessage]

    # Step context
    step_description: str
    original_request: str
    language: str
    attachments: list[str]
    image_content_blocks: list[dict]  # 图片附件的多模态内容块

    # Events produced by nodes (accumulated)
    events: Annotated[list, operator.add]

    # Control
    should_interrupt: bool
    soft_hint_sent: bool  # True after first SOFT_HINT returned in this step (reset per step)

    # Gating counters (for message_ask_user soft-hint throttling)
    attempt_count: int
    failure_count: int

    # R2 CS2: 前缀闭合 exactly-once 机制
    # tool_node 每次 replay 时跳过这些 ids (prefix-closure). 正常 batch 结束时清空.
    # NotRequired 以兼容没有写这些字段的旧调用点 (invocation payload / checkpoint).
    completed_tool_call_prefix: NotRequired[list[str]]

    # R2 CS2: interrupt_helper → tool_node 的 "pre-approved" 桥接.
    # approve 路径下 interrupt_helper 把 pending_id 写进这个 list.
    # tool_node replay 时看到 tc.id 在这个 list 里, 直接 bypass Layer 1
    # policy chain 到 Layer 2. 使用完后清空.
    # 理由: _resume_tool_confirmation 的 ApprovalCache 写是 post-resume 顺序
    # (agent_service.py:313-395), tool_node replay 时 cache 还没写,
    # 必须靠 state flag 直接 bypass.
    approved_tool_call_ids: NotRequired[list[str]]

    # R2 CS2: Asked 路径的 pending 状态, 给 interrupt_helper 节点消费.
    # pending_ask_outcome: Asked.model_dump(mode="json") — Layer 3 写入.
    # pending_ask_tool_call_id: 待审批的 tool_call_id.
    # pending_ask_artifact: ToolArtifact.model_dump(mode="json", by_alias=True).
    # pending_ask_tool_args: 原始 tool_call.args — 审计路径需要, 但 ToolArtifact
    #   schema 不存 args (那是 call 壳的一部分不是 outcome), 所以单独存一份.
    #   deny/timeout_fallback 路径下 interrupt_helper 读回去交给 _translate_outcome,
    #   让产出的 ToolEvent.function_args 反映用户真实拒绝的调用参数, 不再丢.
    pending_ask_outcome: NotRequired[dict | None]
    pending_ask_tool_call_id: NotRequired[str | None]
    pending_ask_artifact: NotRequired[dict | None]
    pending_ask_tool_args: NotRequired[dict | None]
