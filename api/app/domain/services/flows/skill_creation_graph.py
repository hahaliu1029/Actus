"""Skill 创建子图 — 基于 LangGraph StateGraph 的显式状态机。

将 Skill 创建链路（brainstorm → generate → install）固化为图状态机，
以"图状态决定下一步"替代"LLM 自行决定下一步"，消除确认后回退循环的风险。

设计文档: docs/plans/2026-03-09-langgraph-skill-creation-subgraph-design.md
"""

from __future__ import annotations

import logging
import operator
from datetime import datetime
from typing import Annotated, Any, AsyncGenerator, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain.models.event import BaseEvent, MessageEvent, WaitEvent
from app.domain.models.message import SkillConfirmationAction
from app.domain.models.skill_graph_state import SkillGraphState
from app.domain.models.tool_result import ToolResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LangGraph 内部状态
# --------------------------------------------------------------------------- #


class _InternalState(TypedDict):
    """LangGraph 图的内部运行状态。

    与 SkillGraphState 字段一一对应，额外增加 action（本次用户动作）
    和 events（累积产出的事件）两个瞬态字段。
    """

    status: str
    pending_action: str  # "" 表示 None
    approval_status: str  # "" 表示 None
    original_request: str
    blueprint: Any  # dict | None
    blueprint_json: str
    skill_data: str
    last_tool_call_id: str
    saved_tool_result_json: str
    last_error: str
    retry_count: int
    # 瞬态字段
    action: str  # 当前用户动作（仅本次调用有效）
    events: Annotated[list, operator.add]  # 跨节点累积产出的事件


# --------------------------------------------------------------------------- #
# 纯函数节点（不依赖外部工具）
# --------------------------------------------------------------------------- #


def _cancel_node(_state: _InternalState) -> dict:
    """取消 Skill 创建。"""
    return {
        "status": "cancelled",
        "pending_action": "",
        "events": [MessageEvent(role="assistant", message="Skill 创建已取消。")],
    }


def _reentry_guard(_state: _InternalState) -> dict:
    """防重入：正在执行中，拒绝新请求。"""
    return {
        "events": [
            MessageEvent(role="assistant", message="当前操作正在执行中，请稍候。")
        ],
    }


def _prompt_wait_generate(_state: _InternalState) -> dict:
    """wait_generate 状态下收到未知动作，提示用户选择。"""
    return {
        "events": [
            MessageEvent(
                role="assistant",
                message="当前等待确认蓝图。请选择：确认生成、修改蓝图或取消。",
            ),
            WaitEvent(pending_action="generate"),
        ],
    }


def _prompt_wait_install(_state: _InternalState) -> dict:
    """wait_install 状态下收到未知动作，提示用户选择。"""
    return {
        "events": [
            MessageEvent(
                role="assistant",
                message="当前等待确认安装。请选择：确认安装、重新生成、修改蓝图或取消。",
            ),
            WaitEvent(pending_action="install"),
        ],
    }


def _prompt_error(state: _InternalState) -> dict:
    """error 状态下提示用户可选操作。"""
    hint = "已达重试上限，" if state["retry_count"] >= 2 else ""
    return {
        "events": [
            MessageEvent(
                role="assistant",
                message=(
                    f"Skill 创建遇到错误：{state['last_error']}\n"
                    f"{hint}请选择：修改蓝图或取消。"
                ),
            ),
        ],
    }


def _retry_node(state: _InternalState) -> dict:
    """重试前置：递增计数、清除错误，随后路由到对应执行节点。"""
    return {
        "retry_count": state["retry_count"] + 1,
        "last_error": "",
    }


def _noop(_state: _InternalState) -> dict:
    """终态或未知状态，不做任何操作。"""
    return {}


# --------------------------------------------------------------------------- #
# 路由函数
# --------------------------------------------------------------------------- #


def _route_entry(state: _InternalState) -> str:
    """入口路由：根据当前 status + action 决定下一步执行的节点。"""
    status = state["status"]
    action = state["action"]

    if status == "init":
        return "blueprint_node"

    if status == "wait_generate":
        if action == "generate":
            return "generate_node"
        if action == "revise":
            return "blueprint_node"
        if action == "cancel":
            return "cancel_node"
        return "prompt_wait_generate"

    if status == "wait_install":
        if action == "install":
            return "install_node"
        if action == "regenerate":
            return "generate_node"
        if action == "revise":
            return "blueprint_node"
        if action == "cancel":
            return "cancel_node"
        return "prompt_wait_install"

    if status == "error":
        if action == "retry" and state["retry_count"] < 2:
            return "retry_node"
        if action == "revise":
            return "blueprint_node"
        if action == "cancel":
            return "cancel_node"
        return "prompt_error"

    if status in ("generating", "installing"):
        return "reentry_guard"

    return "noop"


def _route_retry(state: _InternalState) -> str:
    """retry_node 后续路由：根据 pending_action 决定重试 generate 还是 install。"""
    if state["pending_action"] == "install":
        return "install_node"
    return "generate_node"


# --------------------------------------------------------------------------- #
# SkillCreationGraph
# --------------------------------------------------------------------------- #


class SkillCreationGraph:
    """Skill 创建子图，基于 LangGraph StateGraph 实现。

    使用方式：
        graph = SkillCreationGraph(brainstorm_tool, create_skill_tool)
        async for event in graph.run(state, action, original_request):
            yield event  # BaseEvent
        # graph.state 包含更新后的 SkillGraphState
    """

    def __init__(
        self,
        brainstorm_tool: Any,
        create_skill_tool: Any,
    ) -> None:
        self._brainstorm_tool = brainstorm_tool
        self._create_skill_tool = create_skill_tool
        self._state: SkillGraphState | None = None
        self._compiled = self._build_graph()

    @property
    def state(self) -> SkillGraphState | None:
        """执行后的最新状态。"""
        return self._state

    # ----- 图构建 ---------------------------------------------------------- #

    def _build_graph(self) -> Any:
        """构建并编译 LangGraph StateGraph。"""
        g: StateGraph = StateGraph(_InternalState)

        # 执行节点（需要工具引用，使用绑定方法）
        g.add_node("blueprint_node", self._blueprint_node)
        g.add_node("generate_node", self._generate_node)
        g.add_node("install_node", self._install_node)

        # 纯函数节点
        g.add_node("cancel_node", _cancel_node)
        g.add_node("retry_node", _retry_node)
        g.add_node("reentry_guard", _reentry_guard)
        g.add_node("prompt_wait_generate", _prompt_wait_generate)
        g.add_node("prompt_wait_install", _prompt_wait_install)
        g.add_node("prompt_error", _prompt_error)
        g.add_node("noop", _noop)

        # 入口路由：START → 根据 status+action 分发
        g.add_conditional_edges(START, _route_entry)

        # retry_node → 根据 pending_action 路由到 generate_node 或 install_node
        g.add_conditional_edges("retry_node", _route_retry)

        # 所有终端节点连接到 END
        for node_name in [
            "blueprint_node",
            "generate_node",
            "install_node",
            "cancel_node",
            "reentry_guard",
            "prompt_wait_generate",
            "prompt_wait_install",
            "prompt_error",
            "noop",
        ]:
            g.add_edge(node_name, END)

        return g.compile()

    # ----- 执行节点（异步，依赖工具） ---------------------------------------- #

    async def _blueprint_node(self, state: _InternalState) -> dict:
        """blueprint_node: 调用 brainstorm_skill 生成蓝图。"""
        result: ToolResult = await self._brainstorm_tool.brainstorm_skill(
            description=state["original_request"],
        )

        if result.success:
            data = result.data or {}
            preview = result.message or ""
            return {
                "blueprint": data.get("blueprint"),
                "blueprint_json": data.get("blueprint_json", ""),
                "status": "wait_generate",
                "pending_action": "generate",
                "approval_status": "pending",
                "last_tool_call_id": "",
                "saved_tool_result_json": result.model_dump_json(),
                "skill_data": "",
                "last_error": "",
                "retry_count": 0,
                "events": [
                    MessageEvent(
                        role="assistant",
                        message=f"{preview}\n\n请确认蓝图是否符合预期。",
                    ),
                    WaitEvent(pending_action="generate"),
                ],
            }

        return {
            "status": "error",
            "last_error": result.message or "蓝图生成失败",
            "pending_action": "",
            "events": [
                MessageEvent(
                    role="assistant",
                    message=f"蓝图生成失败：{result.message}",
                ),
            ],
        }

    async def _generate_node(self, state: _InternalState) -> dict:
        """generate_node: 调用 generate_skill 生成代码。"""
        kwargs: dict[str, Any] = {
            "description": state["original_request"],
        }
        if state["blueprint"]:
            kwargs["blueprint"] = state["blueprint"]
        elif state["blueprint_json"]:
            kwargs["blueprint_json"] = state["blueprint_json"]

        result: ToolResult = await self._create_skill_tool.generate_skill(**kwargs)

        if result.success:
            data = result.data or {}
            return {
                "skill_data": data.get("skill_data", ""),
                "status": "wait_install",
                "pending_action": "install",
                "approval_status": "pending",
                "last_error": "",
                "saved_tool_result_json": result.model_dump_json(),
                "events": [
                    MessageEvent(
                        role="assistant",
                        message="Skill 代码生成并验证通过，是否确认安装？",
                    ),
                    WaitEvent(pending_action="install"),
                ],
            }

        return {
            "status": "error",
            "last_error": result.message or "代码生成失败",
            "pending_action": "generate",
            "events": [
                MessageEvent(
                    role="assistant",
                    message=f"Skill 生成失败：{result.message}\n可选择重试、修改蓝图或取消。",
                ),
            ],
        }

    async def _install_node(self, state: _InternalState) -> dict:
        """install_node: 调用 install_skill 安装已生成的 Skill。"""
        if not state["skill_data"]:
            return {
                "status": "error",
                "last_error": "缺少 skill_data，请重新生成",
                "pending_action": "generate",
                "events": [
                    MessageEvent(
                        role="assistant",
                        message="安装失败：缺少生成数据，请重新生成 Skill。",
                    ),
                ],
            }

        result: ToolResult = await self._create_skill_tool.install_skill(
            skill_data=state["skill_data"],
        )

        if result.success:
            return {
                "status": "done",
                "pending_action": "",
                "approval_status": "",
                "events": [
                    MessageEvent(
                        role="assistant",
                        message=f"Skill 安装成功！{result.message or ''}",
                    ),
                ],
            }

        return {
            "status": "error",
            "last_error": result.message or "安装失败",
            "pending_action": "install",
            "events": [
                MessageEvent(
                    role="assistant",
                    message=f"Skill 安装失败：{result.message}\n可选择重试、修改蓝图或取消。",
                ),
            ],
        }

    # ----- 状态转换 -------------------------------------------------------- #

    def _to_internal(
        self,
        state: SkillGraphState | None,
        action: str,
        original_request: str,
    ) -> _InternalState:
        """将外部 SkillGraphState 转换为 LangGraph 内部状态。"""
        if state is None:
            return _InternalState(
                status="init",
                pending_action="",
                approval_status="",
                original_request=original_request,
                blueprint=None,
                blueprint_json="",
                skill_data="",
                last_tool_call_id="",
                saved_tool_result_json="",
                last_error="",
                retry_count=0,
                action=action,
                events=[],
            )

        return _InternalState(
            status=state.status,
            pending_action=state.pending_action or "",
            approval_status=state.approval_status or "",
            original_request=state.original_request,
            blueprint=state.blueprint,
            blueprint_json=state.blueprint_json,
            skill_data=state.skill_data,
            last_tool_call_id=state.last_tool_call_id,
            saved_tool_result_json=state.saved_tool_result_json,
            last_error=state.last_error,
            retry_count=state.retry_count,
            action=action,
            events=[],
        )

    def _to_external(self, result: dict) -> SkillGraphState:
        """将 LangGraph 执行结果转换回 SkillGraphState。"""
        return SkillGraphState(
            status=result["status"],
            pending_action=result["pending_action"] or None,
            approval_status=result["approval_status"] or None,
            original_request=result["original_request"],
            blueprint=result["blueprint"],
            blueprint_json=result["blueprint_json"],
            skill_data=result["skill_data"],
            last_tool_call_id=result["last_tool_call_id"],
            saved_tool_result_json=result["saved_tool_result_json"],
            last_error=result["last_error"],
            retry_count=result["retry_count"],
            updated_at=datetime.now(),
        )

    # ----- 公开接口 -------------------------------------------------------- #

    async def run(
        self,
        state: SkillGraphState | None,
        action: SkillConfirmationAction | None,
        original_request: str = "",
    ) -> AsyncGenerator[BaseEvent, None]:
        """驱动子图前进一步，返回产出的事件流。

        Parameters
        ----------
        state : 当前持久化的 SkillGraphState（首次为 None）。
        action : 用户的结构化确认动作。
        original_request : 用户原始需求（仅首次需要）。
        """
        action_str = (action or "").strip().lower()
        internal = self._to_internal(state, action_str, original_request)

        result = await self._compiled.ainvoke(internal)

        self._state = self._to_external(result)

        for event in result.get("events", []):
            yield event
