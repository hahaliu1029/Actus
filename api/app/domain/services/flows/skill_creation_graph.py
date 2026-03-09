"""Skill 创建子图 — 基于 LangGraph StateGraph 的显式状态机。

将 Skill 创建链路（brainstorm → generate → install）固化为图状态机，
以"图状态决定下一步"替代"LLM 自行决定下一步"，消除确认后回退循环的风险。

设计文档: docs/plans/2026-03-09-langgraph-skill-creation-subgraph-design.md
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, AsyncGenerator

from app.domain.models.event import BaseEvent, MessageEvent, WaitEvent
from app.domain.models.message import SkillConfirmationAction
from app.domain.models.skill_graph_state import SkillGraphState
from app.domain.models.tool_result import ToolResult

logger = logging.getLogger(__name__)

class SkillCreationGraph:
    """Skill 创建子图，封装 brainstorm → generate → install 的完整链路。

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
        self._pending_events: list[BaseEvent] = []

    @property
    def state(self) -> SkillGraphState | None:
        """执行后的最新状态。"""
        return self._state

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
        self._pending_events = []

        if state is None:
            # 首次进入：从 blueprint_node 开始
            state = SkillGraphState(
                status="wait_generate",
                original_request=original_request,
            )
            await self._execute_blueprint_node(state, original_request)
            self._state = state
            for event in self._pending_events:
                yield event
            return

        # 根据当前状态 + 动作决定下一步
        action_str = (action or "").strip().lower()

        if state.status == "wait_generate":
            async for event in self._handle_wait_generate(state, action_str):
                yield event
        elif state.status == "wait_install":
            async for event in self._handle_wait_install(state, action_str):
                yield event
        elif state.status == "error":
            async for event in self._handle_error(state, action_str):
                yield event
        elif state.is_executing:
            # 防重入：正在执行中，拒绝新请求
            yield MessageEvent(
                role="assistant",
                message="当前操作正在执行中，请稍候。",
            )
            self._state = state
        else:
            # 终态或未知状态，不处理
            self._state = state

    # ----- 状态处理器 ------------------------------------------------------ #

    async def _handle_wait_generate(
        self,
        state: SkillGraphState,
        action: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """处理 wait_generate 状态的转移。"""
        if action == "generate":
            state.approval_status = "approved"
            await self._execute_generate_node(state)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "revise":
            await self._execute_blueprint_node(state, state.original_request)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "cancel":
            state.status = "cancelled"
            state.pending_action = None
            state.updated_at = datetime.now()
            self._state = state
            yield MessageEvent(
                role="assistant",
                message="Skill 创建已取消。",
            )
        else:
            self._state = state
            yield MessageEvent(
                role="assistant",
                message="当前等待确认蓝图。请选择：确认生成、修改蓝图或取消。",
            )
            yield WaitEvent(pending_action="generate")

    async def _handle_wait_install(
        self,
        state: SkillGraphState,
        action: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """处理 wait_install 状态的转移。"""
        if action == "install":
            state.approval_status = "approved"
            await self._execute_install_node(state)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "regenerate":
            # 仅重新生成代码，保留蓝图
            await self._execute_generate_node(state)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "revise":
            # 回到蓝图重新设计
            state.skill_data = ""
            await self._execute_blueprint_node(state, state.original_request)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "cancel":
            state.status = "cancelled"
            state.pending_action = None
            state.updated_at = datetime.now()
            self._state = state
            yield MessageEvent(
                role="assistant",
                message="Skill 创建已取消。",
            )
        else:
            self._state = state
            yield MessageEvent(
                role="assistant",
                message="当前等待确认安装。请选择：确认安装、重新生成、修改蓝图或取消。",
            )
            yield WaitEvent(pending_action="install")

    async def _handle_error(
        self,
        state: SkillGraphState,
        action: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """处理 error 状态的转移。"""
        if action == "retry" and state.retry_count < 2:
            state.retry_count += 1
            state.last_error = ""
            # 根据 pending_action 决定重试哪个节点
            if state.pending_action == "install":
                await self._execute_install_node(state)
            else:
                await self._execute_generate_node(state)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "revise":
            state.last_error = ""
            state.retry_count = 0
            state.skill_data = ""
            await self._execute_blueprint_node(state, state.original_request)
            self._state = state
            for event in self._pending_events:
                yield event
        elif action == "cancel":
            state.status = "cancelled"
            state.pending_action = None
            state.updated_at = datetime.now()
            self._state = state
            yield MessageEvent(
                role="assistant",
                message="Skill 创建已取消。",
            )
        else:
            self._state = state
            hint = "已达重试上限，" if state.retry_count >= 2 else ""
            yield MessageEvent(
                role="assistant",
                message=f"Skill 创建遇到错误：{state.last_error}\n{hint}请选择：修改蓝图或取消。",
            )

    # ----- 节点实现 -------------------------------------------------------- #

    async def _execute_blueprint_node(
        self,
        state: SkillGraphState,
        description: str,
    ) -> None:
        """blueprint_node: 调用 brainstorm_skill 生成蓝图。"""
        self._pending_events = []

        result: ToolResult = await self._brainstorm_tool.brainstorm_skill(
            description=description,
        )

        if result.success:
            data = result.data or {}
            state.blueprint = data.get("blueprint")
            state.blueprint_json = data.get("blueprint_json", "")
            state.status = "wait_generate"
            state.pending_action = "generate"
            state.approval_status = "pending"
            state.last_tool_call_id = ""
            state.saved_tool_result_json = result.model_dump_json()
            state.updated_at = datetime.now()

            preview = result.message or ""
            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message=f"{preview}\n\n请确认蓝图是否符合预期。",
                )
            )
            self._pending_events.append(WaitEvent(pending_action="generate"))
        else:
            state.status = "error"
            state.last_error = result.message or "蓝图生成失败"
            state.pending_action = None
            state.updated_at = datetime.now()

            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message=f"蓝图生成失败：{result.message}",
                )
            )

    async def _execute_generate_node(self, state: SkillGraphState) -> None:
        """generate_node: 调用 generate_skill 生成代码（含自动修复重试）。"""
        self._pending_events = []
        state.status = "generating"
        state.updated_at = datetime.now()

        # 构建参数：优先使用 blueprint，否则用 blueprint_json
        kwargs: dict[str, Any] = {
            "description": state.original_request,
        }
        if state.blueprint:
            kwargs["blueprint"] = state.blueprint
        elif state.blueprint_json:
            kwargs["blueprint_json"] = state.blueprint_json

        result: ToolResult = await self._create_skill_tool.generate_skill(**kwargs)

        if result.success:
            data = result.data or {}
            state.skill_data = data.get("skill_data", "")
            state.status = "wait_install"
            state.pending_action = "install"
            state.approval_status = "pending"
            state.last_error = ""
            state.saved_tool_result_json = result.model_dump_json()
            state.updated_at = datetime.now()

            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message="Skill 代码生成并验证通过，是否确认安装？",
                )
            )
            self._pending_events.append(WaitEvent(pending_action="install"))
        else:
            state.status = "error"
            state.last_error = result.message or "代码生成失败"
            state.pending_action = "generate"
            state.updated_at = datetime.now()

            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message=f"Skill 生成失败：{result.message}\n可选择重试、修改蓝图或取消。",
                )
            )

    async def _execute_install_node(self, state: SkillGraphState) -> None:
        """install_node: 调用 install_skill 安装已生成的 Skill。"""
        self._pending_events = []
        state.status = "installing"
        state.updated_at = datetime.now()

        if not state.skill_data:
            state.status = "error"
            state.last_error = "缺少 skill_data，请重新生成"
            state.pending_action = "generate"
            state.updated_at = datetime.now()
            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message="安装失败：缺少生成数据，请重新生成 Skill。",
                )
            )
            return

        result: ToolResult = await self._create_skill_tool.install_skill(
            skill_data=state.skill_data,
        )

        if result.success:
            state.status = "done"
            state.pending_action = None
            state.approval_status = None
            state.updated_at = datetime.now()

            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message=f"Skill 安装成功！{result.message or ''}",
                )
            )
        else:
            state.status = "error"
            state.last_error = result.message or "安装失败"
            state.pending_action = "install"
            state.updated_at = datetime.now()

            self._pending_events.append(
                MessageEvent(
                    role="assistant",
                    message=f"Skill 安装失败：{result.message}\n可选择重试、修改蓝图或取消。",
                )
            )
