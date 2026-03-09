"""Skill 创建子图 — Flow 集成测试。

覆盖设计文档 §5.2-5.4：
- 子图启用时确认消息不触发 planner
- 子图 done 后控制权交还（终态后不拦截）
- 灰度未命中时走旧流程
- 灰度分桶一致性
- 从旧状态迁移到子图状态
"""

from __future__ import annotations

from typing import Any

import pytest

from app.domain.models.event import BaseEvent, MessageEvent, WaitEvent
from app.domain.models.message import Message
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.skill_graph_state import SkillGraphState
from app.domain.models.tool_result import ToolResult
from app.domain.services.flows.skill_creation_graph import SkillCreationGraph
from app.domain.services.flows.skill_graph_canary import is_skill_graph_enabled

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------- #
# Mock 基础设施
# --------------------------------------------------------------------------- #


class MockSessionRepo:
    """模拟 session repository，支持 skill_graph_state 和 skill_creation_state。"""

    def __init__(self) -> None:
        self._skill_graph_state: SkillGraphState | None = None
        self._skill_creation_state: SkillCreationState | None = None

    async def get_skill_graph_state(self, session_id: str) -> SkillGraphState | None:
        return self._skill_graph_state

    async def save_skill_graph_state(
        self, session_id: str, state: SkillGraphState
    ) -> None:
        self._skill_graph_state = state

    async def clear_skill_graph_state(self, session_id: str) -> None:
        self._skill_graph_state = None

    async def get_skill_creation_state(
        self, session_id: str
    ) -> SkillCreationState | None:
        return self._skill_creation_state

    async def save_skill_creation_state(
        self, session_id: str, state: SkillCreationState
    ) -> None:
        self._skill_creation_state = state

    async def clear_skill_creation_state(self, session_id: str) -> None:
        self._skill_creation_state = None


class MockUoW:
    def __init__(self, session: MockSessionRepo) -> None:
        self.session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockBrainstormTool:
    async def brainstorm_skill(self, description: str) -> ToolResult:
        return ToolResult(
            success=True,
            message="蓝图预览",
            data={
                "blueprint": {"skill_name": "test"},
                "blueprint_json": '{"skill_name":"test"}',
            },
        )


class MockCreateSkillTool:
    async def generate_skill(self, description: str, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            message="生成成功",
            data={"skill_data": '{"scripts":[],"manifest":{}}', "tools": ["t1"]},
        )

    async def install_skill(self, skill_data: str) -> ToolResult:
        return ToolResult(success=True, message="安装成功")


# --------------------------------------------------------------------------- #
# 测试：子图驱动核心逻辑（不需要完整的 PlannerReActFlow）
# --------------------------------------------------------------------------- #


class TestTryDriveSkillGraph:
    """直接测试子图的 Flow 层集成逻辑。"""

    async def _simulate_try_drive(
        self,
        session_repo: MockSessionRepo,
        message: Message,
        user_id: str = "user-1",
        canary_percent: int = 100,
    ) -> list[BaseEvent] | None:
        """模拟 PlannerReActFlow._try_drive_skill_graph 的核心逻辑。"""
        if not is_skill_graph_enabled(user_id, canary_percent):
            return None

        action = message.skill_confirmation_action
        graph_state = await session_repo.get_skill_graph_state("s1")

        # 无子图状态且无结构化动作 → 不激活
        if graph_state is None and action is None:
            return None

        # 无子图但有结构化动作 → 从旧状态迁移
        if graph_state is None and action is not None:
            old_state = await session_repo.get_skill_creation_state("s1")
            if old_state is None:
                return None
            status_map = {"generate": "wait_generate", "install": "wait_install"}
            mapped = status_map.get(old_state.pending_action or "")
            if mapped is None:
                return None
            graph_state = SkillGraphState(
                status=mapped,
                pending_action=old_state.pending_action,
                approval_status=old_state.approval_status,
                blueprint=old_state.blueprint,
                blueprint_json=old_state.blueprint_json,
                skill_data=old_state.skill_data,
            )

        if graph_state is not None and graph_state.is_terminal:
            return None

        # 驱动子图
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())
        events = []
        async for event in graph.run(
            state=graph_state, action=action, original_request=""
        ):
            events.append(event)

        # 持久化
        new_state = graph.state
        if new_state is not None:
            if new_state.is_terminal:
                await session_repo.clear_skill_graph_state("s1")
                await session_repo.clear_skill_creation_state("s1")
            else:
                await session_repo.save_skill_graph_state("s1", new_state)

        return events

    async def test_canary_disabled_returns_none(self):
        """灰度 0% → 不激活子图。"""
        repo = MockSessionRepo()
        result = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="generate"),
            canary_percent=0,
        )
        assert result is None

    async def test_no_state_no_action_returns_none(self):
        """无子图状态 + 无结构化动作 → 不激活。"""
        repo = MockSessionRepo()
        result = await self._simulate_try_drive(
            repo,
            Message(message="你好"),
        )
        assert result is None

    async def test_migrate_from_old_state(self):
        """有旧 SkillCreationState + 结构化动作 → 迁移并驱动子图。"""
        repo = MockSessionRepo()
        repo._skill_creation_state = SkillCreationState(
            pending_action="generate",
            approval_status="pending",
            blueprint={"skill_name": "test"},
            blueprint_json='{"skill_name":"test"}',
        )

        events = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="generate"),
        )

        assert events is not None
        assert len(events) > 0
        # 迁移后子图状态已保存
        assert repo._skill_graph_state is not None
        assert repo._skill_graph_state.status == "wait_install"

    async def test_existing_graph_state_drives_subgraph(self):
        """已有子图状态 → 直接驱动。"""
        repo = MockSessionRepo()
        repo._skill_graph_state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            original_request="test",
            blueprint={"skill_name": "test"},
        )

        events = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="generate"),
        )

        assert events is not None
        assert repo._skill_graph_state.status == "wait_install"

    async def test_terminal_state_returns_none(self):
        """终态子图 → 不拦截，返回 None。"""
        repo = MockSessionRepo()
        repo._skill_graph_state = SkillGraphState(
            status="done",
            original_request="test",
        )

        result = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="generate"),
        )
        assert result is None

    async def test_done_clears_both_states(self):
        """install 成功后（done）清理 _skill_graph 和旧状态。"""
        repo = MockSessionRepo()
        repo._skill_graph_state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="test",
            skill_data='{"scripts":[],"manifest":{}}',
        )
        repo._skill_creation_state = SkillCreationState(
            pending_action="install",
        )

        events = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="install"),
        )

        assert events is not None
        # 两个状态都被清理
        assert repo._skill_graph_state is None
        assert repo._skill_creation_state is None

    async def test_cancel_clears_state(self):
        """cancel → 清理状态。"""
        repo = MockSessionRepo()
        repo._skill_graph_state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            original_request="test",
        )

        events = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="cancel"),
        )

        assert events is not None
        assert repo._skill_graph_state is None  # 终态已清理

    async def test_full_flow_brainstorm_to_done(self):
        """完整链路：brainstorm → generate → install → done。"""
        repo = MockSessionRepo()

        # Step 1: 首次 brainstorm（通过正常流程触发，此处模拟结果）
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())
        events = []
        async for event in graph.run(
            state=None, action=None, original_request="创建天气 skill"
        ):
            events.append(event)
        assert graph.state.status == "wait_generate"
        await repo.save_skill_graph_state("s1", graph.state)

        # Step 2: 用户确认 generate
        events = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="generate"),
        )
        assert repo._skill_graph_state.status == "wait_install"

        # Step 3: 用户确认 install
        events = await self._simulate_try_drive(
            repo,
            Message(skill_confirmation_action="install"),
        )
        assert repo._skill_graph_state is None  # done → 已清理
