"""Skill 创建子图（SkillCreationGraph）单元测试。

覆盖设计文档 §5.1 全部场景：
- 正常链路 blueprint → wait_generate → generate → wait_install → install → done
- wait_generate + revise 回蓝图
- wait_install + revise 回蓝图
- wait_install + regenerate 仅重新生成（保留蓝图）
- 非结构化输入不推进状态
- error + retry 重试
- error + retry 超出上限
- error + revise 回蓝图
- generate_node 失败进入 error
- install_node 失败进入 error
- cancel 取消
- 防重入（executing 状态拒绝新请求）
"""

from __future__ import annotations

from typing import Any

import pytest

from app.domain.models.event import BaseEvent, MessageEvent, WaitEvent
from app.domain.models.skill_graph_state import SkillGraphState
from app.domain.models.tool_result import ToolResult
from app.domain.services.flows.skill_creation_graph import SkillCreationGraph

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------- #
# Mock 工具
# --------------------------------------------------------------------------- #


class MockBrainstormTool:
    """模拟 brainstorm_skill 工具。"""

    def __init__(self, success: bool = True):
        self._success = success
        self.call_count = 0

    async def brainstorm_skill(self, description: str) -> ToolResult:
        self.call_count += 1
        if self._success:
            return ToolResult(
                success=True,
                message="Skill 蓝图预览\n\n名称: test-skill",
                data={
                    "blueprint": {"skill_name": "test-skill", "description": description},
                    "blueprint_json": '{"skill_name":"test-skill"}',
                },
            )
        return ToolResult(success=False, message="蓝图生成失败: LLM error")


class MockCreateSkillTool:
    """模拟 generate_skill 和 install_skill 工具。"""

    def __init__(
        self,
        generate_success: bool = True,
        install_success: bool = True,
    ):
        self._generate_success = generate_success
        self._install_success = install_success
        self.generate_call_count = 0
        self.install_call_count = 0

    async def generate_skill(self, description: str, **kwargs) -> ToolResult:
        self.generate_call_count += 1
        if self._generate_success:
            return ToolResult(
                success=True,
                message="Skill 生成并验证通过",
                data={
                    "skill_data": '{"scripts":[],"manifest":{}}',
                    "tools": ["tool_a"],
                },
            )
        return ToolResult(success=False, message="生成失败: sandbox error")

    async def install_skill(self, skill_data: str) -> ToolResult:
        self.install_call_count += 1
        if self._install_success:
            return ToolResult(
                success=True,
                message="已安装 test-skill (1 tools, 2 files)",
            )
        return ToolResult(success=False, message="安装失败: disk error")


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #


async def collect_events(gen) -> list[BaseEvent]:
    events = []
    async for event in gen:
        events.append(event)
    return events


def find_event(events: list[BaseEvent], event_type: type) -> BaseEvent | None:
    for e in events:
        if isinstance(e, event_type):
            return e
    return None


# --------------------------------------------------------------------------- #
# 测试用例
# --------------------------------------------------------------------------- #


class TestNormalFlow:
    """正常链路：blueprint → wait_generate → generate → wait_install → install → done"""

    async def test_initial_brainstorm(self):
        """首次进入：执行 blueprint_node，进入 wait_generate。"""
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())
        events = await collect_events(
            graph.run(state=None, action=None, original_request="创建一个天气查询 skill")
        )

        assert graph.state is not None
        assert graph.state.status == "wait_generate"
        assert graph.state.pending_action == "generate"
        assert graph.state.approval_status == "pending"
        assert graph.state.blueprint is not None

        wait = find_event(events, WaitEvent)
        assert wait is not None
        assert wait.pending_action == "generate"

    async def test_generate_after_confirm(self):
        """wait_generate + generate → 调用 generate_skill → wait_install。"""
        brainstorm = MockBrainstormTool()
        create = MockCreateSkillTool()
        graph = SkillCreationGraph(brainstorm, create)

        state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            approval_status="pending",
            original_request="天气 skill",
            blueprint={"skill_name": "test"},
            blueprint_json='{"skill_name":"test"}',
        )

        events = await collect_events(graph.run(state=state, action="generate"))

        assert graph.state.status == "wait_install"
        assert graph.state.pending_action == "install"
        assert graph.state.skill_data != ""
        assert create.generate_call_count == 1

        wait = find_event(events, WaitEvent)
        assert wait is not None
        assert wait.pending_action == "install"

    async def test_install_after_confirm(self):
        """wait_install + install → 调用 install_skill → done。"""
        create = MockCreateSkillTool()
        graph = SkillCreationGraph(MockBrainstormTool(), create)

        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            approval_status="pending",
            original_request="天气 skill",
            skill_data='{"scripts":[],"manifest":{}}',
        )

        events = await collect_events(graph.run(state=state, action="install"))

        assert graph.state.status == "done"
        assert graph.state.is_terminal
        assert create.install_call_count == 1


class TestRevise:
    """修改蓝图测试。"""

    async def test_wait_generate_revise(self):
        """wait_generate + revise → 重新调用 brainstorm_skill → 回到 wait_generate。"""
        brainstorm = MockBrainstormTool()
        graph = SkillCreationGraph(brainstorm, MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            original_request="天气 skill",
        )

        events = await collect_events(graph.run(state=state, action="revise"))

        assert graph.state.status == "wait_generate"
        assert brainstorm.call_count == 1

        wait = find_event(events, WaitEvent)
        assert wait is not None
        assert wait.pending_action == "generate"

    async def test_wait_install_revise(self):
        """wait_install + revise → 回到 blueprint_node → wait_generate。"""
        brainstorm = MockBrainstormTool()
        graph = SkillCreationGraph(brainstorm, MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="天气 skill",
            skill_data="some_data",
        )

        events = await collect_events(graph.run(state=state, action="revise"))

        assert graph.state.status == "wait_generate"
        assert brainstorm.call_count == 1


class TestRegenerate:
    """wait_install + regenerate：仅重新生成代码，保留蓝图。"""

    async def test_regenerate_keeps_blueprint(self):
        create = MockCreateSkillTool()
        graph = SkillCreationGraph(MockBrainstormTool(), create)

        original_blueprint = {"skill_name": "test", "tools": []}
        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="天气 skill",
            blueprint=original_blueprint,
            blueprint_json='{"skill_name":"test","tools":[]}',
            skill_data="old_data",
        )

        events = await collect_events(graph.run(state=state, action="regenerate"))

        assert graph.state.status == "wait_install"
        assert graph.state.blueprint == original_blueprint  # 蓝图保留
        assert create.generate_call_count == 1

        wait = find_event(events, WaitEvent)
        assert wait is not None
        assert wait.pending_action == "install"


class TestUnstructuredInput:
    """非结构化输入不推进状态。"""

    async def test_wait_generate_unknown_action(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            original_request="天气 skill",
        )

        events = await collect_events(graph.run(state=state, action=None))

        assert graph.state.status == "wait_generate"  # 未推进

        wait = find_event(events, WaitEvent)
        assert wait is not None

    async def test_wait_install_unknown_action(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="天气 skill",
            skill_data="data",
        )

        events = await collect_events(graph.run(state=state, action=None))

        assert graph.state.status == "wait_install"  # 未推进

        wait = find_event(events, WaitEvent)
        assert wait is not None


class TestErrorAndRetry:
    """错误状态和重试测试。"""

    async def test_generate_failure_enters_error(self):
        """generate_skill 失败 → status=error。"""
        create = MockCreateSkillTool(generate_success=False)
        graph = SkillCreationGraph(MockBrainstormTool(), create)

        state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            original_request="天气 skill",
            blueprint={"skill_name": "test"},
        )

        events = await collect_events(graph.run(state=state, action="generate"))

        assert graph.state.status == "error"
        assert "生成失败" in graph.state.last_error

    async def test_install_failure_enters_error(self):
        """install_skill 失败 → status=error。"""
        create = MockCreateSkillTool(install_success=False)
        graph = SkillCreationGraph(MockBrainstormTool(), create)

        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="天气 skill",
            skill_data='{"scripts":[],"manifest":{}}',
        )

        events = await collect_events(graph.run(state=state, action="install"))

        assert graph.state.status == "error"
        assert graph.state.pending_action == "install"

    async def test_error_retry_success(self):
        """error + retry → 重试成功 → 进入下一阶段。"""
        create = MockCreateSkillTool(generate_success=True)
        graph = SkillCreationGraph(MockBrainstormTool(), create)

        state = SkillGraphState(
            status="error",
            pending_action="generate",
            original_request="天气 skill",
            blueprint={"skill_name": "test"},
            last_error="之前失败了",
            retry_count=0,
        )

        events = await collect_events(graph.run(state=state, action="retry"))

        assert graph.state.status == "wait_install"
        assert graph.state.retry_count == 1

    async def test_error_retry_exceeds_limit(self):
        """error + retry 超出上限（retry_count >= 2）→ 保持 error。"""
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="error",
            pending_action="generate",
            original_request="天气 skill",
            last_error="反复失败",
            retry_count=2,  # 已达上限
        )

        events = await collect_events(graph.run(state=state, action="retry"))

        assert graph.state.status == "error"  # 保持 error
        assert graph.state.retry_count == 2  # 未增加

        msg = find_event(events, MessageEvent)
        assert msg is not None
        assert "重试上限" in msg.message

    async def test_error_revise(self):
        """error + revise → 回到 blueprint_node。"""
        brainstorm = MockBrainstormTool()
        graph = SkillCreationGraph(brainstorm, MockCreateSkillTool())

        state = SkillGraphState(
            status="error",
            pending_action="generate",
            original_request="天气 skill",
            last_error="失败了",
            retry_count=1,
        )

        events = await collect_events(graph.run(state=state, action="revise"))

        assert graph.state.status == "wait_generate"
        assert graph.state.retry_count == 0
        assert graph.state.last_error == ""
        assert brainstorm.call_count == 1


class TestCancel:
    """cancel 取消测试。"""

    async def test_cancel_from_wait_generate(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_generate",
            pending_action="generate",
            original_request="天气 skill",
        )

        events = await collect_events(graph.run(state=state, action="cancel"))

        assert graph.state.status == "cancelled"
        assert graph.state.is_terminal

    async def test_cancel_from_wait_install(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="天气 skill",
            skill_data="data",
        )

        events = await collect_events(graph.run(state=state, action="cancel"))

        assert graph.state.status == "cancelled"
        assert graph.state.is_terminal

    async def test_cancel_from_error(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="error",
            last_error="something broke",
            original_request="天气 skill",
        )

        events = await collect_events(graph.run(state=state, action="cancel"))

        assert graph.state.status == "cancelled"
        assert graph.state.is_terminal


class TestReentryGuard:
    """防重入：executing 状态拒绝新请求。"""

    async def test_generating_rejects_request(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="generating",
            original_request="天气 skill",
        )

        events = await collect_events(graph.run(state=state, action="generate"))

        assert graph.state.status == "generating"  # 未变化

        msg = find_event(events, MessageEvent)
        assert msg is not None
        assert "正在执行中" in msg.message

    async def test_installing_rejects_request(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="installing",
            original_request="天气 skill",
        )

        events = await collect_events(graph.run(state=state, action="install"))

        assert graph.state.status == "installing"  # 未变化


class TestBlueprintFailure:
    """brainstorm_skill 失败 → error。"""

    async def test_brainstorm_failure(self):
        brainstorm = MockBrainstormTool(success=False)
        graph = SkillCreationGraph(brainstorm, MockCreateSkillTool())

        events = await collect_events(
            graph.run(state=None, action=None, original_request="失败测试")
        )

        assert graph.state.status == "error"
        assert "蓝图生成失败" in graph.state.last_error


class TestMissingSkillData:
    """install_node 缺少 skill_data → error。"""

    async def test_install_missing_skill_data(self):
        graph = SkillCreationGraph(MockBrainstormTool(), MockCreateSkillTool())

        state = SkillGraphState(
            status="wait_install",
            pending_action="install",
            original_request="天气 skill",
            skill_data="",  # 空
        )

        events = await collect_events(graph.run(state=state, action="install"))

        assert graph.state.status == "error"
        assert "skill_data" in graph.state.last_error
