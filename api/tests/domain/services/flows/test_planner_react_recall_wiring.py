"""B8 PR-3: PlannerReActFlow 侧接线合同。

1. _ensure_graphs：_build_memory_recall_provider() 结果存 self 并透传
   build_main_graph（monkeypatch 图构建器，镜像 approval_state_reader
   wiring 测试的 ctor 级契约风格 + graph 构建 spy）
2. detection 入口：material 契约（entry="detection"、original_request=
   self.plan.goal）+ 围栏进 detection SystemMessage + provider 异常降级

fixture（_make_flow / _recalled_fixture / _make_structured_planner_llm）
本文件内本地定义——分别镜像 test_memory_recall_provider.py 与
test_main_graph_memory_plumbing.py 的形态；测试文件间不互相 import。
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.app_config import MemoryConfig
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef
from app.domain.models.memory_recall import (
    RecallQueryMaterial,
    RecalledMemory,
    RecalledMemoryItem,
)
from app.domain.models.message import Message
from app.domain.services.flows import planner_react as planner_react_mod
from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---- fixtures ----------------------------------------------------------- #


def _memory_cfg(**overrides):
    defaults = dict(recall_mode="on")
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _make_structured_planner_llm(
    create_response: PlanResponse | None = None,
    update_response: PlanUpdateResponse | None = None,
):
    """Return ``(llm, create_structured)``. ``create_structured.ainvoke`` is
    an AsyncMock whose ``await_args.args[0][0].content`` is the detection
    planner SystemMessage text (messages[0])."""
    if create_response is None:
        create_response = PlanResponse(
            title="t", goal="g", language="zh",
            steps=[StepDef(description="Step 1")], message="ok",
        )
    if update_response is None:
        update_response = PlanUpdateResponse(steps=[])

    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=create_response)
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(return_value=update_response)

    llm = MagicMock()

    def _with_structured_output(schema, **_kwargs):
        if schema is PlanResponse:
            return create_structured
        if schema is PlanUpdateResponse:
            return update_structured
        return create_structured

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
    return llm, create_structured


def _make_flow(llm=None, **overrides):
    """镜像 test_memory_recall_provider.py._make_flow 的最小构造。"""
    defaults = dict(
        uow_factory=MagicMock(),
        llm=llm if llm is not None else MagicMock(),
        agent_config=MagicMock(memory=_memory_cfg()),
        session_id="sess-1",
        browser=MagicMock(),
        sandbox=MagicMock(),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
    )
    defaults.update(overrides)
    return PlannerReActFlow(**defaults)


def _recalled_fixture():
    item = RecalledMemoryItem(
        chunk_id="c1", category="fact", content="数据库是 PostgreSQL 17",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), score=0.5,
    )
    return RecalledMemory(items=(item,), query_hash="qh", cache_hit=False, recall_id="rid")


class TestEnsureGraphsPassThrough:
    async def test_provider_stored_and_passed_to_build_main_graph(self, monkeypatch):
        flow = _make_flow(_allow_default_prompt_assembler=True)
        # 轻量化 _ensure_graphs 的重依赖
        flow._get_checkpointer = AsyncMock(return_value=None)
        flow._collect_all_tools = AsyncMock(return_value=[])
        monkeypatch.setattr(planner_react_mod, "build_react_graph", MagicMock(return_value=MagicMock()))
        captured: dict = {}

        def _spy_build_main_graph(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(planner_react_mod, "build_main_graph", _spy_build_main_graph)
        sentinel_provider = AsyncMock()
        monkeypatch.setattr(
            flow, "_build_memory_recall_provider", MagicMock(return_value=sentinel_provider),
        )
        await flow._ensure_graphs()
        assert flow._memory_recall_provider is sentinel_provider
        assert captured["memory_recall_provider"] is sentinel_provider


class TestDetectionEntry:
    async def test_material_contract_and_fence_injection(self):
        llm, create_structured = _make_structured_planner_llm()
        flow = _make_flow(llm=llm, _allow_default_prompt_assembler=True)
        flow.plan = MagicMock(goal="分析季度销售数据")
        seen: list[RecallQueryMaterial] = []

        async def _spy_provider(material):
            seen.append(material)
            return _recalled_fixture()

        flow._memory_recall_provider = _spy_provider
        message = Message(message="输出为ppt", language="zh")
        plan, events = await flow._run_planner_for_detection(message, [])

        assert seen[0].entry == "detection"
        assert seen[0].message == "输出为ppt"
        assert seen[0].original_request == "分析季度销售数据"
        assert seen[0].session_title is None
        sys_text = create_structured.ainvoke.await_args.args[0][0].content
        assert '<recalled_memory nonce="' in sys_text

    async def test_first_turn_original_request_none(self):
        llm, _ = _make_structured_planner_llm()
        flow = _make_flow(llm=llm, _allow_default_prompt_assembler=True)
        assert flow.plan is None
        seen: list[RecallQueryMaterial] = []

        async def _spy_provider(material):
            seen.append(material)
            return None

        flow._memory_recall_provider = _spy_provider
        await flow._run_planner_for_detection(Message(message="你好", language="zh"), [])
        assert seen[0].original_request is None

    async def test_no_provider_keeps_prompt_clean(self):
        llm, create_structured = _make_structured_planner_llm()
        flow = _make_flow(llm=llm, _allow_default_prompt_assembler=True)
        assert flow._memory_recall_provider is None
        await flow._run_planner_for_detection(Message(message="你好", language="zh"), [])
        sys_text = create_structured.ainvoke.await_args.args[0][0].content
        assert "<recalled_memory" not in sys_text

    async def test_provider_error_degrades(self):
        llm, create_structured = _make_structured_planner_llm()
        flow = _make_flow(llm=llm, _allow_default_prompt_assembler=True)

        async def _boom(material):
            raise RuntimeError("recall down")

        flow._memory_recall_provider = _boom
        plan, _ = await flow._run_planner_for_detection(Message(message="你好", language="zh"), [])
        assert plan is not None     # 不抛、照常出 plan
        assert "<recalled_memory" not in create_structured.ainvoke.await_args.args[0][0].content
