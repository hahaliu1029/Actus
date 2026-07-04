"""【GATE】B8 INV-B8-OFF：flag-OFF byte-identical（spec §7 R3#1）。

契约：``ctx.recalled_memory=None``（off 模式 / 默认态）下，从「新 planner
registry（已声明 recalled_memory）」assemble 出的 prompt 必须与「B8 前
registry 形态（同五 section、无 recalled_memory）」的输出在 ``text`` 与
``version_hash`` 两个层面完全一致——zh/en 各一，graph 态与 detection 态
各一。另附正向对照（recalled_memory 非 None → 围栏出现），防 gate 空转。
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.domain.models.memory_recall import RecalledMemory, RecalledMemoryItem
from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.bundles.en import EN_PLANNER_REGISTRY
from app.domain.services.prompts.bundles.zh import ZH_PLANNER_REGISTRY
from app.domain.services.prompts.render_context import build_render_context
from app.domain.services.prompts.section import PromptMode, SectionRegistry

_AGENT_CONFIG = SimpleNamespace(supports_vision=True, supports_pdf_input=False)


def _assembler() -> PromptAssembler:
    # 镜像 build_main_graph 的默认 assembler 参数
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


def _pre_b8_baseline(registry: SectionRegistry) -> SectionRegistry:
    """B8 前的 planner registry 形态：同一批 section 单例、去掉 recalled_memory。"""
    return SectionRegistry(
        sections=tuple(s for s in registry.sections if s.id != "recalled_memory"),
        name=f"{registry.name}_pre_b8_baseline",
    )


def _graph_state(lang: str) -> dict:
    return {
        "language": lang,
        "message": "整理季度销售数据",
        "skill_context": "## Available Tool Summary\n- file_read",
        "conversation_summaries": ["上一轮完成了数据清洗"],
    }


def _detection_state(lang: str) -> dict:
    # 镜像 _run_planner_for_detection 的 detection_state：无 message key
    return {
        "language": lang,
        "skill_context": "## Available Tool Summary\n- file_read",
        "conversation_summaries": ["上一轮完成了数据清洗"],
    }


@pytest.mark.parametrize("lang,registry", [("zh", ZH_PLANNER_REGISTRY), ("en", EN_PLANNER_REGISTRY)])
@pytest.mark.parametrize("state_builder", [_graph_state, _detection_state])
def test_flag_off_planner_prompt_byte_identical(lang, registry, state_builder):
    ctx = build_render_context(state_builder(lang), {"configurable": {}}, _AGENT_CONFIG)
    assert ctx.recalled_memory is None  # off 态前提
    new = _assembler().assemble(registry, ctx, PromptMode.FULL, fallback_used=False)
    old = _assembler().assemble(_pre_b8_baseline(registry), ctx, PromptMode.FULL, fallback_used=False)
    assert new.text == old.text
    assert new.version_hash == old.version_hash


def test_positive_control_fence_appears_when_recalled_present():
    """防 gate 空转：同一 registry、recalled_memory 非 None → 输出必须变。"""
    item = RecalledMemoryItem(
        chunk_id="c1", category="fact", content="数据库是 PostgreSQL 17",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), score=0.5,
    )
    recalled = RecalledMemory(items=(item,), query_hash="qh", cache_hit=False, recall_id="rid")
    ctx = build_render_context(
        _graph_state("zh"), {"configurable": {}}, _AGENT_CONFIG, recalled_memory=recalled,
    )
    result = _assembler().assemble(ZH_PLANNER_REGISTRY, ctx, PromptMode.FULL, fallback_used=False)
    assert '<recalled_memory nonce="' in result.text


def test_executor_updater_registries_ignore_recalled_memory():
    """结构守卫（spec N3 + §7 新守卫）：executor/updater registry 不含
    recalled_memory section，即使 ctx 带了召回结果也零渲染。"""
    from app.domain.services.prompts.bundles.en import (
        EN_EXECUTOR_REGISTRY, EN_UPDATER_REGISTRY,
    )
    from app.domain.services.prompts.bundles.zh import (
        ZH_EXECUTOR_REGISTRY, ZH_UPDATER_REGISTRY,
    )

    item = RecalledMemoryItem(
        chunk_id="c1", category="fact", content="内容",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc), score=0.5,
    )
    recalled = RecalledMemory(items=(item,), query_hash="qh", cache_hit=False, recall_id="rid")
    ctx = build_render_context(
        _graph_state("zh"), {"configurable": {}}, _AGENT_CONFIG, recalled_memory=recalled,
    )
    for registry in (
        ZH_EXECUTOR_REGISTRY, ZH_UPDATER_REGISTRY,
        EN_EXECUTOR_REGISTRY, EN_UPDATER_REGISTRY,
    ):
        result = _assembler().assemble(registry, ctx, PromptMode.FULL, fallback_used=False)
        assert "<recalled_memory" not in (result.text or ""), registry.name
