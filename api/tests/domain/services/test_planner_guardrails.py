from app.domain.models.llm_responses import PlanResponse, StepDef
from app.domain.services.planner_guardrails import salvage_empty_memory_recall_plan


def test_empty_steps_with_memory_tools_gets_consult_memory_step() -> None:
    parsed = PlanResponse(
        title="职业信息查询",
        goal="",
        language="zh",
        steps=[],
        message="作为AI助手，我无法直接知道您的职业是什么。",
    )

    result = salvage_empty_memory_recall_plan(
        parsed,
        user_message="我的职业是什么",
        fallback_language="zh",
        has_memory_tools=True,
        skill_context="",
    )

    assert len(result.steps) == 1
    assert result.steps[0].id == "consult_memory"
    assert result.message == "正在查询你的记忆以回答这个问题……"


def test_non_empty_steps_are_left_untouched() -> None:
    parsed = PlanResponse(
        title="整理任务",
        goal="完成整理",
        language="zh",
        steps=[StepDef(id="1", description="先搜索资料"), StepDef(id="2", description="整理成 PPT")],
        message="我先开始执行。",
    )

    result = salvage_empty_memory_recall_plan(
        parsed,
        user_message="帮我搜 X 并整理成 PPT",
        fallback_language="zh",
        has_memory_tools=True,
        skill_context="## Available Tool Summary\n- memory: memory_search, memory_get",
    )

    assert result == parsed


def test_empty_steps_without_memory_tools_stays_original() -> None:
    parsed = PlanResponse(
        title="职业信息查询",
        goal="",
        language="zh",
        steps=[],
        message="作为AI助手，我无法直接知道您的职业是什么。",
    )

    result = salvage_empty_memory_recall_plan(
        parsed,
        user_message="我的职业是什么",
        fallback_language="zh",
        has_memory_tools=False,
        skill_context="",
    )

    assert result == parsed
