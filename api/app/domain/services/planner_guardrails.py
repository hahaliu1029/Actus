"""Small planner-side guardrails for empty-step plans with memory tools.

If the planner returns no executable steps but memory tools are available,
we replace the planner's refusal-like message with a neutral progress update
and force a single consult-memory step so execution can continue.
"""

from __future__ import annotations

from app.domain.models.llm_responses import PlanResponse, StepDef


def _memory_tools_available(
    *,
    has_memory_tools: bool,
    skill_context: str | None,
) -> bool:
    if has_memory_tools:
        return True
    text = (skill_context or "").lower()
    return "memory_search" in text and "memory_get" in text


def _progress_message(language: str) -> str:
    if language == "en":
        return "Checking memory to answer this question..."
    return "正在查询你的记忆以回答这个问题……"


def _step_description(language: str, user_message: str) -> str:
    if language == "en":
        return (
            "Consult memory to answer this question. "
            f"Original question: {user_message}"
        )
    return f"查询记忆以回答这个问题。原问题：{user_message}"


def salvage_empty_memory_recall_plan(
    parsed: PlanResponse,
    *,
    user_message: str,
    fallback_language: str,
    has_memory_tools: bool = False,
    skill_context: str | None = None,
) -> PlanResponse:
    """Rewrite an empty planner result into a consult-memory step when safe."""
    if parsed.steps:
        return parsed
    if not _memory_tools_available(
        has_memory_tools=has_memory_tools,
        skill_context=skill_context,
    ):
        return parsed

    language = parsed.language or fallback_language or "zh"
    return PlanResponse(
        title=parsed.title,
        goal=parsed.goal,
        language=language,
        steps=[StepDef(id="consult_memory", description=_step_description(language, user_message))],
        message=_progress_message(language),
    )
