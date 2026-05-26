# api/app/domain/models/llm_responses.py
"""LLM 结构化输出的 Pydantic Response Models。

用于 with_structured_output(Model) 调用，所有字段使用宽松默认值
以容忍 LLM 返回部分字段。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.domain.models.work_unit import ParallelWorkUnitGroupRequest


class StepDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # [r3 P1-1] id 保持 optional；deterministic fallback 在 plan builder
    # （Task 1.9 _assign_fallback_step_id）
    id: Optional[str] = None
    description: str = ""
    # [C2 PR-1] planner 输出 parallel_work_units（with_structured_output 由 LLM 填）
    parallel_work_units: Optional[ParallelWorkUnitGroupRequest] = None


class PlanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str = ""
    goal: str = ""
    title: str = ""
    language: str = "zh"
    steps: list[StepDef] = []


class PlanUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    steps: list[StepDef] = []


class SummarizerOutput(BaseModel):
    """summarizer_node 的 LLM 返回（用户最终交付：message + attachments）。

    Some LLMs return ``result`` instead of ``message``; we accept both.
    """
    model_config = ConfigDict(extra="ignore")
    message: str = ""
    result: str = ""
    attachments: list[str] = []

    @property
    def text(self) -> str:
        """Return whichever of message/result was populated."""
        return self.message or self.result


class ConversationSummaryResponse(BaseModel):
    """_generate_summary 的 LLM 返回（ConversationSummary 持久化）。"""
    model_config = ConfigDict(extra="ignore")
    user_intent: str = ""
    plan_summary: str = ""
    execution_results: list[str] = []
    decisions: list[str] = []
    unresolved: list[str] = []


class ContinuationIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    is_continuation: bool = False
