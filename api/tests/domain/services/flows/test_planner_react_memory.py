"""Tests for PlannerReActFlow memory integration methods."""

import pytest
from unittest.mock import MagicMock

from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step, ExecutionStatus
from app.domain.services.flows.planner_react import PlannerReActFlow


class TestBuildContextAnchor:
    """Test _build_context_anchor method."""

    def _make_flow(self) -> PlannerReActFlow:
        """Create a minimal flow instance for testing helper methods."""
        flow = object.__new__(PlannerReActFlow)
        flow.plan = None
        flow.status = MagicMock()
        return flow

    def test_anchor_without_plan(self):
        flow = self._make_flow()
        msg = MagicMock(spec=Message)
        msg.message = "继续"
        result = flow._build_context_anchor(msg)
        assert "[上下文回顾]" in result
        assert "当前消息：继续" in result

    def test_anchor_with_plan(self):
        flow = self._make_flow()
        step1 = Step(description="收集链接")
        step1.status = ExecutionStatus.COMPLETED
        step2 = Step(description="生成报告")
        step2.status = ExecutionStatus.RUNNING
        flow.plan = Plan(
            title="新闻收集",
            goal="收集AI新闻",
            steps=[step1, step2],
        )
        msg = MagicMock(spec=Message)
        msg.message = "继续"
        result = flow._build_context_anchor(msg)
        assert "原始需求：收集AI新闻" in result
        assert "已完成：收集链接" in result
        assert "待完成：生成报告" in result
