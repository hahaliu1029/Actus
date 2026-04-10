"""B5 post-audit LOW #1: verify per-session telemetry lang plumbing.

End-to-end:
1. ``AgentTaskRunner.__init__`` attaches telemetry with default ``lang="zh"``
2. ``AgentTaskRunner.set_language("en")`` re-attaches telemetry with new lang
3. ``main_graph.planner_node`` reads ``config["configurable"]["language_callback"]``
   and calls it with the detected language
4. ``planner_react._build_config`` forwards ``self._language_callback`` to the
   configurable dict
5. After the callback fires, subsequent LLM invocations record the new lang
   on their telemetry events.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.infrastructure.external.llm._telemetry_mixin import (
    emit_invocation_telemetry,
)
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_llm_invocation(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def record_assembly(self, **kwargs) -> None:
        pass

    def record_lc_tools_degradation(self, **kwargs) -> None:
        pass


class TestAgentTaskRunnerSetLanguage:
    def test_set_language_updates_adapter_lang_field(self) -> None:
        """``set_language`` re-attaches telemetry and the adapter's
        ``_telemetry_lang`` reflects the new value."""
        from app.domain.services.agent_task_runner import AgentTaskRunner

        runner = AgentTaskRunner.__new__(AgentTaskRunner)
        telemetry = _RecordingTelemetry()
        runner._prompt_telemetry = telemetry
        runner._current_language = "zh"
        llm = ActusChatModel(api_key="test")
        runner._llm = llm
        runner._summary_llm_for_telemetry = None

        # Initial attach
        runner._attach_telemetry_to_llms("zh")
        assert getattr(llm, "_telemetry_lang", None) == "zh"

        # Switch to en
        runner.set_language("en")
        assert runner._current_language == "en"
        assert getattr(llm, "_telemetry_lang", None) == "en"

    def test_set_language_noop_when_same(self) -> None:
        from app.domain.services.agent_task_runner import AgentTaskRunner

        runner = AgentTaskRunner.__new__(AgentTaskRunner)
        runner._prompt_telemetry = _RecordingTelemetry()
        runner._current_language = "zh"
        llm = ActusChatModel(api_key="test")
        runner._llm = llm
        runner._summary_llm_for_telemetry = None
        runner._attach_telemetry_to_llms("zh")

        # Same-lang call should be a no-op
        runner.set_language("zh")
        assert runner._current_language == "zh"
        assert getattr(llm, "_telemetry_lang", None) == "zh"

    def test_set_language_empty_string_noop(self) -> None:
        from app.domain.services.agent_task_runner import AgentTaskRunner

        runner = AgentTaskRunner.__new__(AgentTaskRunner)
        runner._prompt_telemetry = _RecordingTelemetry()
        runner._current_language = "zh"
        llm = ActusChatModel(api_key="test")
        runner._llm = llm
        runner._summary_llm_for_telemetry = None
        runner._attach_telemetry_to_llms("zh")

        runner.set_language("")  # empty → no-op
        assert runner._current_language == "zh"

    def test_set_language_updates_summary_llm_too(self) -> None:
        from app.domain.services.agent_task_runner import AgentTaskRunner

        runner = AgentTaskRunner.__new__(AgentTaskRunner)
        runner._prompt_telemetry = _RecordingTelemetry()
        runner._current_language = "zh"
        llm = ActusChatModel(api_key="main")
        summary_llm = ActusChatModel(api_key="summary")
        runner._llm = llm
        runner._summary_llm_for_telemetry = summary_llm
        runner._attach_telemetry_to_llms("zh")

        assert getattr(llm, "_telemetry_lang", None) == "zh"
        assert getattr(summary_llm, "_telemetry_lang", None) == "zh"

        runner.set_language("en")
        assert getattr(llm, "_telemetry_lang", None) == "en"
        assert getattr(summary_llm, "_telemetry_lang", None) == "en"


class TestPlumbingEndToEnd:
    def test_emit_telemetry_reflects_post_set_language_value(self) -> None:
        """End-to-end: attach-telemetry initially with 'zh', call
        ``set_language('en')``, then emit a telemetry event. The event
        must record ``lang='en'``."""
        from app.domain.services.agent_task_runner import AgentTaskRunner
        from langchain_core.messages import SystemMessage

        runner = AgentTaskRunner.__new__(AgentTaskRunner)
        telemetry = _RecordingTelemetry()
        runner._prompt_telemetry = telemetry
        runner._current_language = "zh"
        llm = ActusChatModel(api_key="test")
        runner._llm = llm
        runner._summary_llm_for_telemetry = None
        runner._attach_telemetry_to_llms("zh")

        # Emit one event at "zh"
        emit_invocation_telemetry(llm, [SystemMessage(content="hi")], None)
        assert telemetry.calls[0]["lang"] == "zh"

        # Switch and emit again
        runner.set_language("en")
        emit_invocation_telemetry(llm, [SystemMessage(content="hi")], None)
        assert len(telemetry.calls) == 2
        assert telemetry.calls[1]["lang"] == "en"


class TestPlannerNodeCallbackWiring:
    """AST-level verification that planner_node reads ``language_callback``
    from config.configurable and calls it with the parsed language."""

    def test_planner_node_reads_language_callback(self) -> None:
        import ast
        from pathlib import Path

        main_graph_path = (
            Path(__file__).resolve().parents[3]
            / "app"
            / "domain"
            / "services"
            / "graphs"
            / "main_graph.py"
        )
        source = main_graph_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        planner_fn = None
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                and node.name == "planner_node"
            ):
                planner_fn = node
                break
        assert planner_fn is not None

        src = ast.unparse(planner_fn)
        # The callback must be read from configurable
        assert "language_callback" in src
        # And it must be called with plan.language
        assert "plan.language" in src

    def test_planner_react_flow_exposes_language_callback_field(self) -> None:
        """PlannerReActFlow must have a ``_language_callback`` attribute
        so AgentTaskRunner can inject its ``set_language`` method."""
        from app.domain.models.app_config import AgentConfig
        from app.domain.services.flows.planner_react import PlannerReActFlow

        flow = PlannerReActFlow(
            uow_factory=MagicMock(),
            llm=MagicMock(),
            agent_config=AgentConfig(
                max_iterations=10, max_retries=3, max_search_results=5
            ),
            session_id="test",
            browser=MagicMock(),
            sandbox=MagicMock(),
            search_engine=MagicMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            _allow_default_prompt_assembler=True,
        )
        assert hasattr(flow, "_language_callback")
        assert flow._language_callback is None  # Not set by default
