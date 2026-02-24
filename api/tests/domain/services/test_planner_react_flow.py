from app.domain.models.app_config import AgentConfig
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.services.flows.planner_react import PlannerReActFlow


class _DummyUoW:
    async def __aenter__(self) -> "_DummyUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _uow_factory() -> _DummyUoW:
    return _DummyUoW()


class _CapturedAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def set_runtime_system_context(self, _context: str) -> None:
        return None


def test_planner_react_flow_passes_overflow_config_to_agents(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        _CapturedAgent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        _CapturedAgent,
    )

    overflow_config = ContextOverflowConfig(
        context_window=131072,
        context_overflow_guard_enabled=True,
    )

    flow = PlannerReActFlow(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="session-ctx-overflow",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
        overflow_config=overflow_config,
    )

    assert flow.planner.kwargs["overflow_config"] is overflow_config
    assert flow.react.kwargs["overflow_config"] is overflow_config
