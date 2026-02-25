from __future__ import annotations

import pytest
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AgentConfig,
    MCPConfig,
    MCPServerConfig,
    MCPTransport,
)
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.event import MessageEvent
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.user_tool_preference import ToolType
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _NoopSessionRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, object]] = []

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_title(self, session_id: str, title: str) -> None:
        return None

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        return None

    async def increment_unread_message_count(self, session_id: str) -> None:
        return None


class _NoopUoW:
    def __init__(self) -> None:
        self.session = _NoopSessionRepository()

    async def __aenter__(self) -> "_NoopUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def _uow_factory() -> _NoopUoW:
    return _NoopUoW()


class _DummyFlow:
    def __init__(self, **kwargs) -> None:
        self.skill_contexts: list[str] = []
        self.kwargs = kwargs

    def set_skill_context(self, skill_context: str) -> None:
        self.skill_contexts.append(skill_context)

    async def invoke(self, message):
        if False:
            yield message


class _CapturingFlow:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.skill_contexts: list[str] = []

    def set_skill_context(self, skill_context: str) -> None:
        self.skill_contexts.append(skill_context)

    async def invoke(self, message):
        if False:
            yield message


class _EmptyInputStream:
    async def is_empty(self) -> bool:
        return True


class _SingleMessageInputStream:
    def __init__(self, message: str) -> None:
        self._items = [
            (
                "evt-1",
                MessageEvent(role="user", message=message, attachments=[]).model_dump_json(),
            )
        ]

    async def is_empty(self) -> bool:
        return len(self._items) == 0

    async def pop(self):
        return self._items.pop(0)


class _OutputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        self.events.append(event_json)
        return f"event-{len(self.events)}"


class _DummyTask:
    def __init__(self) -> None:
        self.input_stream = _EmptyInputStream()
        self.output_stream = _OutputStream()


class _DummyMessageTask:
    def __init__(self, message: str) -> None:
        self.input_stream = _SingleMessageInputStream(message)
        self.output_stream = _OutputStream()


class _FakeSandbox:
    async def ensure_sandbox(self) -> None:
        return None


class _FakeMCPTool:
    def __init__(self) -> None:
        self.initialized_with: MCPConfig | None = None

    async def initialize(self, mcp_config: MCPConfig) -> None:
        self.initialized_with = mcp_config

    async def cleanup(self) -> None:
        return None


class _FakeA2ATool:
    def __init__(self) -> None:
        self.initialized_with: A2AConfig | None = None
        self.manager = None

    async def initialize(self, a2a_config: A2AConfig) -> None:
        self.initialized_with = a2a_config

    async def cleanup(self) -> None:
        return None


class _FakeSkillTool:
    def __init__(self) -> None:
        self.initialized_with: list[Skill] | None = None
        self.initialize_history: list[list[Skill]] = []

    async def initialize(self, skills: list[Skill]) -> None:
        self.initialized_with = skills
        self.initialize_history.append(list(skills))

    async def cleanup(self) -> None:
        return None


class _FakeSkillBundleSync:
    def __init__(self) -> None:
        self.prepare_calls: list[tuple[list[Skill], list[Skill]]] = []
        self.await_initial_calls = 0
        self.start_calls = 0
        self.cleanup_calls = 0
        self.sequence: list[str] = []

    async def prepare_startup_sync(
        self,
        skill_pool: list[Skill],
        initial_selected: list[Skill],
    ) -> None:
        self.prepare_calls.append((list(skill_pool), list(initial_selected)))
        self.sequence.append("prepare")

    async def await_initial_sync(self) -> None:
        self.await_initial_calls += 1
        self.sequence.append("await")

    def start_background_sync(self) -> None:
        self.start_calls += 1
        self.sequence.append("start")

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        self.sequence.append("cleanup")


class _FakeContinuationClassifier:
    def __init__(self, *, decision: bool = False) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str]] = []

    async def classify(self, current_message: str, previous_substantive_message: str) -> bool:
        self.calls.append((current_message, previous_substantive_message))
        return self.decision


def _build_skill(skill_id: str, *, name: str) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=name,
        description=f"{name} description",
        source_type=SkillSourceType.GITHUB,
        source_ref=f"github:owner/{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={"runtime_type": "native", "tools": []},
        enabled=True,
    )


async def test_invoke_applies_user_preferences_before_tool_initialization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(
            mcpServers={
                "mcp-enabled": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://enabled.example.com/mcp",
                ),
                "mcp-disabled-by-user": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://disabled.example.com/mcp",
                ),
            }
        ),
        a2a_config=A2AConfig(
            a2a_servers=[
                A2AServerConfig(id="a2a-enabled", base_url="https://a2a-enabled.example.com", enabled=True),
                A2AServerConfig(id="a2a-disabled-by-user", base_url="https://a2a-disabled.example.com", enabled=True),
            ]
        ),
        session_id="session-1",
        user_id="user-1",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )

    fake_mcp_tool = _FakeMCPTool()
    fake_a2a_tool = _FakeA2ATool()
    fake_skill_tool = _FakeSkillTool()
    fake_sync = _FakeSkillBundleSync()
    runner._mcp_tool = fake_mcp_tool
    runner._a2a_tool = fake_a2a_tool
    runner._skill_tool = fake_skill_tool
    runner._skill_bundle_sync = fake_sync

    preferences_by_tool_type = {
        ToolType.MCP: {"mcp-disabled-by-user": False},
        ToolType.A2A: {"a2a-disabled-by-user": False},
        ToolType.SKILL: {"skill-disabled-by-user": False},
    }

    async def fake_load_user_preferences_map(tool_type: ToolType) -> dict[str, bool]:
        return preferences_by_tool_type.get(tool_type, {})

    async def fake_load_enabled_skills() -> list[Skill]:
        return [
            _build_skill("skill-enabled", name="Skill Enabled"),
            _build_skill("skill-disabled-by-user", name="Skill Disabled"),
        ]

    monkeypatch.setattr(runner, "_load_user_preferences_map", fake_load_user_preferences_map)
    monkeypatch.setattr(runner, "_load_enabled_skills", fake_load_enabled_skills)

    await runner.invoke(_DummyTask())

    assert fake_mcp_tool.initialized_with is not None
    assert fake_mcp_tool.initialized_with.mcpServers["mcp-enabled"].enabled is True
    assert (
        fake_mcp_tool.initialized_with.mcpServers["mcp-disabled-by-user"].enabled
        is False
    )

    assert fake_a2a_tool.initialized_with is not None
    a2a_enabled_map = {
        server.id: server.enabled for server in fake_a2a_tool.initialized_with.a2a_servers
    }
    assert a2a_enabled_map["a2a-enabled"] is True
    assert a2a_enabled_map["a2a-disabled-by-user"] is False

    assert fake_skill_tool.initialized_with is not None
    assert [skill.id for skill in fake_skill_tool.initialized_with] == ["skill-enabled"]
    assert runner._flow.skill_contexts
    assert "Skill Enabled" in runner._flow.skill_contexts[-1]


async def test_invoke_starts_skill_sync_with_filtered_pool(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="session-2",
        user_id="user-2",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )

    fake_skill_tool = _FakeSkillTool()
    fake_sync = _FakeSkillBundleSync()
    runner._mcp_tool = _FakeMCPTool()
    runner._a2a_tool = _FakeA2ATool()
    runner._skill_tool = fake_skill_tool
    runner._skill_bundle_sync = fake_sync

    async def fake_load_user_preferences_map(tool_type: ToolType) -> dict[str, bool]:
        if tool_type == ToolType.SKILL:
            return {"skill-disabled-by-user": False}
        return {}

    async def fake_load_enabled_skills() -> list[Skill]:
        return [
            _build_skill("skill-enabled", name="Skill Enabled"),
            _build_skill("skill-disabled-by-user", name="Skill Disabled"),
        ]

    monkeypatch.setattr(runner, "_load_user_preferences_map", fake_load_user_preferences_map)
    monkeypatch.setattr(runner, "_load_enabled_skills", fake_load_enabled_skills)

    await runner.invoke(_DummyTask())

    assert fake_sync.prepare_calls
    prepared_pool, initial_selected = fake_sync.prepare_calls[0]
    assert [skill.id for skill in prepared_pool] == ["skill-enabled"]
    assert [skill.id for skill in initial_selected] == ["skill-enabled"]
    assert fake_sync.await_initial_calls == 1
    assert fake_sync.start_calls == 1
    assert fake_sync.sequence[:3] == ["prepare", "await", "start"]


async def test_invoke_uses_frozen_skill_pool_for_each_message(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="session-3",
        user_id="user-3",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )

    fake_skill_tool = _FakeSkillTool()
    fake_sync = _FakeSkillBundleSync()
    runner._mcp_tool = _FakeMCPTool()
    runner._a2a_tool = _FakeA2ATool()
    runner._skill_tool = fake_skill_tool
    runner._skill_bundle_sync = fake_sync

    load_calls = {"count": 0}

    async def fake_load_user_preferences_map(tool_type: ToolType) -> dict[str, bool]:
        return {}

    async def fake_load_enabled_skills() -> list[Skill]:
        load_calls["count"] += 1
        return [_build_skill("skill-enabled", name="Skill Enabled")]

    monkeypatch.setattr(runner, "_load_user_preferences_map", fake_load_user_preferences_map)
    monkeypatch.setattr(runner, "_load_enabled_skills", fake_load_enabled_skills)

    await runner.invoke(_DummyMessageTask("please use skill"))

    assert load_calls["count"] == 1
    assert len(fake_skill_tool.initialize_history) >= 2
    assert [skill.id for skill in fake_skill_tool.initialize_history[0]] == ["skill-enabled"]
    assert [skill.id for skill in fake_skill_tool.initialize_history[1]] == ["skill-enabled"]


async def test_runner_passes_overflow_config_to_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _CapturingFlow,
    )

    overflow_config = ContextOverflowConfig(
        context_window=131072,
        context_overflow_guard_enabled=True,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="session-4",
        user_id="user-4",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
        overflow_config=overflow_config,
    )

    assert runner._flow.kwargs["overflow_config"] is overflow_config


async def test_initial_skills_do_not_seed_anchor(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="session-initial-anchor",
        user_id="user-initial-anchor",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )
    runner._mcp_tool = _FakeMCPTool()
    runner._a2a_tool = _FakeA2ATool()
    runner._skill_tool = _FakeSkillTool()
    runner._skill_bundle_sync = _FakeSkillBundleSync()

    async def fake_load_user_preferences_map(tool_type: ToolType) -> dict[str, bool]:
        return {}

    async def fake_load_enabled_skills() -> list[Skill]:
        return [
            _build_skill("skill-a", name="Skill A"),
            _build_skill("skill-b", name="Skill B"),
        ]

    monkeypatch.setattr(runner, "_load_user_preferences_map", fake_load_user_preferences_map)
    monkeypatch.setattr(runner, "_load_enabled_skills", fake_load_enabled_skills)

    await runner.invoke(_DummyTask())

    assert runner._last_effective_selected_skills == []
    assert runner._last_substantive_user_message == ""


async def test_skill_anchor_switches_a_continue_b_continue(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="session-anchor-switch",
        user_id="user-anchor-switch",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )

    sql_skill = _build_skill("sql", name="SQL")
    sql_skill.description = "sql optimize query index SQL优化"
    log_skill = _build_skill("log", name="Log")
    log_skill.description = "analyze log trace diagnose 查日志 日志"
    pool = [sql_skill, log_skill]

    selected_a, _ = await runner._select_skills_for_message(pool, "请做 sql 优化")
    selected_continue_a, _ = await runner._select_skills_for_message(pool, "继续")
    selected_b, _ = await runner._select_skills_for_message(pool, "查日志")
    selected_continue_b, _ = await runner._select_skills_for_message(pool, "继续")

    assert selected_a[0].id == "sql"
    assert selected_continue_a[0].id == "sql"
    assert selected_b[0].id == "log"
    assert selected_continue_b[0].id == "log"


async def test_ambiguous_message_triggers_llm_but_explicit_message_does_not(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="session-llm-trigger",
        user_id="user-llm-trigger",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )

    sql_skill = _build_skill("sql", name="SQL")
    sql_skill.description = "sql optimize query index"
    pool = [sql_skill]

    fake_classifier = _FakeContinuationClassifier(decision=True)
    runner._continuation_classifier = fake_classifier

    await runner._select_skills_for_message(pool, "请帮我做 sql 优化")
    await runner._select_skills_for_message(pool, "咋办")
    assert len(fake_classifier.calls) == 1

    await runner._select_skills_for_message(pool, "请继续处理 SQL 优化并补充执行计划")
    assert len(fake_classifier.calls) == 1
