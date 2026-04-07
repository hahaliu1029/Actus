from unittest.mock import MagicMock

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MemoryConfig,
    MCPConfig,
    SkillRiskPolicy,
)
from app.interfaces import service_dependencies


class _FakeAppConfigRepository:
    def __init__(self, app_config: AppConfig) -> None:
        self._app_config = app_config

    def load(self) -> AppConfig:
        return self._app_config


class _FakeLLM:
    """Fake LLM that captures the keyword arguments passed to ActusChatModel."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def with_fallbacks(self, fallbacks):
        return self


class _FakeFileStorage:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _CapturedAgentService:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_get_agent_service_builds_context_overflow_config_from_llm(monkeypatch) -> None:
    app_config = AppConfig(
        llm_config=LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="key",
            model_name="gpt-4o",
            temperature=0.5,
            max_tokens=3000,
            context_window=131072,
            context_overflow_guard_enabled=True,
            overflow_retry_cap=1,
            soft_trigger_ratio=0.81,
            hard_trigger_ratio=0.92,
            reserved_output_tokens=2048,
            reserved_output_tokens_cap_ratio=0.2,
            token_estimator="provider_api",
            token_safety_factor=1.2,
            unknown_model_context_window=65536,
        ),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
    )

    monkeypatch.setattr(
        service_dependencies,
        "FileAppConfigRepository",
        lambda *args, **kwargs: _FakeAppConfigRepository(app_config),
    )
    monkeypatch.setattr(service_dependencies, "ActusChatModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "ActusResponsesModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "MinioFileStorage", _FakeFileStorage)
    monkeypatch.setattr(service_dependencies, "AgentService", _CapturedAgentService)
    monkeypatch.setattr(service_dependencies, "get_postgres", lambda: MagicMock(session_factory=MagicMock()))

    service = service_dependencies.get_agent_service(minio_store=object())
    overflow_config = service.kwargs["overflow_config"]

    assert overflow_config.context_window == 131072
    assert overflow_config.context_overflow_guard_enabled is True
    assert overflow_config.overflow_retry_cap == 1
    assert overflow_config.soft_trigger_ratio == 0.81
    assert overflow_config.hard_trigger_ratio == 0.92
    assert overflow_config.reserved_output_tokens == 2048
    assert overflow_config.reserved_output_tokens_cap_ratio == 0.2
    assert overflow_config.token_estimator == "provider_api"
    assert overflow_config.token_safety_factor == 1.2
    assert overflow_config.unknown_model_context_window == 65536


def test_get_agent_service_builds_dedicated_summary_llm(monkeypatch) -> None:
    app_config = AppConfig(
        llm_config=LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="key",
            model_name="gpt-4o",
        ),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
            memory=MemoryConfig(summary_model="gpt-4o-mini"),
        ),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
    )

    monkeypatch.setattr(
        service_dependencies,
        "FileAppConfigRepository",
        lambda *args, **kwargs: _FakeAppConfigRepository(app_config),
    )
    monkeypatch.setattr(service_dependencies, "ActusChatModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "ActusResponsesModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "MinioFileStorage", _FakeFileStorage)
    monkeypatch.setattr(service_dependencies, "AgentService", _CapturedAgentService)
    monkeypatch.setattr(service_dependencies, "get_postgres", lambda: MagicMock(session_factory=MagicMock()))

    service = service_dependencies.get_agent_service(minio_store=object())
    summary_llm = service.kwargs["summary_llm"]

    assert isinstance(summary_llm, _FakeLLM)
    assert summary_llm.kwargs["model_name"] == "gpt-4o-mini"


def test_get_agent_service_passes_memory_deps(monkeypatch) -> None:
    """C6: get_agent_service passes memory_embedding_provider,
    memory_session_factory, memory_repo_factory to AgentService."""
    from app.infrastructure.repositories.db_memory_chunk_repository import DBMemoryChunkRepository

    app_config = AppConfig(
        llm_config=LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="key",
            model_name="gpt-4o",
        ),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
    )

    monkeypatch.setattr(
        service_dependencies,
        "FileAppConfigRepository",
        lambda *args, **kwargs: _FakeAppConfigRepository(app_config),
    )
    monkeypatch.setattr(service_dependencies, "ActusChatModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "ActusResponsesModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "MinioFileStorage", _FakeFileStorage)
    monkeypatch.setattr(service_dependencies, "AgentService", _CapturedAgentService)

    mock_session_factory = MagicMock()
    mock_postgres = MagicMock()
    mock_postgres.session_factory = mock_session_factory
    monkeypatch.setattr(service_dependencies, "get_postgres", lambda: mock_postgres)

    mock_provider = MagicMock()

    service = service_dependencies.get_agent_service(
        minio_store=object(),
        memory_embedding_provider=mock_provider,
    )

    assert service.kwargs["memory_embedding_provider"] is mock_provider
    assert service.kwargs["memory_session_factory"] is mock_session_factory
    assert service.kwargs["memory_repo_factory"] is DBMemoryChunkRepository
