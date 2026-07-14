"""Tests for service_dependencies: _build_config_snapshot, _build_agent_service, get_agent_service."""
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends, FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    FileUnderstandingConfig,
    LLMConfig,
    MemoryConfig,
    MCPConfig,
    SkillRiskPolicy,
    VisionFallbackConfig,
)
from app.interfaces import service_dependencies
from app.interfaces.service_dependencies import get_agent_service, get_session_service


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


def test_get_session_service_supports_websocket_dependency() -> None:
    app = FastAPI()
    app.state.sandbox_lifecycle_service = MagicMock()
    fake_service = MagicMock()

    with patch.object(service_dependencies, "SessionService", return_value=fake_service):

        @app.websocket("/ws")
        async def ws_endpoint(
            websocket: WebSocket,
            session_service=Depends(get_session_service),
        ) -> None:
            await websocket.accept()
            await websocket.send_text("ok" if session_service is fake_service else "bad")
            await websocket.close()

        client = TestClient(app)
        try:
            with client.websocket_connect("/ws") as websocket:
                assert websocket.receive_text() == "ok"
        finally:
            client.close()


def test_get_agent_service_supports_websocket_dependency(monkeypatch) -> None:
    app = FastAPI()
    app.state.agent_service = MagicMock()

    monkeypatch.setattr(service_dependencies, "_load_app_config", lambda: None)
    monkeypatch.setattr(service_dependencies, "_config_generation", 1)
    monkeypatch.setattr(service_dependencies, "_last_refresh_generation", 1)

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket,
        agent_service=Depends(get_agent_service),
    ) -> None:
        await websocket.accept()
        await websocket.send_text("ok" if agent_service is app.state.agent_service else "bad")
        await websocket.close()

    client = TestClient(app)
    try:
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_text() == "ok"
    finally:
        client.close()


def test_build_config_snapshot_builds_context_overflow_config_from_llm(monkeypatch) -> None:
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

    # Clear LLM cache to avoid stale entries from other tests
    service_dependencies._llm_cache.clear()

    snapshot = service_dependencies._build_config_snapshot(app_config)
    overflow_config = snapshot.overflow_config

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


def test_build_config_snapshot_builds_dedicated_summary_llm(monkeypatch) -> None:
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

    # Clear LLM cache to avoid stale entries from other tests
    service_dependencies._llm_cache.clear()

    snapshot = service_dependencies._build_config_snapshot(app_config)
    summary_llm = snapshot.summary_llm

    assert isinstance(summary_llm, _FakeLLM)
    assert summary_llm.kwargs["model_name"] == "gpt-4o-mini"


# --- D5.1: per-call timeout wiring regression ----------------------------- #


def test_llm_fingerprint_changes_with_timeout(monkeypatch) -> None:
    """D5.1: different timeout_seconds values must produce different cached
    instances — otherwise a config change would not invalidate the cache.
    """
    monkeypatch.setattr(service_dependencies, "ActusChatModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "ActusResponsesModel", _FakeLLM)
    service_dependencies._llm_cache.clear()

    base_kwargs = dict(
        base_url="https://api.openai.com/v1",
        api_key="key",
        model_name="gpt-4o",
    )
    fp_45 = service_dependencies._llm_fingerprint(
        LLMConfig(**base_kwargs, timeout_seconds=45.0)
    )
    fp_46 = service_dependencies._llm_fingerprint(
        LLMConfig(**base_kwargs, timeout_seconds=46.0)
    )
    assert fp_45 != fp_46

    # Build two LLMs with different timeouts — cache must not collide
    llm_a = service_dependencies._build_llm(
        LLMConfig(**base_kwargs, timeout_seconds=45.0)
    )
    llm_b = service_dependencies._build_llm(
        LLMConfig(**base_kwargs, timeout_seconds=46.0)
    )
    assert llm_a is not llm_b
    assert llm_a.kwargs["timeout_seconds"] == 45.0
    assert llm_b.kwargs["timeout_seconds"] == 46.0


def test_summary_llm_uses_default_when_field_omitted(monkeypatch) -> None:
    """D5.1 tri-state (see config.yaml.example:88):
    omitting summary_timeout_seconds should NOT silently inherit the main
    llm_config.timeout_seconds — it loads as the model default (30.0).

    This locks in the semantics so a future change to MemoryConfig's default
    (e.g., flipping it to None for "inherit on omit") would trip this test.
    """
    app_config = AppConfig(
        llm_config=LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="key",
            model_name="gpt-4o",
            timeout_seconds=90.0,
        ),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
            # summary_timeout_seconds intentionally omitted: should use the
            # 30.0 default, not the main LLM's 90.0 timeout.
            memory=MemoryConfig(summary_model="gpt-4o-mini"),
        ),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy=SkillRiskPolicy(),
    )
    monkeypatch.setattr(service_dependencies, "ActusChatModel", _FakeLLM)
    monkeypatch.setattr(service_dependencies, "ActusResponsesModel", _FakeLLM)
    service_dependencies._llm_cache.clear()

    snapshot = service_dependencies._build_config_snapshot(app_config)

    assert isinstance(snapshot.summary_llm, _FakeLLM)
    # 30.0 default — NOT inherited from the main LLM's 90.0
    assert snapshot.summary_llm.kwargs["timeout_seconds"] == 30.0
    # Main LLM unchanged
    assert snapshot.llm.kwargs["timeout_seconds"] == 90.0


class TestBuildLlmTimeoutSeconds:
    """D5.1: _build_llm must pass timeout_seconds to adapter constructors.

    Uses the REAL adapter classes (not patched mocks) so the assertion
    hits the adapter's own ``timeout_seconds`` attribute — a regression
    guard against accidentally dropping the kwarg in _build_llm.
    """

    def test_build_llm_propagates_timeout_seconds_to_chat_model(self) -> None:
        from app.domain.models.app_config import LLMConfig
        from app.interfaces.service_dependencies import _build_llm, _llm_cache

        _llm_cache.clear()

        cfg = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            api_type="chat_completions",
            timeout_seconds=55.0,
        )
        llm = _build_llm(cfg)
        assert llm.timeout_seconds == 55.0

    def test_build_llm_propagates_timeout_seconds_to_responses_model(self) -> None:
        from app.domain.models.app_config import LLMConfig
        from app.interfaces.service_dependencies import _build_llm, _llm_cache

        _llm_cache.clear()

        cfg = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            api_type="responses",
            timeout_seconds=55.0,
        )
        llm = _build_llm(cfg)
        assert llm.timeout_seconds == 55.0

    def test_build_llm_fallback_mode_propagates_to_both_children(self) -> None:
        from app.domain.models.app_config import LLMConfig
        from app.interfaces.service_dependencies import _build_llm, _llm_cache

        _llm_cache.clear()

        cfg = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            api_type="auto",
            timeout_seconds=55.0,
        )
        llm = _build_llm(cfg)
        assert llm.primary.timeout_seconds == 55.0
        assert llm.fallback.timeout_seconds == 55.0


class TestBuildConfigSnapshotSummaryTimeout:
    """D5.1: summary_llm must receive memory.summary_timeout_seconds when set."""

    def _make_app_config(self, *, summary_timeout: float | None) -> AppConfig:
        """Build a minimal AppConfig with a memory.summary_model override."""
        memory = MemoryConfig(
            summary_model="gpt-4o-mini",
            summary_timeout_seconds=summary_timeout,
        )
        agent = AgentConfig(memory=memory)
        llm = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="primary-model",
            timeout_seconds=120.0,
        )
        return AppConfig(
            llm_config=llm,
            agent_config=agent,
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
            file_understanding=FileUnderstandingConfig(),
        )

    def test_summary_uses_override_when_set(self) -> None:
        from app.interfaces.service_dependencies import _build_config_snapshot, _llm_cache

        _llm_cache.clear()
        app_cfg = self._make_app_config(summary_timeout=10.0)
        snap = _build_config_snapshot(app_cfg)
        assert snap.summary_llm is not None
        assert snap.summary_llm.timeout_seconds == 10.0

    def test_summary_inherits_llm_config_when_none(self) -> None:
        from app.interfaces.service_dependencies import _build_config_snapshot, _llm_cache

        _llm_cache.clear()
        app_cfg = self._make_app_config(summary_timeout=None)
        snap = _build_config_snapshot(app_cfg)
        assert snap.summary_llm is not None
        # Inherits primary's 120.0 because override is None
        assert snap.summary_llm.timeout_seconds == 120.0


class TestBuildConfigSnapshotVisionFallbackTimeout:
    """D5.1 cleanup: vision_fallback_model must inherit timeout_seconds
    from the main llm_config (Codex review finding, MEDIUM).

    Pre-fix: ``_build_config_snapshot`` constructed ``vision_llm_config``
    without passing ``timeout_seconds``, so the vision fallback adapter
    always fell back to ``LLMConfig.timeout_seconds``'s Pydantic default
    (120s) regardless of what the user set in their main ``llm_config``.
    Downstream (``image.py``, ``video.py``) used this adapter for vision
    description and frame analysis, so their user-facing timeout was
    effectively pinned at 120s.

    Fix: inherit ``app_config.llm_config.timeout_seconds`` directly,
    matching the ``summary_llm`` inheritance semantics used when
    ``MemoryConfig.summary_timeout_seconds`` is ``None``.
    """

    def _make_app_config_with_vision_fallback(
        self, *, main_timeout: float
    ) -> AppConfig:
        """Build a minimal AppConfig with vision_fallback enabled."""
        memory = MemoryConfig(summary_model="gpt-4o-mini")
        agent = AgentConfig(memory=memory)
        llm = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="primary-model",
            timeout_seconds=main_timeout,
        )
        vision_fallback = VisionFallbackConfig(
            enabled=True,
            base_url="https://vision.test/v1",
            api_key="vk",
            model_name="vision-model",
        )
        file_understanding = FileUnderstandingConfig(
            vision_fallback=vision_fallback,
        )
        return AppConfig(
            llm_config=llm,
            agent_config=agent,
            mcp_config=MCPConfig(),
            a2a_config=A2AConfig(),
            file_understanding=file_understanding,
            skill_risk_policy=SkillRiskPolicy(),
        )

    def test_vision_fallback_inherits_main_timeout_seconds(self) -> None:
        from app.interfaces.service_dependencies import (
            _build_config_snapshot,
            _llm_cache,
        )

        _llm_cache.clear()
        app_cfg = self._make_app_config_with_vision_fallback(main_timeout=55.0)
        snap = _build_config_snapshot(app_cfg)
        assert snap.vision_fallback_model is not None
        assert snap.vision_fallback_model.timeout_seconds == 55.0, (
            f"vision_fallback_model lost main timeout_seconds; got "
            f"{snap.vision_fallback_model.timeout_seconds}, expected 55.0"
        )

    def test_vision_fallback_inherits_custom_main_timeout(self) -> None:
        """Second data point: ensure it's not accidentally hardcoded to 55.0."""
        from app.interfaces.service_dependencies import (
            _build_config_snapshot,
            _llm_cache,
        )

        _llm_cache.clear()
        app_cfg = self._make_app_config_with_vision_fallback(main_timeout=180.0)
        snap = _build_config_snapshot(app_cfg)
        assert snap.vision_fallback_model is not None
        assert snap.vision_fallback_model.timeout_seconds == 180.0


class TestBuildLlmBudgetIndependence:
    """Fallback per-call timeouts are independent from an unlimited root run."""

    def test_no_budget_warning_with_default_timeout(self, caplog: pytest.LogCaptureFixture) -> None:
        from app.domain.models.app_config import LLMConfig
        from app.interfaces.service_dependencies import _build_llm, _llm_cache

        _llm_cache.clear()
        cfg = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            api_type="auto",
        )
        with caplog.at_level(logging.WARNING, logger="app.interfaces.service_dependencies"):
            _build_llm(cfg)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not any("budget" in msg for msg in warning_messages), (
            f"Unexpected root-budget warning: {warning_messages}"
        )

    def test_no_warning_when_auto_and_under_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        from app.domain.models.app_config import LLMConfig
        from app.interfaces.service_dependencies import _build_llm, _llm_cache

        _llm_cache.clear()
        cfg = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            api_type="auto",
            timeout_seconds=90.0,  # 90 + 90 = 180 < 200 threshold
        )
        with caplog.at_level(logging.WARNING, logger="app.interfaces.service_dependencies"):
            _build_llm(cfg)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not any(
            "budget" in msg for msg in warning_messages
        ), f"Unexpected budget warning: {warning_messages}"

    def test_no_warning_when_api_type_not_auto(self, caplog: pytest.LogCaptureFixture) -> None:
        from app.domain.models.app_config import LLMConfig
        from app.interfaces.service_dependencies import _build_llm, _llm_cache

        _llm_cache.clear()
        cfg = LLMConfig(
            base_url="https://x.test/v1",
            api_key="k",
            model_name="m",
            api_type="chat_completions",
            timeout_seconds=300.0,  # would trigger if api_type were auto
        )
        with caplog.at_level(logging.WARNING, logger="app.interfaces.service_dependencies"):
            _build_llm(cfg)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not any(
            "budget" in msg for msg in warning_messages
        ), f"Unexpected budget warning for non-auto mode: {warning_messages}"


def test_build_agent_service_passes_memory_deps(monkeypatch) -> None:
    """_build_agent_service passes memory deps through to AgentService."""
    from app.infrastructure.repositories.db_memory_chunk_repository import DBMemoryChunkRepository

    class _FakeMeter:
        def __init__(self) -> None:
            self.created_counters: list[str] = []

        def create_counter(self, name: str):
            self.created_counters.append(name)
            return MagicMock()

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
    fake_meter = _FakeMeter()

    import app.infrastructure.observability as observability

    monkeypatch.setattr(observability, "OtelMeter", lambda: fake_meter)

    mock_session_factory = MagicMock()
    mock_postgres = MagicMock()
    mock_postgres.session_factory = mock_session_factory
    monkeypatch.setattr(service_dependencies, "get_postgres", lambda: mock_postgres)

    # Clear caches to avoid stale entries from other tests
    service_dependencies._llm_cache.clear()
    service_dependencies._config_cache = None
    service_dependencies._config_expiry = 0.0

    mock_provider = MagicMock()

    service = service_dependencies._build_agent_service(
        minio_store=object(),
        redis_client=MagicMock(),
        checkpointer_pool=MagicMock(),
        flush_service=MagicMock(),
        memory_embedding_provider=mock_provider,
    )

    assert service.kwargs["memory_embedding_provider"] is mock_provider
    assert service.kwargs["memory_session_factory"] is mock_session_factory
    assert service.kwargs["memory_repo_factory"] is DBMemoryChunkRepository
    assert "actus_supervisor_admit_total" in fake_meter.created_counters


# ── PR-2 regression: get_memory_management_service handles uninitialized Redis ──


def test_get_memory_management_service_handles_uninitialized_redis(monkeypatch):
    """Redis 未 init（lifespan 绕过 / 单测环境）→ factory 必须降级为
    redis=None+quota=None 一起传，不能触发 MemoryManagementService 的
    'both or neither' 校验抛 ValueError。
    """
    from app.interfaces.service_dependencies import get_memory_management_service
    from app.application.services.memory_management_service import MemoryManagementService

    class _FakeRedisSingleton:
        @property
        def client(self):
            raise RuntimeError("Redis客户端未初始化")

    class _FakePostgres:
        session_factory = MagicMock()

    monkeypatch.setattr(
        "app.infrastructure.storage.redis.get_redis",
        lambda: _FakeRedisSingleton(),
    )
    monkeypatch.setattr(service_dependencies, "get_postgres", lambda: _FakePostgres())

    request = MagicMock()
    request.app.state.memory_embedding_provider = MagicMock()
    # file_memory_store 属性不存在 → getattr 走 None fallback（PR-0 验证过）
    delattr_safe = lambda obj, name: (
        delattr(obj, name) if hasattr(obj, name) else None
    )
    delattr_safe(request.app.state, "file_memory_store")

    svc = get_memory_management_service(request)

    assert isinstance(svc, MemoryManagementService)
    # 内部 redis 和 quota 都应被降级为 None（否则 __init__ 已 ValueError）
    assert svc._redis is None
    assert svc._user_daily_quota is None
