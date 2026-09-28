from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.models.app_config import AppConfig, LLMConfig, AgentConfig, MCPConfig, A2AConfig, FileUnderstandingConfig, VisionFallbackConfig
from app.interfaces import service_dependencies as deps
from app.interfaces.endpoints import app_config_routes
from app.domain.services.provider_profiles import get_profile
from app.application.errors.exceptions import ValidationError
from app.interfaces.dependencies.auth import get_current_user


def test_vision_explicit_profile_and_response_format_reach_factory():
    cfg = AppConfig(
        llm_config=LLMConfig(base_url="https://api.openai.com/v1"),
        agent_config=AgentConfig(), mcp_config=MCPConfig(), a2a_config=A2AConfig(),
        file_understanding=FileUnderstandingConfig(vision_fallback=VisionFallbackConfig(
            enabled=True, base_url="https://proxy.example.test/v1", model_name="claude-sonnet-4-6",
            provider="anthropic_compat", supports_response_format=False, api_type="responses",
        )),
    )
    with patch.object(deps, "_build_skill_service", return_value=None):
        snapshot = deps._build_config_snapshot(cfg)
    model = snapshot.vision_fallback_model
    assert model.profile.provider_id == "anthropic_compat"
    assert model.supports_response_format is False
    assert model._llm_type == "actus-responses"


@pytest.mark.anyio
async def test_connection_probe_uses_production_factory_without_saving(monkeypatch):
    factory = Mock(return_value=SimpleNamespace(profile=get_profile("glm_5_2_coding"), ainvoke=AsyncMock(return_value=AIMessage(content="OK"))))
    monkeypatch.setattr(app_config_routes, "_build_llm", factory)
    service = SimpleNamespace(get_llm_config=AsyncMock(return_value=LLMConfig(api_key="private-test-key")), update_llm_config=AsyncMock())
    request = LLMConfig(base_url="https://example.test/v1", api_type="auto")
    result = await app_config_routes.test_llm_connection(request, Mock(), service)
    passed = factory.call_args.args[0]
    assert passed.api_key == "private-test-key"
    assert passed.api_type == "auto"
    assert passed.max_tokens == 256
    assert request.api_key == ""
    assert result.data.success is True
    assert factory.return_value.ainvoke.call_args.kwargs == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert "private-test-key" not in result.model_dump_json()
    service.update_llm_config.assert_not_called()


@pytest.mark.anyio
async def test_connection_probe_never_returns_provider_error_body(monkeypatch):
    fake = SimpleNamespace(profile=get_profile("generic_openai"), ainvoke=AsyncMock(side_effect=RuntimeError("key=private-test-key")))
    monkeypatch.setattr(app_config_routes, "_build_llm", lambda _: fake)
    request = LLMConfig(api_key="provided-test-key")
    result = await app_config_routes.test_llm_connection(request, Mock(), Mock())
    assert result.data.success is False
    assert "RuntimeError" in result.data.message
    assert "private-test-key" not in result.model_dump_json()


@pytest.mark.anyio
async def test_connection_probe_does_not_accept_reasoning_only_result(monkeypatch):
    fake = SimpleNamespace(profile=get_profile("glm_5_2"), ainvoke=AsyncMock(
        return_value=AIMessage(content="", additional_kwargs={"reasoning_content": "private reasoning"}),
    ))
    monkeypatch.setattr(app_config_routes, "_build_llm", lambda _: fake)
    result = await app_config_routes.test_llm_connection(LLMConfig(api_key="test-key"), Mock(), Mock())
    assert result.data.success is False
    assert "private reasoning" not in result.model_dump_json()


@pytest.mark.parametrize("logged_in,status_code", [(False, 401), (True, 403)])
def test_connection_probe_requires_admin(logged_in, status_code, monkeypatch):
    app = FastAPI()
    app.include_router(app_config_routes.router)
    app.dependency_overrides[app_config_routes.rate_limit_write] = lambda: None
    app.dependency_overrides[app_config_routes.get_app_config_service] = lambda: Mock()
    if logged_in:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(is_admin=lambda: False)
    factory = Mock()
    monkeypatch.setattr(app_config_routes, "_build_llm", factory)
    with TestClient(app) as client:
        response = client.post("/app-config/llm/test", json={"base_url": "https://example.test/v1"})
    assert response.status_code == status_code
    factory.assert_not_called()


@pytest.mark.anyio
async def test_unknown_probe_profile_is_validation_error_before_factory(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(app_config_routes, "_build_llm", factory)
    with pytest.raises(ValidationError) as exc:
        await app_config_routes.test_llm_connection(LLMConfig(provider="missing-profile"), Mock(), Mock())
    assert exc.value.code == 422
    factory.assert_not_called()
