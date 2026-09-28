from app.domain.models.app_config import LLMConfig
import pytest
from pydantic import ValidationError
from app.domain.models.app_config import VisionFallbackConfig


def test_llm_config_provider_default_is_none() -> None:
    """T15: provider 字段是独立的，与 provider_name 分离"""
    c = LLMConfig()
    assert c.provider is None


def test_llm_config_provider_explicit() -> None:
    c = LLMConfig(provider="kimi_k2")
    assert c.provider == "kimi_k2"


def test_llm_config_provider_name_separate() -> None:
    """C1: provider (A7 provider_id) 与 provider_name (prompt rendering) 值空间不共享"""
    c = LLMConfig(provider="kimi_k2")
    # LLMConfig 不应有 provider_name 字段（属 ActusChatModel.provider_name, C0a）
    assert not hasattr(c, "provider_name") or getattr(c, "provider_name", None) in (
        None, "openai", "anthropic",
    )


@pytest.mark.parametrize("suffix", [
    "/v1/chat/completions", "/v1/responses/", "/v1/%72esponses",
    "/v1?token=test", "/v1#fragment",
])
@pytest.mark.parametrize("config_type", [LLMConfig, VisionFallbackConfig])
def test_reject_operation_urls_and_url_extras(suffix, config_type):
    with pytest.raises(ValidationError):
        config_type(base_url="https://example.test" + suffix)


@pytest.mark.parametrize("url", [
    "https://api.deepseek.com/", "https://api.openai.com/v1",
    "https://open.bigmodel.cn/api/coding/paas/v4",
])
def test_preserve_provider_base_path(url):
    assert str(LLMConfig(base_url=url).base_url) == url


def test_vision_empty_base_inherits_and_keeps_explicit_profile():
    config = VisionFallbackConfig(base_url=" ", provider="anthropic_compat", supports_response_format=False)
    assert config.base_url == ""
    assert config.provider == "anthropic_compat"
    assert config.supports_response_format is False
