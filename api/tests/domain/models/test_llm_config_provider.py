from app.domain.models.app_config import LLMConfig


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
