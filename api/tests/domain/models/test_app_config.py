from app.domain.models.app_config import LLMConfig


def test_llm_config_has_context_overflow_default_values() -> None:
    config = LLMConfig()

    assert config.context_window is None
    assert config.context_overflow_guard_enabled is False
    assert config.overflow_retry_cap == 2
    assert config.soft_trigger_ratio == 0.85
    assert config.hard_trigger_ratio == 0.95
    assert config.reserved_output_tokens == 4096
    assert config.reserved_output_tokens_cap_ratio == 0.25
    assert config.token_estimator == "hybrid"
    assert config.token_safety_factor == 1.15
    assert config.unknown_model_context_window == 32768


def test_llm_config_accepts_context_overflow_custom_values() -> None:
    config = LLMConfig(
        context_window=131072,
        context_overflow_guard_enabled=True,
        overflow_retry_cap=1,
        soft_trigger_ratio=0.8,
        hard_trigger_ratio=0.9,
        reserved_output_tokens=2048,
        reserved_output_tokens_cap_ratio=0.2,
        token_estimator="char",
        token_safety_factor=1.05,
        unknown_model_context_window=65536,
    )

    assert config.context_window == 131072
    assert config.context_overflow_guard_enabled is True
    assert config.overflow_retry_cap == 1
    assert config.soft_trigger_ratio == 0.8
    assert config.hard_trigger_ratio == 0.9
    assert config.reserved_output_tokens == 2048
    assert config.reserved_output_tokens_cap_ratio == 0.2
    assert config.token_estimator == "char"
    assert config.token_safety_factor == 1.05
    assert config.unknown_model_context_window == 65536
