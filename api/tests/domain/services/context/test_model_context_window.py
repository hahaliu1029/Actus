from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.services.context.model_context_window import resolve_context_window


def test_resolve_context_window_prefers_explicit_config() -> None:
    config = ContextOverflowConfig(
        context_window=200000,
        unknown_model_context_window=32768,
    )

    assert resolve_context_window(model_name="gpt-4o", config=config) == 200000


def test_resolve_context_window_falls_back_to_model_map() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=32768,
    )

    assert resolve_context_window(model_name="gpt-4o", config=config) == 128000


def test_resolve_context_window_uses_unknown_model_fallback() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=65536,
    )

    assert (
        resolve_context_window(model_name="unknown-provider-model", config=config)
        == 65536
    )
