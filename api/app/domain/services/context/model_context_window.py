from app.domain.models.context_overflow_config import ContextOverflowConfig

MODEL_CONTEXT_WINDOW_MAP: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1_048_576,
    "gpt-4.1-mini": 1_048_576,
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
}


def _normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    if "/" in name:
        name = name.split("/")[-1]
    return name


def resolve_context_window(model_name: str, config: ContextOverflowConfig) -> int:
    """解析上下文窗口：显式配置 > 模型映射 > 未知模型兜底。"""
    if config.context_window is not None:
        return config.context_window

    normalized = _normalize_model_name(model_name)
    mapped = MODEL_CONTEXT_WINDOW_MAP.get(normalized)
    if mapped is not None:
        return mapped

    for prefix, context_window in MODEL_CONTEXT_WINDOW_MAP.items():
        if normalized.startswith(prefix):
            return context_window

    return config.unknown_model_context_window
