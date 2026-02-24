from typing import Literal

from pydantic import BaseModel, Field

from app.domain.models.app_config import LLMConfig


class ContextOverflowConfig(BaseModel):
    """上下文超限治理配置（从LLM配置投影而来）"""

    context_window: int | None = Field(default=None, ge=1024)
    context_overflow_guard_enabled: bool = False
    overflow_retry_cap: int = Field(2, ge=0, le=10)
    soft_trigger_ratio: float = Field(0.85, gt=0, le=1)
    hard_trigger_ratio: float = Field(0.95, gt=0, le=1)
    reserved_output_tokens: int = Field(4096, ge=0)
    reserved_output_tokens_cap_ratio: float = Field(0.25, gt=0, le=1)
    token_estimator: Literal["hybrid", "char", "provider_api"] = "hybrid"
    token_safety_factor: float = Field(1.15, ge=1.0)
    unknown_model_context_window: int = Field(32768, ge=1024)

    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig) -> "ContextOverflowConfig":
        """从LLM配置构建治理参数，避免Agent层直接依赖LLMConfig。"""
        return cls(
            context_window=llm_config.context_window,
            context_overflow_guard_enabled=llm_config.context_overflow_guard_enabled,
            overflow_retry_cap=llm_config.overflow_retry_cap,
            soft_trigger_ratio=llm_config.soft_trigger_ratio,
            hard_trigger_ratio=llm_config.hard_trigger_ratio,
            reserved_output_tokens=llm_config.reserved_output_tokens,
            reserved_output_tokens_cap_ratio=llm_config.reserved_output_tokens_cap_ratio,
            token_estimator=llm_config.token_estimator,
            token_safety_factor=llm_config.token_safety_factor,
            unknown_model_context_window=llm_config.unknown_model_context_window,
        )
