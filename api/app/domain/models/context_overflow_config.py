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
    model_name: str = ""
    tool_result_max_chars: int = Field(8000, ge=100)
    tool_compress_trigger_ratio: float = Field(0.75, gt=0, le=1)
    target_ratio: float = Field(0.65, gt=0, le=1)
    summary_max_chars: int = Field(16_000, ge=1000)
    system_prompt_max_tokens: int = Field(10000, ge=0)
    """B5 C9 / M2-PR0: system prompt token 预算上限（canonical config source）。

    Consumed by:
    - ``PromptAssembler`` via ``SystemPromptBudget(max_tokens=...)``
    - ``compute_effective_window()`` to derive the history window
      (``total - system_prompt_max_tokens - reserved_output_tokens``).

    M2-PR0 bumped the default from 3500 → 10000 to make room for the
    three memory sections (``memory_user_profile`` ~1500,
    ``memory_rules`` ~2500, ``memory_fact_index`` ~1000) alongside
    identity / behavior_core / output_format / tools_guide. Under
    budget pressure the existing priority-DESC drop loop still applies.
    """

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
            model_name=llm_config.model_name,
            tool_result_max_chars=llm_config.tool_result_max_chars,
            tool_compress_trigger_ratio=llm_config.tool_compress_trigger_ratio,
            system_prompt_max_tokens=llm_config.system_prompt_max_tokens,
        )
