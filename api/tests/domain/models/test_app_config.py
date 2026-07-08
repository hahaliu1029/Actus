import pytest
from pydantic import ValidationError

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
    MemoryConfig,
)


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


def test_agent_config_has_default_skill_selection_policy() -> None:
    config = AgentConfig()

    assert config.skill_selection.base_threshold == 3
    assert config.skill_selection.short_message_max_chars == 24
    assert config.skill_selection.llm_trigger_token_count == 4
    assert config.skill_selection.continuation_llm_enabled is True
    assert config.skill_selection.continuation_llm_timeout_seconds == 3.0
    assert "继续" in config.skill_selection.continuation_phrases
    assert config.skill_selection.continuation_patterns


def test_agent_config_has_new_stability_defaults() -> None:
    config = AgentConfig()
    policy = config.skill_selection

    assert policy.ask_user_min_attempt_rounds_per_step == 1
    assert policy.step_skill_lock_enabled is True
    assert policy.step_skill_reselect_unknown_tool_threshold == 3
    assert policy.step_skill_reselect_max_per_step == 1
    assert policy.available_tool_summary_token_budget == 500
    assert policy.unknown_tool_candidate_limit == 10


def test_agent_config_rejects_invalid_continuation_pattern() -> None:
    with pytest.raises(ValueError) as exc_info:
        AgentConfig(
            skill_selection={
                "continuation_patterns": [r"^(继续$"],
            }
        )

    error_message = str(exc_info.value)
    assert "continuation_patterns[0]" in error_message
    assert "^(继续$" in error_message


def test_agent_config_has_default_memory_config() -> None:
    config = AgentConfig()

    assert config.memory.summary_enabled is True
    assert config.memory.summary_model is None
    assert config.memory.summary_max_rounds == 5
    assert config.memory.summary_token_budget == 2000
    assert config.memory.summary_min_steps == 1
    assert config.memory.context_anchor_enabled is True
    assert config.memory.compact_keep_summary is True


def test_agent_config_accepts_custom_memory_config() -> None:
    from app.domain.models.app_config import MemoryConfig

    config = AgentConfig(
        memory=MemoryConfig(
            summary_enabled=False,
            summary_model="gpt-4o-mini",
            summary_max_rounds=3,
            summary_token_budget=1000,
            summary_min_steps=1,
            context_anchor_enabled=False,
            compact_keep_summary=False,
        )
    )

    assert config.memory.summary_enabled is False
    assert config.memory.summary_model == "gpt-4o-mini"
    assert config.memory.summary_max_rounds == 3


def test_app_config_accepts_legacy_bool_skill_risk_mode() -> None:
    config = AppConfig(
        llm_config=LLMConfig(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        skill_risk_policy={"mode": False},
    )

    assert config.skill_risk_policy.mode.value == "off"


class TestLLMConfigTimeoutSeconds:
    """D5.1: per-call LLM hard timeout field."""

    def test_default_value_is_120(self) -> None:
        cfg = LLMConfig()
        assert cfg.timeout_seconds == 120.0

    def test_explicit_value_accepted(self) -> None:
        cfg = LLMConfig(timeout_seconds=45.0)
        assert cfg.timeout_seconds == 45.0

    def test_zero_accepted_as_escape_hatch(self) -> None:
        cfg = LLMConfig(timeout_seconds=0.0)
        assert cfg.timeout_seconds == 0.0

    def test_max_3600_accepted(self) -> None:
        cfg = LLMConfig(timeout_seconds=3600.0)
        assert cfg.timeout_seconds == 3600.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=-1.0)

    def test_over_3600_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=3601.0)


class TestMemoryConfigSummaryTimeoutSeconds:
    """D5.1: summarizer LLM hard timeout override."""

    def test_default_value_is_30(self) -> None:
        cfg = MemoryConfig()
        assert cfg.summary_timeout_seconds == 30.0

    def test_none_accepted_for_inherit_semantics(self) -> None:
        cfg = MemoryConfig(summary_timeout_seconds=None)
        assert cfg.summary_timeout_seconds is None

    def test_zero_accepted(self) -> None:
        cfg = MemoryConfig(summary_timeout_seconds=0.0)
        assert cfg.summary_timeout_seconds == 0.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(summary_timeout_seconds=-1.0)

    def test_over_3600_raises(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(summary_timeout_seconds=3601.0)


def test_tool_display_metadata_enabled_defaults_off():
    """B10 D7: dark-launch flag 默认 OFF."""
    from app.domain.models.app_config import ToolRuntimeConfig

    assert ToolRuntimeConfig().tool_display_metadata_enabled is False


def test_agent_config_has_default_slash_commands_config() -> None:
    config = AgentConfig()

    assert config.slash_commands.enabled is False
    assert config.slash_commands.skill_commands_enabled is False
    assert config.slash_commands.manual_compaction_enabled is False


def test_agent_config_accepts_custom_slash_commands_config() -> None:
    from app.domain.models.app_config import SlashCommandsConfig

    config = AgentConfig(
        slash_commands=SlashCommandsConfig(
            enabled=True,
            skill_commands_enabled=True,
            manual_compaction_enabled=True,
        )
    )

    assert config.slash_commands.enabled is True
    assert config.slash_commands.skill_commands_enabled is True
    assert config.slash_commands.manual_compaction_enabled is True


def test_slash_commands_config_round_trips_through_agent_config_dict() -> None:
    # settings 保存 round-trip：非默认值经 dict 序列化后不丢（对齐 §10 INV-B11-1 传输链）
    from app.domain.models.app_config import SlashCommandsConfig

    config = AgentConfig(
        slash_commands=SlashCommandsConfig(enabled=True, manual_compaction_enabled=True)
    )
    rebuilt = AgentConfig(**config.model_dump())

    assert rebuilt.slash_commands.enabled is True
    assert rebuilt.slash_commands.manual_compaction_enabled is True
    assert rebuilt.slash_commands.skill_commands_enabled is False


def test_config_yaml_example_declares_slash_commands_flags() -> None:
    """§12 test 15: the shipped config.yaml.example must carry the three
    slash_commands flags under agent_config, all default-OFF, so a mis-indent or
    omission in the release template is caught by a test (the plan itself flags
    YAML indentation as a real ship risk — B8 R1). Red until Step 4 adds the YAML.
    """
    from pathlib import Path

    import yaml

    # test file: api/tests/domain/models/test_app_config.py → parents[3] == api/
    example = Path(__file__).resolve().parents[3] / "config.yaml.example"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    sc = data["agent_config"]["slash_commands"]
    assert sc["enabled"] is False
    assert sc["skill_commands_enabled"] is False
    assert sc["manual_compaction_enabled"] is False


def test_tool_runtime_config_has_b12_flags_default_off() -> None:
    from app.domain.models.app_config import ToolRuntimeConfig

    cfg = ToolRuntimeConfig()
    assert cfg.file_view_media_type_enabled is False
    assert cfg.file_view_provider_materialize_enabled is False
    assert cfg.file_view_image_cache_enabled is False
    assert cfg.pdf_page_parallel_enabled is False
    assert cfg.document_preview_enabled is False


def test_tool_runtime_config_b12_flags_json_round_trip() -> None:
    """flag 是纯 bool → 可进持久 AppConfig（YAML dump/load）。INV-B12-4。"""
    from app.domain.models.app_config import ToolRuntimeConfig

    cfg = ToolRuntimeConfig(
        file_view_media_type_enabled=True,
        document_preview_enabled=True,
    )
    dumped = cfg.model_dump(mode="json")
    assert dumped["file_view_media_type_enabled"] is True
    assert dumped["document_preview_enabled"] is True
    restored = ToolRuntimeConfig.model_validate(dumped)
    assert restored.file_view_media_type_enabled is True
    assert restored.document_preview_enabled is True
