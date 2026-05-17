"""ToolConfirmationConfig gains 5 PE-0 feature flags, all default True."""

import pytest
from pydantic import ValidationError

from app.domain.models.app_config import ToolConfirmationConfig


def test_pe_flags_default_true():
    cfg = ToolConfirmationConfig()
    assert cfg.permission_engine_native_enabled is True
    assert cfg.permission_engine_skill_enabled is True
    assert cfg.permission_engine_mcp_enabled is True
    assert cfg.permission_engine_a2a_enabled is True
    assert cfg.permission_engine_a4_events_enabled is True


def test_pe_flags_can_be_disabled():
    cfg = ToolConfirmationConfig(
        permission_engine_native_enabled=False,
    )
    assert cfg.permission_engine_native_enabled is False


def test_pe_flags_unknown_field_rejected():
    # ToolConfirmationConfig uses extra='forbid', so unknown PE-style flags fail.
    # C-P2-1 path (a): no unknown keys found in config.yaml.example or config.yaml,
    # so extra="forbid" was added to ToolConfirmationConfig.
    with pytest.raises((ValidationError, ValueError)):
        ToolConfirmationConfig(permission_engine_bogus_enabled=False)
