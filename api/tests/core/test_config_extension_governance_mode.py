from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from app.domain.models.extension_governance import GovernanceMode
from core.config import Settings


def test_default_off():
    assert Settings(env="test").extension_governance_mode == "off"


def test_kwarg_shadow_enforce():
    assert Settings(env="test", extension_governance_mode="shadow").extension_governance_mode == "shadow"
    assert Settings(env="test", extension_governance_mode="enforce").extension_governance_mode == "enforce"


def test_env_var_mapping(monkeypatch):
    monkeypatch.setenv("EXTENSION_GOVERNANCE_MODE", "shadow")
    assert Settings(env="test").extension_governance_mode == "shadow"


def test_invalid_value_rejected():
    with pytest.raises(ValidationError):
        Settings(env="test", extension_governance_mode="on")


def test_validator_allowed_set_matches_domain_vocabulary():
    """INV-D1-1 缝合（spec §2 R13#4）：core 不 import domain——
    validator 允许值集 == domain GovernanceMode 词表由本测试在测试层缝合。"""
    assert Settings.EXTENSION_GOVERNANCE_MODE_ALLOWED == set(get_args(GovernanceMode))
