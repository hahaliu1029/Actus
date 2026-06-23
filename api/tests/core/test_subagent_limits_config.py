"""S3 PR-2 / §4.4: max_subagent_depth ceiling is clamped to [1, 2]."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import SubagentLimitsConfig


def test_default_depth_unchanged():
    assert SubagentLimitsConfig().max_subagent_depth == 1


def test_depth_2_is_allowed():
    assert SubagentLimitsConfig(max_subagent_depth=2).max_subagent_depth == 2


def test_depth_3_rejected_at_load():
    with pytest.raises(ValidationError):
        SubagentLimitsConfig(max_subagent_depth=3)


def test_depth_8_rejected_at_load():
    # Previously le=8 tolerated this (then broke at spawn time); now it fails loud.
    with pytest.raises(ValidationError):
        SubagentLimitsConfig(max_subagent_depth=8)
