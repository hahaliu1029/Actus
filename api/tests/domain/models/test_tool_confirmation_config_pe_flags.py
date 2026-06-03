"""PE-4c: the 4 per-source PermissionEngine flags are deleted.

After PE-4c there is exactly ONE confirmation master switch (``enabled``).
Constructing ToolConfirmationConfig with any retired per-source flag must
raise (extra='forbid').
"""

import pytest
from pydantic import ValidationError

from app.domain.models.app_config import ToolConfirmationConfig

_RETIRED_FLAGS = (
    "permission_engine_native_enabled",
    "permission_engine_skill_enabled",
    "permission_engine_mcp_enabled",
    "permission_engine_a2a_enabled",
    "permission_engine_a4_events_enabled",
)


def test_default_config_has_no_per_source_pe_flags():
    cfg = ToolConfirmationConfig()
    for flag in _RETIRED_FLAGS:
        assert not hasattr(cfg, flag), f"{flag} should be deleted in PE-4c"


@pytest.mark.parametrize("flag", _RETIRED_FLAGS)
def test_retired_pe_flag_kwargs_rejected(flag):
    # extra='forbid' → constructing with a retired flag raises.
    with pytest.raises((ValidationError, ValueError)):
        ToolConfirmationConfig(**{flag: False})


def test_master_switch_still_present():
    cfg = ToolConfirmationConfig()
    assert cfg.enabled is True
