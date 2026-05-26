import os
from unittest.mock import patch
import pytest
from app.domain.services.coordinator_feature_flag import (
    is_coordinator_enabled, assert_coordinator_enabled,
)

class TestCoordinatorFeatureFlag:
    def test_default_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ACTUS_C2_COORDINATOR_ENABLED", None)
            assert is_coordinator_enabled() is False

    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("True", True), ("1", True), ("yes", True),
        ("false", False), ("0", False), ("", False), ("invalid", False),
    ])
    def test_env_override(self, value, expected):
        with patch.dict(os.environ, {"ACTUS_C2_COORDINATOR_ENABLED": value}):
            assert is_coordinator_enabled() is expected

    def test_assert_raises_when_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ACTUS_C2_COORDINATOR_ENABLED", None)
            with pytest.raises(RuntimeError, match="ACTUS_C2_COORDINATOR_ENABLED is false"):
                assert_coordinator_enabled()
