"""C7 PR2 — LifecycleRuntimeConfig default-OFF + AppConfig 挂载（spec §8/F20）。"""
from app.domain.models.app_config import AppConfig, LifecycleRuntimeConfig


def test_defaults_are_off():
    cfg = LifecycleRuntimeConfig()
    assert cfg.lifecycle_events_enabled is False
    assert cfg.lifecycle_subagent_events_enabled is False


def test_app_config_mounts_lifecycle_runtime():
    field = AppConfig.model_fields["lifecycle_runtime"]
    assert field.default_factory is LifecycleRuntimeConfig


def test_config_snapshot_carries_lifecycle_runtime():
    from app.application.services.agent_service import _ConfigSnapshot
    assert "lifecycle_runtime" in _ConfigSnapshot.__dataclass_fields__
