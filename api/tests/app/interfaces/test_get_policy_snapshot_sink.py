from __future__ import annotations

import app.interfaces.service_dependencies as sd
from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.external.policy_snapshot_sink import NoopPolicySnapshotSink
from app.infrastructure.external.safety.logging_policy_snapshot_sink import (
    LoggingPolicySnapshotSink,
)


def test_provider_returns_noop_when_off(monkeypatch):
    monkeypatch.setattr(sd.settings, "sandbox_policy_compiler_enabled", False, raising=False)
    assert isinstance(sd.get_policy_snapshot_sink(), NoopPolicySnapshotSink)


def test_provider_returns_logging_when_on(monkeypatch):
    monkeypatch.setattr(sd.settings, "sandbox_policy_compiler_enabled", True, raising=False)
    assert isinstance(sd.get_policy_snapshot_sink(), LoggingPolicySnapshotSink)


def test_lifecycle_service_defaults_to_noop_sink_and_disabled():
    svc = SandboxLifecycleService(sandbox_cls=object, uow_factory=lambda: None)
    assert isinstance(svc._policy_sink, NoopPolicySnapshotSink)
    assert svc._policy_snapshot_enabled is False  # C5a flag OFF by default (INV-0)


def test_lifecycle_service_stores_injected_sink_and_flag():
    sink = NoopPolicySnapshotSink()
    svc = SandboxLifecycleService(
        sandbox_cls=object, uow_factory=lambda: None, sink=sink, policy_snapshot_enabled=True,
    )
    assert svc._policy_sink is sink
    assert svc._policy_snapshot_enabled is True
