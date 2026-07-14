"""PR-9b-A — _NullCoordinatorRuntimeDeps null-object contract (INV-A10 support)."""
from __future__ import annotations

import pytest

from app.application.services.coordinator_runtime_deps import (
    _CoordinatorRuntimeDeps,
    _NullCoordinatorRuntimeDeps,
)


FIELD_NAMES = (
    "parallel_execution_subgraph",
    "session_service",
    "rehydrate_service",
    "child_runner_starter",
    "mailbox_publisher",
    "mailbox_subscriber",
    "envelope_factory",
    "orchestrator_factory",
    "terminal_waiter",
    "probe_quota",
    "coordinator_limits",
    "session_repository",
    "patch_reducer_service",
    "patch_applier_deps",
    "artifact_storage",
    "cost_rollup_service",
    "coordinator_envelope_store",
    "parent_sandbox_adapter_factory",
    "coordinator_metrics",  # [C2b budget D10] defaulted field — see test below
    "coordinator_metrics_recorder",  # [C2b rollout WS1b] 2nd defaulted tail field
    "coordinator_wait_guard_factory",
    "coordinator_liveness_service",
)


def test_null_deps_constructs_with_zero_args() -> None:
    deps = _NullCoordinatorRuntimeDeps()
    assert deps is not None


def test_null_deps_yields_none_for_every_field() -> None:
    deps = _NullCoordinatorRuntimeDeps()
    for name in FIELD_NAMES:
        assert getattr(deps, name) is None, f"{name} must be None on null deps"


def test_null_deps_is_isinstance_friendly_sentinel() -> None:
    a = _NullCoordinatorRuntimeDeps()
    b = _NullCoordinatorRuntimeDeps()
    assert isinstance(a, _NullCoordinatorRuntimeDeps)
    assert isinstance(b, _NullCoordinatorRuntimeDeps)


def test_real_deps_rejects_missing_field() -> None:
    with pytest.raises(TypeError):
        _CoordinatorRuntimeDeps()  # type: ignore[call-arg]


def test_real_deps_is_frozen() -> None:
    sentinels = {f: object() for f in FIELD_NAMES}
    deps = _CoordinatorRuntimeDeps(**sentinels)
    with pytest.raises(Exception):
        deps.session_service = object()  # type: ignore[misc]


def test_coordinator_metrics_field_defaults_to_none() -> None:
    """[C2b budget D10] pre-existing required-field construction stays valid."""
    sentinels = {f: object() for f in FIELD_NAMES if f != "coordinator_metrics"}
    deps = _CoordinatorRuntimeDeps(**sentinels)
    assert deps.coordinator_metrics is None


def test_coordinator_metrics_recorder_field_defaults_to_none() -> None:
    sentinels = {
        f: object()
        for f in FIELD_NAMES
        if f not in ("coordinator_metrics", "coordinator_metrics_recorder")
    }
    deps = _CoordinatorRuntimeDeps(**sentinels)
    assert deps.coordinator_metrics_recorder is None
