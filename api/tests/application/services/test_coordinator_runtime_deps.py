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
