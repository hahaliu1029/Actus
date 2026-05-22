"""C3 PR-3c — Unit tests for ``SandboxLifecycleService.reconcile_orphans``
mailbox-supervisor recovery dual path (plan §11.3).

Verifies the new tail of ``reconcile_orphans`` (added in PR-3c at line
740+) without spinning up a real Postgres / DockerSandbox stack:

* Skipped entirely when ``supervisor_registry`` is ``None`` (legacy /
  pre-mailbox deployments — backwards compat).
* For each root id returned by ``find_running_mailbox_plane_root_ids``,
  ``spawn`` is called UNLESS ``health_check`` says the slot is already
  ``alive``, ``restarting``, or ``crashed`` (the per-pod restart loop
  owns crashed-slot recovery — see codex r1 [HIGH ARCH]).
* ``health_check`` is invoked exactly once per reconcile (not per root)
  so a partial restart loop tick doesn't race the spawn list.
* DB query failure on ``find_running_mailbox_plane_root_ids`` aborts the
  supervisor path safely (logs + returns) without crashing the whole
  reconcile.
* Per-root ``spawn`` failure is logged + the loop continues — one
  flaky root must not starve siblings.

Pure unit tests — no Postgres, no Docker, no Redis. The "DESTROYING"
and "CREATING" branches of ``reconcile_orphans`` are exercised
elsewhere (e.g. lifecycle PR-1 tests); these tests stub the
``uow.session.get_all`` call to return an empty list so we focus on
the new PR-3c tail.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.application.services.sandbox_lifecycle_service import (
    SandboxLifecycleService,
)


# ---- Minimal stubs ----------------------------------------------------------


class _StubSessionRepo:
    """Stand-in for ``uow.session`` covering only the methods reconcile
    invokes: ``get_all`` (existing branch — return []) and the new
    ``find_running_mailbox_plane_root_ids`` (PR-3c)."""

    def __init__(
        self,
        *,
        running_root_ids: list[str],
        raise_on_find: bool = False,
    ) -> None:
        self._running_root_ids = running_root_ids
        self._raise_on_find = raise_on_find
        self.find_calls = 0

    async def get_all(self) -> list[Any]:
        return []

    async def find_running_mailbox_plane_root_ids(self) -> list[str]:
        self.find_calls += 1
        if self._raise_on_find:
            raise RuntimeError("synthetic DB outage")
        return list(self._running_root_ids)


class _StubUow:
    def __init__(self, session_repo: _StubSessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self) -> "_StubUow":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _make_uow_factory(session_repo: _StubSessionRepo):
    def _factory() -> _StubUow:
        return _StubUow(session_repo)
    return _factory


class _StubSupervisorRegistry:
    """Minimal port surface — ``spawn`` + ``health_check``. Tracks every
    call so tests can assert order / counts.
    """

    def __init__(self, *, health: dict[str, str] | None = None) -> None:
        self._health = dict(health) if health else {}
        self.spawn_calls: list[str] = []
        self.health_check_calls = 0
        self.spawn_failure_for: set[str] = set()

    async def health_check(self) -> dict[str, str]:
        self.health_check_calls += 1
        return dict(self._health)

    async def spawn(self, root_id: str) -> None:
        self.spawn_calls.append(root_id)
        if root_id in self.spawn_failure_for:
            raise RuntimeError(f"synthetic spawn failure for {root_id}")
        self._health[root_id] = "alive"

    async def stop(self, root_id: str) -> None:
        self._health.pop(root_id, None)


@pytest.fixture(autouse=True)
def _single_worker(monkeypatch):
    """The lifecycle service ctor demands WEB_CONCURRENCY=1."""
    monkeypatch.setenv("WEB_CONCURRENCY", "1")


def _build_service(
    *,
    session_repo: _StubSessionRepo,
    supervisor_registry: _StubSupervisorRegistry | None,
) -> SandboxLifecycleService:
    """Construct the SUT with a stub UoW factory + minimal DockerSandbox
    placeholder. Sandbox class won't be touched because get_all() returns
    [] (no DESTROYING / CREATING sessions to handle)."""
    return SandboxLifecycleService(
        sandbox_cls=MagicMock(),
        uow_factory=_make_uow_factory(session_repo),
        supervisor_registry=supervisor_registry,
    )


# ---- Tests ------------------------------------------------------------------


@pytest.mark.anyio
class TestReconcileSupervisorRecoveryPath:
    """PR-3c plan §11.3 supervisor recovery tail (lines 740-790)."""

    async def test_registry_none_skips_supervisor_path_entirely(self) -> None:
        """Legacy deployments without a registry must observe NO change
        in behavior — the recovery tail is gated."""
        repo = _StubSessionRepo(running_root_ids=["r1", "r2"])
        svc = _build_service(session_repo=repo, supervisor_registry=None)

        await svc.reconcile_orphans()

        # The new DB query MUST NOT fire when registry is absent (avoids
        # a useless SELECT on every reconcile for pre-mailbox installs).
        assert repo.find_calls == 0

    async def test_spawns_for_each_running_root_when_health_empty(self) -> None:
        repo = _StubSessionRepo(running_root_ids=["root-a", "root-b", "root-c"])
        reg = _StubSupervisorRegistry(health={})
        svc = _build_service(session_repo=repo, supervisor_registry=reg)

        await svc.reconcile_orphans()

        assert sorted(reg.spawn_calls) == ["root-a", "root-b", "root-c"]
        # Single health snapshot per reconcile, not per root.
        assert reg.health_check_calls == 1

    async def test_skips_root_whose_slot_is_alive(self) -> None:
        repo = _StubSessionRepo(running_root_ids=["live", "dead"])
        reg = _StubSupervisorRegistry(health={"live": "alive"})
        svc = _build_service(session_repo=repo, supervisor_registry=reg)

        await svc.reconcile_orphans()

        assert reg.spawn_calls == ["dead"]

    async def test_skips_root_whose_slot_is_restarting(self) -> None:
        """``restarting`` is a future-reserved transient state per the
        ``SupervisorRegistryPort`` docstring; reconcile must respect it
        so a per-pod restart-loop tick mid-reconcile doesn't get
        double-spawned."""
        repo = _StubSessionRepo(running_root_ids=["x", "y"])
        reg = _StubSupervisorRegistry(health={"x": "restarting"})
        svc = _build_service(session_repo=repo, supervisor_registry=reg)

        await svc.reconcile_orphans()

        assert reg.spawn_calls == ["y"]

    async def test_skips_root_whose_slot_is_crashed(self) -> None:
        """codex r1 [HIGH ARCH] regression — ``crashed`` slots are owned by
        the per-pod ``_restart_loop`` (SupervisorRegistry §6.2). Reconcile
        must NOT call ``spawn`` on them because ``spawn`` is idempotent on
        existing slots (no-op when ``root in self._slots``) and would
        emit a misleading "ensured" log without actually resurrecting.
        """
        repo = _StubSessionRepo(running_root_ids=["dead", "fresh"])
        reg = _StubSupervisorRegistry(health={"dead": "crashed"})
        svc = _build_service(session_repo=repo, supervisor_registry=reg)

        await svc.reconcile_orphans()

        # crashed slot is left to the restart loop; only "fresh" is spawned.
        assert reg.spawn_calls == ["fresh"]

    async def test_db_query_failure_aborts_supervisor_path_safely(self) -> None:
        """A flaky DB MUST NOT crash the whole reconcile; supervisor
        recovery is best-effort. ``reconcile_orphans`` only runs at
        FastAPI lifespan startup, so a DB failure here means this pod
        has no supervisor recovery until the next pod restart re-runs
        ``reconcile_orphans`` (codex r3 doc fix — no in-process retry)."""
        repo = _StubSessionRepo(
            running_root_ids=["unused"], raise_on_find=True
        )
        reg = _StubSupervisorRegistry(health={})
        svc = _build_service(session_repo=repo, supervisor_registry=reg)

        # Must not raise.
        await svc.reconcile_orphans()

        # No spawns, no health check — bailed out cleanly.
        assert reg.spawn_calls == []
        assert reg.health_check_calls == 0

    async def test_per_root_spawn_failure_does_not_starve_siblings(self) -> None:
        """One flaky root must not skip the rest of the list."""
        repo = _StubSessionRepo(running_root_ids=["good-1", "bad", "good-2"])
        reg = _StubSupervisorRegistry(health={})
        reg.spawn_failure_for = {"bad"}
        svc = _build_service(session_repo=repo, supervisor_registry=reg)

        await svc.reconcile_orphans()

        # All three were attempted; the failure was swallowed.
        assert sorted(reg.spawn_calls) == ["bad", "good-1", "good-2"]


# ---- Wiring call-site enforcement -------------------------------------------
#
# codex r2 [HIGH TEST] — the T6 pod-restart integration test
# (test_mailbox_reconcile.py) auto-skips in PR-3c pending PR-4 fixtures,
# so the production wiring from ``reconcile_orphans`` → repo query +
# registry.spawn isn't CI-enforced via that file. AST scan locks the
# call-site contract structurally so future refactors can't silently
# remove the supervisor recovery dual-path.


import ast
import inspect
import pathlib

from app.application.services import sandbox_lifecycle_service as _slm


class TestReconcileOrphansWiringContract:
    """AST gates over ``reconcile_orphans`` body — the dual-path
    supervisor recovery must keep its load-bearing call sites."""

    def _reconcile_body_source(self) -> str:
        path = pathlib.Path(inspect.getsourcefile(_slm) or "")
        return path.read_text()

    def _find_reconcile(self, tree: ast.Module) -> ast.AsyncFunctionDef:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "SandboxLifecycleService":
                for item in node.body:
                    if (
                        isinstance(item, ast.AsyncFunctionDef)
                        and item.name == "reconcile_orphans"
                    ):
                        return item
        raise LookupError("SandboxLifecycleService.reconcile_orphans not found")

    def test_reconcile_calls_find_running_mailbox_plane_root_ids(self) -> None:
        tree = ast.parse(self._reconcile_body_source())
        reconcile = self._find_reconcile(tree)
        found = False
        for node in ast.walk(reconcile):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "find_running_mailbox_plane_root_ids"
            ):
                found = True
                break
        assert found, (
            "reconcile_orphans must call session.find_running_mailbox_plane_root_ids() "
            "to power the PR-3c supervisor recovery dual path"
        )

    @staticmethod
    def _receiver_is_self_supervisor_registry(call: ast.Call) -> bool:
        """codex r4 [MEDIUM TEST] — bind the receiver. Without this,
        any object's ``.spawn()`` / ``.health_check()`` would satisfy
        the gate (e.g. a future refactor that calls some_other.spawn).
        Check the call's ``func`` chain is exactly
        ``self._supervisor_registry.<method>``.
        """
        func = call.func
        if not isinstance(func, ast.Attribute):
            return False
        recv = func.value
        if not isinstance(recv, ast.Attribute):
            return False
        if recv.attr != "_supervisor_registry":
            return False
        if not isinstance(recv.value, ast.Name) or recv.value.id != "self":
            return False
        return True

    def test_reconcile_calls_supervisor_registry_spawn(self) -> None:
        tree = ast.parse(self._reconcile_body_source())
        reconcile = self._find_reconcile(tree)
        found = False
        for node in ast.walk(reconcile):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "spawn"
                and self._receiver_is_self_supervisor_registry(node)
            ):
                found = True
                break
        assert found, (
            "reconcile_orphans must call self._supervisor_registry.spawn(root_id) "
            "for missing slots — PR-3c plan §11.3 (codex r4: receiver MUST be "
            "self._supervisor_registry, not just any .spawn())"
        )

    def test_reconcile_calls_supervisor_registry_health_check(self) -> None:
        tree = ast.parse(self._reconcile_body_source())
        reconcile = self._find_reconcile(tree)
        found = False
        for node in ast.walk(reconcile):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "health_check"
                and self._receiver_is_self_supervisor_registry(node)
            ):
                found = True
                break
        assert found, (
            "reconcile_orphans must consult self._supervisor_registry.health_check() "
            "to skip alive/restarting/crashed roots — PR-3c plan §11.3 "
            "(codex r4: receiver MUST be self._supervisor_registry)"
        )
