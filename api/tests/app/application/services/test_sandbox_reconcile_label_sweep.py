"""SPM Task 6 — reconcile_orphans DESTROYING-probe distinction + label-sweep.

Two behaviours are locked here:

1. **DESTROYING probe distinction** — reconcile now rehydrates a DESTROYING
   binding via ``get_strict`` (Task 5). ``get_strict`` returning ``None`` is a
   *definite* NotFound → finalize DESTROYED; raising
   ``SandboxDaemonUnreachable`` (Docker daemon blip) → keep DESTROYING for the
   next pass. Previously ``get`` collapsed both into ``None`` and could
   mis-finalize a live container as DESTROYED during an outage.

2. **Label-sweep** — after the CREATING→UNBOUND repair, reconcile enumerates
   platform-managed containers (``actus.session_id`` label) and removes any
   with no live binding that are past a 120s grace window. Fail-safe: any
   enumeration / DB error skips the whole round (rather miss an orphan than
   mis-delete a live container).

Async-runner note: this repo ships **pytest-anyio**, NOT pytest-asyncio (see
``tests/conftest.py`` + the sibling ``test_sandbox_provision_flight.py``). The
brief's illustrative ``@pytest.mark.asyncio`` markers would leave the coroutine
un-awaited under this repo, so the module-level ``pytestmark =
pytest.mark.anyio`` is authoritative and the per-method markers are omitted.

The ``svc_and_fakes`` fixture + the whole fake facade (extended in Task 6 with
``get_strict`` / ``list_managed_containers`` / ``remove_container`` +
``FakeSessionRepo.get_all`` + ``FakeRegistry`` teardown no-ops) is imported
from the shared flight module — same-directory sibling import (pytest prepend
mode). ``anyio_backend`` comes from the global ``tests/conftest.py``.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.errors.sandbox_lifecycle import (
    SandboxDaemonUnreachable,
    SandboxLifecycleError,
)
from app.domain.models.session import DestroyReason, SandboxBindingState

# Shared facade + fixture (Task 6-extended). noqa: re-exported for pytest.
from test_sandbox_provision_flight import svc_and_fakes  # noqa: F401

pytestmark = pytest.mark.anyio

ACTIVE = SandboxBindingState.ACTIVE
SUSPENDED = SandboxBindingState.SUSPENDED
CREATING = SandboxBindingState.CREATING
DESTROYING = SandboxBindingState.DESTROYING
DESTROYED = SandboxBindingState.DESTROYED
UNBOUND = SandboxBindingState.UNBOUND


class TestDestroyingProbeDistinction:
    """场景⑧: reconcile 的 DESTROYING 探测用 get_strict 区分 daemon-unreachable
    (保持 DESTROYING) vs terminal/NotFound (finalize DESTROYED)。"""

    async def test_daemon_unreachable_keeps_destroying(self, svc_and_fakes):
        """daemon 不可达 → DESTROYING 保持，不误写 DESTROYED。"""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=DESTROYING, sandbox_id="sb-1")
        fakes.sandbox_cls.get_strict_raises = SandboxDaemonUnreachable("down")

        await svc.reconcile_orphans()

        # get_strict must have been the probe (not the legacy ambiguous get).
        assert fakes.sandbox_cls.get_strict_calls == 1
        assert fakes.binding_state(fakes.session_id) == DESTROYING

    async def test_not_found_finalizes_destroyed(self, svc_and_fakes):
        """get_strict → None (确定 NotFound) → finalize DESTROYED。

        FIX-C (P1-2 hardening): a strict-probe ``None`` also covers a container
        that EXISTS in a non-running (exited/paused) state, so reconcile now
        physically removes ``binding.id`` before finalizing DESTROYED instead of
        leaving it on disk until the next startup label sweep."""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=DESTROYING, sandbox_id="sb-1")
        fakes.sandbox_cls.get_strict_returns = None

        await svc.reconcile_orphans()

        assert fakes.sandbox_cls.get_strict_calls == 1
        assert fakes.binding_state(fakes.session_id) == DESTROYED
        assert "sb-1" in fakes.sandbox_cls.removed  # FIX-C: removed before finalize


class TestDestroyRetryProbeDistinction:
    """冻结决策2: destroy() 的 DESTROYING-retry 分支同样用 get_strict 区分——
    registry miss + binding.id 存在时:
      * get_strict → None (确定 NotFound) → 短路 terminal-success DESTROYED
      * get_strict → SandboxDaemonUnreachable → raise (retryable), 保持 DESTROYING

    (legacy get-only 回退路径由 tests/domain/services/test_sandbox_lifecycle_service.py
    的 test_destroy_*_on_destroying_retry_* 覆盖 — 那里的 fake 无 get_strict。)
    """

    async def test_destroy_retry_get_strict_notfound_finalizes_destroyed(
        self, svc_and_fakes
    ):
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=DESTROYING, sandbox_id="sb-1")  # registry empty → miss
        fakes.sandbox_cls.get_strict_returns = None

        # No raise — terminal success.
        await svc.destroy(fakes.session_id, reason=DestroyReason.WATCHDOG_TIMEOUT)

        assert fakes.sandbox_cls.get_strict_calls == 1
        assert fakes.binding_state(fakes.session_id) == DESTROYED
        assert "sb-1" in fakes.sandbox_cls.removed  # FIX-C: removed before finalize

    async def test_destroy_retry_daemon_unreachable_raises_keeps_destroying(
        self, svc_and_fakes
    ):
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=DESTROYING, sandbox_id="sb-1")
        fakes.sandbox_cls.get_strict_raises = SandboxDaemonUnreachable("down")

        with pytest.raises(SandboxLifecycleError):
            await svc.destroy(fakes.session_id, reason=DestroyReason.WATCHDOG_TIMEOUT)

        assert fakes.sandbox_cls.get_strict_calls == 1
        assert fakes.binding_state(fakes.session_id) == DESTROYING


class TestLabelSweep:
    """label-sweep: 无主 + 过宽限窗容器清扫，谓词 / 顺序 / fail-safe。"""

    async def test_orphan_container_past_grace_removed(self, svc_and_fakes):
        """无对应 binding + 过 120s 宽限窗 → 删除。"""
        svc, fakes = svc_and_fakes  # DB 无对应 binding (session_id="ghost")
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-orphan",
                "session_id": "ghost",
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=300),
            },
        ]

        await svc.reconcile_orphans()

        assert "sb-orphan" in fakes.sandbox_cls.removed

    async def test_recent_container_in_grace_kept(self, svc_and_fakes):
        """无主但仍在宽限窗内 → 保留 (宁漏勿误删刚起的容器)。"""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-new",
                "session_id": "ghost",
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=30),
            },
        ]

        await svc.reconcile_orphans()

        assert fakes.sandbox_cls.removed == []

    async def test_live_binding_container_kept(self, svc_and_fakes):
        """有 ACTIVE binding 且 id==name → 保留 (是活容器)。"""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=ACTIVE, sandbox_id="sb-live")
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-live",
                "session_id": fakes.session_id,
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=999),
            },
        ]

        await svc.reconcile_orphans()

        assert fakes.sandbox_cls.removed == []

    async def test_suspended_binding_container_kept(self, svc_and_fakes):
        """SUSPENDED 也在 KEEP 谓词内 → 保留 (可 resume 的容器不能扫掉)。"""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=SUSPENDED, sandbox_id="sb-susp")
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-susp",
                "session_id": fakes.session_id,
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=999),
            },
        ]

        await svc.reconcile_orphans()

        assert fakes.sandbox_cls.removed == []

    async def test_binding_id_mismatch_removed(self, svc_and_fakes):
        """binding 存在但 id != 容器名 (换代/残留) 且过宽限窗 → 删除。"""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=ACTIVE, sandbox_id="sb-current")
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-stale",  # != binding.id "sb-current"
                "session_id": fakes.session_id,
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=300),
            },
        ]

        await svc.reconcile_orphans()

        assert "sb-stale" in fakes.sandbox_cls.removed

    async def test_docker_list_failure_skips_sweep(self, svc_and_fakes):
        """fail-safe: 枚举抛异常 → 整轮跳过 (不抛 + 不删)。"""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.list_raises = RuntimeError("daemon down")

        await svc.reconcile_orphans()  # 不抛

        assert fakes.sandbox_cls.removed == []

    async def test_sweep_runs_after_creating_repair(self, svc_and_fakes):
        """场景⑨: restart-mid-create → CREATING repair 先行 (binding 修好)，
        再按谓词清扫无主容器 (binding.id=None != 容器名 → 删)。"""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=CREATING, sandbox_id=None)
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-halfway",
                "session_id": fakes.session_id,
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=300),
            },
        ]

        await svc.reconcile_orphans()

        assert fakes.binding_state(fakes.session_id) == UNBOUND       # repair 先行
        assert "sb-halfway" in fakes.sandbox_cls.removed              # 清扫 (id=None≠name)

    async def test_destroying_binding_container_kept(self, svc_and_fakes):
        """FIX-F(a): DESTROYING 亦在 sweep 的 KEEP 谓词内
        (ACTIVE/SUSPENDED/DESTROYING) —— 正在拆除中且 binding.id==name 的容器不能
        被 sweep 抢删。直接调用 ``_sweep_orphan_containers`` 以隔离谓词：若走
        ``reconcile_orphans``，DESTROYING 分支会先把该 binding finalize 掉，谓词就
        再也观察不到 DESTROYING 态 (这正是本用例要锁住的谓词分支)。"""
        svc, fakes = svc_and_fakes
        fakes.make_binding(state=DESTROYING, sandbox_id="sb-destroying")
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-destroying",
                "session_id": fakes.session_id,
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=999),
            },
        ]

        await svc._sweep_orphan_containers()

        assert fakes.sandbox_cls.removed == []

    async def test_sweep_get_all_failure_skips_safely(self, svc_and_fakes):
        """FIX-F(b): DB fail-safe —— 若 sweep 自己查 ``get_all`` 抛错，sweep 内部
        兜底 (log + return) 跳过整轮 (零删除)，且异常不冒泡打断 reconcile。用
        ``get_all_raise_on_call=2`` 让 reconcile 顶层 get_all (#1) 正常、仅 sweep 的
        get_all (#2) 抛错，证明是 sweep 的兜底生效而非 reconcile 提前 bail-out。"""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.managed = [
            {
                "name": "sb-orphan",
                "session_id": "ghost",
                "attempt": "a",
                "created_at": datetime.now(UTC) - timedelta(seconds=300),
            },
        ]
        fakes.uow.get_all_raises = RuntimeError("db down")
        fakes.uow.get_all_raise_on_call = 2  # reconcile body #1 OK, sweep #2 raises

        await svc.reconcile_orphans()  # 不抛

        assert fakes.sandbox_cls.removed == []
