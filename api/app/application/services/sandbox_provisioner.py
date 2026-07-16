"""SPM PR-1c: the per-run on_demand ``SandboxProvisioner`` state machine.

``on_demand`` mode defers container creation until the FIRST sandbox-facing tool
call (or an explicit VNC / takeover trigger). This module owns the provisioner
that bridges the runner/accessor layer to the lifecycle service's
``acquire`` / ``bind_new`` flight (PR-1a hardened): it is a small state machine
with single-flight de-duplication + last-waiter-cancel propagation.

State machine (spec §5.2):

    unprovisioned ──get()──▶ provisioning ──▶ hooks ──▶ ready
          ▲                      │              │
          └── create/ready fail ─┘              │ hooks fail
          ▲                                     ▼
          └───────── (retryable) ────────── hooks_failed
                                             (handle RETAINED; next get()
                                              skips create/ready, reruns hooks)

Key correctness contracts (30-round frozen; do NOT simplify away):

* **Single-flight + last-waiter-cancel (DD-17):** ``get()`` awaits
  ``asyncio.shield(inflight)`` with a waiter count. A *single* waiter's cancel
  does NOT kill the shared flight; the *last* waiter's cancel DOES
  (``inflight.cancel()``) so cancellation propagates into ``bind_new`` → PR-1a
  BaseException rollback + container cleanup. There is deliberately NO
  whole-provisioner shield (spec explicitly rejected it — it would sever
  run-stop / watchdog cancellation and leak a container for an already-cancelled
  run).
* **Error taxonomy (§5.2d):** ``SessionSuspendedError`` / ``SessionFinalizedError``
  pass through UNTOUCHED (never wrapped; INV-SPM-9: NEVER call
  ``lifecycle.resume()``). ``SandboxProvisionInvalidated`` (delete/destroy
  终局) → translated to ``SessionFinalizedError`` (tool-face
  ``SANDBOX_FINALIZED``, retryable=False), NOT a retryable
  ``SANDBOX_PROVISION_FAILED``. ``CancelledError`` → reset + emit cancelled +
  re-raise. ``TimeoutError`` (create/ready/hooks-phase ``asyncio.timeout``) →
  typed ``SandboxProvisionError(cause="timeout")``. Generic → typed error with
  ``cause=repr(e)``.
* **Post-release semantics are EXPLICIT** (PR-1b whole-PR audit carry-forward):
  ``release_held_handle()`` sets ``state="unprovisioned"`` and clears the handle,
  so a subsequent ``get()`` RE-PROVISIONS cleanly — it does NOT inherit the Eager
  accessor's silent-None-after-release.

INV-SPM-2: NEW code only — no existing behavior is altered.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from app.domain.errors.sandbox_lifecycle import (
    SandboxProvisionError,
    SandboxProvisionInvalidated,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)

if TYPE_CHECKING:
    from app.domain.external.sandbox import SandboxHandle

logger = logging.getLogger(__name__)


# A post-provision hook runs against the ready handle (skill bundle sync,
# attachment flush, etc.). Hooks are (trigger_label, hook) tuples so a hook
# failure carries its OWN trigger into the error object + metric.
ProvisionHook = Callable[["SandboxHandle"], Awaitable[None]]


class ProvisionIdleGuard(Protocol):
    """Idle-watchdog suppression handle bound at flow time (Task 18 wires the
    real watchdog). ``pause`` is called before a provision attempt and ``resume``
    after it, so a slow container cold-start does not trip an idle timeout."""

    def pause(self) -> None: ...

    def resume(self) -> None: ...


class _WatchdogIdleGuard:
    """Task 18 (INV-SPM-13): adapts a flow-owned ``ExecutionWatchdog`` to the
    ``ProvisionIdleGuard`` Protocol so the on_demand provisioner can pause/resume
    idle evaluation around the WHOLE provision attempt (create + hooks).

    The watchdog is duck-typed via its multi-owner ``pause_idle(key)`` /
    ``resume_idle(key)`` set API — the application layer never imports the domain
    ``ExecutionWatchdog`` type. ``key`` is a structured owner so overlapping runs
    cannot resume one another (spec: ``("sandbox_provision", session_id)``)."""

    __slots__ = ("_watchdog", "_key")

    def __init__(self, watchdog: object, key: object) -> None:
        self._watchdog = watchdog
        self._key = key

    def pause(self) -> None:
        self._watchdog.pause_idle(self._key)  # type: ignore[attr-defined]

    def resume(self) -> None:
        self._watchdog.resume_idle(self._key)  # type: ignore[attr-defined]


class SandboxProvisionMetrics(Protocol):
    """Injectable metrics sink for on_demand provision-flow observability
    (``sandbox_provision_total{mode,trigger,outcome}`` +
    ``sandbox_provision_latency_seconds``; spec §5.9). Task 17 wires the real
    impl; here the provisioner accepts ``metrics=None`` (no-op) and tests inject
    a recording fake. Side-effect-only; implementations must never raise.

    Note (§5.9): this observes the provision FLOW (trigger / outcome / latency);
    it is NOT a physical container-creation counter. "Containers saved" is
    derived instead from the ``sandbox_lifecycle_log`` CREATING→ACTIVE rows.
    """

    def record_provision(
        self, *, mode: str, trigger: str, outcome: str, latency: float | None
    ) -> None: ...


class _SandboxLifecyclePort(Protocol):
    """The slice of ``SandboxLifecycleService`` the provisioner consumes.

    Structural — both the real service and the test fakes satisfy it. The
    provisioner NEVER calls ``resume()`` (INV-SPM-9), so it is not on this port.
    """

    async def acquire(self, session_id: str) -> "SandboxHandle": ...

    async def bind_new(
        self, session_id: str, *, user_id: str | None = None
    ) -> "SandboxHandle": ...


class SandboxProvisioner:
    """Per-run on_demand 供给状态机（spec §5.2）。

    状态：unprovisioned → provisioning → hooks → ready；create/ready 失败回
    unprovisioned；hooks 失败 → hooks_failed（handle 保留，下次 get() 只重试 hooks）。
    永不调用 lifecycle.resume()（INV-SPM-9）。
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        lifecycle: _SandboxLifecyclePort,
        hooks: list[tuple[str, ProvisionHook]] | None = None,
        timeout_seconds: float,
        trigger: str = "tool_call",
        metrics: SandboxProvisionMetrics | None = None,
        idle_guard: ProvisionIdleGuard | None = None,
    ) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._lifecycle = lifecycle
        self._hooks: list[tuple[str, ProvisionHook]] = list(hooks) if hooks else []
        self._timeout_seconds = timeout_seconds
        self._trigger = trigger
        self._metrics = metrics
        self._idle_guard = idle_guard

        self._state = "unprovisioned"
        self._handle: "SandboxHandle | None" = None
        self._inflight: "asyncio.Task[SandboxHandle] | None" = None
        self._waiters = 0
        self._attempt: str | None = None

    # ── introspection ──

    @property
    def state(self) -> str:
        return self._state

    def peek(self) -> "SandboxHandle | None":
        """非供给探针：仅 ready 时返回缓存 handle，否则 None。永不触发创建。"""
        if self._state == "ready":
            return self._handle
        return None

    # ── wiring ──

    def add_hook(self, hook: ProvisionHook, *, trigger: str = "tool_call") -> None:
        self._hooks.append((trigger, hook))

    def bind_idle_guard(self, guard: ProvisionIdleGuard) -> None:
        """flow 回调绑定（Task 18）——把真 watchdog 句柄挂上供给的 pause/resume。"""
        self._idle_guard = guard

    def mark_hooks_dirty(self) -> None:
        """runner 增量 flush 失败时调用（codex planR1#9①）：ready → hooks_failed，
        下次 get() 重跑 hooks（全量幂等重传兜住漏传文件）。非 ready 态 no-op。"""
        if self._state == "ready":
            self._state = "hooks_failed"

    def release_held_handle(self) -> None:
        """r7/codex R6-F3：owner-level 释放——runner destroy 经 accessor.release_owned 调用。
        覆盖 ready 收尾 / hooks_failed（peek()=None 但持 handle）/ mark_hooks_dirty 后。幂等。
        释放后 state=unprovisioned，下次 get() 干净重供给（显式后释放语义，非静默 None）。"""
        if self._handle is not None:
            try:
                self._handle.release()
            except Exception:
                logger.warning("release_held_handle failed", exc_info=True)
            self._handle = None
        self._state = "unprovisioned"

    # ── provisioning ──

    async def get(self) -> "SandboxHandle":
        if self._state == "ready":
            return self._handle  # type: ignore[return-value]
        if self._inflight is None:
            self._inflight = asyncio.create_task(self._provision_once())
            self._inflight.add_done_callback(self._clear_inflight)
        # 单飞 + last-waiter-cancel（codex planR1#2：整体 shield 会隔断 run-stop/watchdog 的
        # 取消传导，已取消的 run 仍会建容器并提交 ACTIVE——spec DD-17 明确否决 provisioner shield）：
        # 单个 waiter 取消不炸共乘者；**最后一个** waiter 取消时把取消传进供给任务 →
        # CancelledError 传导入 bind_new → PR-1a BaseException 回滚 + 容器清理。
        inflight = self._inflight
        self._waiters += 1
        try:
            return await asyncio.shield(inflight)
        except asyncio.CancelledError:
            if self._waiters == 1 and not inflight.done():
                inflight.cancel()
            raise
        finally:
            self._waiters -= 1

    def _clear_inflight(self, task: "asyncio.Task[SandboxHandle]") -> None:
        # Done-callback: drop the finished flight so the next get() (e.g. a
        # hooks_failed retry) starts a fresh provision. Do NOT touch task
        # result/exception here — the awaiting shield already retrieves it, and a
        # cancelled task has nothing to retrieve.
        self._inflight = None

    async def _provision_once(self) -> "SandboxHandle":
        self._guard_pause()  # idle 抑制（无 guard 则 no-op）
        phase = "create"
        trigger = self._trigger  # hook 段按 hook 自带标签覆写（如 skill_sync）
        self._attempt = uuid.uuid4().hex  # per-provision attempt（错误合同 §5.2d 的 attempt 字段；
        t0 = time.monotonic()  #   provisioner 侧供给尝试 id，独立于 lifecycle flight.attempt）
        try:
            async with asyncio.timeout(self._timeout_seconds):
                handle = self._handle  # hooks_failed 保留的 handle
                if handle is None:
                    try:
                        handle = await self._lifecycle.acquire(self._session_id)
                        self._handle = handle  # r7/F3：先赋值再 ensure——否则 ensure 失败时
                        phase = "ready"  # acquire 拿到的 handle（已入 registry）无释放入口
                        await handle.ensure_sandbox()  # acquire-hit 显式 readiness（spec §5.2）
                    except SessionUnboundError:
                        phase = "create"
                        handle = await self._lifecycle.bind_new(
                            self._session_id, user_id=self._user_id
                        )
                        self._handle = handle
                    # SessionSuspendedError / SessionFinalizedError：不捕获 → 原样透传（此前未赋值 self._handle，
                    #   acquire 尚未返回或抛于 acquire 内部，无泄漏）
                self._state = "hooks"
                phase = "hooks"
                for hook_trigger, hook in self._hooks:  # hooks = list[tuple[str, ProvisionHook]]
                    trigger = hook_trigger  # 失败时错误对象与 metric 都用真实 trigger
                    await hook(handle)  #（codex planR1#15：skill hook 失败 →
                    #  trigger="skill_sync" 进错误对象，非仅 metric）
                    # r21/codex R21-U1：每 hook 返回后兜底——hook 内 UoW commit 可能吞掉取消
                    #   （db_uow.py:71 不 uncancel）致 hook「成功」返回；置 ready/emit ok 前 honor
                    #   被吞的取消（走下方 CancelledError 分支 emit cancelled；binding 已 ACTIVE 且
                    #   合法，留 c-1 lazy-rehydrate/teardown，always-parity）：
                    if (_t := asyncio.current_task()) is not None and _t.cancelling() > 0:
                        raise asyncio.CancelledError()
                trigger = self._trigger
            if self._handle is None:
                # FIX-M3: a concurrent release_held_handle() cleared the handle
                # during the flight tail; publishing state="ready" with a None
                # handle would poison peek()/get(). Treat as cancelled-under-release
                # (unreachable in shipped single-caller wiring) — the CancelledError
                # branch below resets + emits once, and the next get() re-provisions.
                raise asyncio.CancelledError()
            self._state = "ready"
            self._emit(trigger=self._trigger, outcome="ok", latency=time.monotonic() - t0)
            return handle
        except (SessionSuspendedError, SessionFinalizedError):
            self._state = "unprovisioned"
            self._handle = None
            self._emit(trigger=trigger, outcome="failed")
            raise  # 透传（§5.2d）
        except SandboxProvisionInvalidated as e:
            # r3（codex planR2 B-P1-4 / spec R10①）：delete/destroy 终局不是「供给自身失败」——
            # 不得落 except Exception 变 retryable 的 SANDBOX_PROVISION_FAILED。
            # 转译为 SessionFinalizedError → 工具 wrapper 既有映射 SANDBOX_FINALIZED(retryable=False)。
            self._state = "unprovisioned"
            self._handle = None
            self._emit(trigger=trigger, outcome="cancelled")
            raise SessionFinalizedError(self._session_id, destroyed_at=None) from e
        except asyncio.CancelledError:
            self._reset_for_phase(phase)
            self._emit(trigger=trigger, outcome="cancelled")
            raise
        except TimeoutError as e:
            self._reset_for_phase(phase)
            self._emit(trigger=trigger, outcome="failed")
            raise SandboxProvisionError(
                self._session_id,
                phase=phase,
                trigger=trigger,
                attempt=self._attempt,
                cause="timeout",
            ) from e
        except SandboxProvisionError:
            self._reset_for_phase(phase)
            self._emit(trigger=trigger, outcome="failed")
            raise
        except Exception as e:
            self._reset_for_phase(phase)
            self._emit(trigger=trigger, outcome="failed")
            raise SandboxProvisionError(
                self._session_id,
                phase=phase,
                trigger=trigger,
                attempt=self._attempt,
                cause=repr(e),
            ) from e
        finally:
            self._guard_resume()

    def _reset_for_phase(self, phase: str) -> None:
        if phase == "hooks":
            self._state = "hooks_failed"  # 沙箱已 ready，handle 保留只重试 hooks
        else:  # create/ready 失败：不再持有 handle
            self._state = "unprovisioned"
            if self._handle is not None:  # r7/F3：ready 段（acquire-hit ensure-fail）self._handle
                try:
                    self._handle.release()  # 已赋值 → best-effort 释放防 registry 泄漏
                except Exception:
                    logger.warning(
                        "provisioner handle release on reset failed", exc_info=True
                    )
                self._handle = None

    # ── side channels ──

    def _guard_pause(self) -> None:
        if self._idle_guard is not None:
            self._idle_guard.pause()

    def _guard_resume(self) -> None:
        if self._idle_guard is not None:
            self._idle_guard.resume()

    def _emit(self, *, trigger: str, outcome: str, latency: float | None = None) -> None:
        """Record a provision-flow metric. No-op when no metrics port is injected
        (Task 17 wires the real impl). Side-effect-only; a metrics hiccup must
        never propagate — especially not from the exception handlers where it
        would mask the real provisioning error."""
        if self._metrics is None:
            return
        try:
            self._metrics.record_provision(
                mode="on_demand", trigger=trigger, outcome=outcome, latency=latency
            )
        except Exception:
            logger.warning("provision metrics emit failed", exc_info=True)
