"""SPM PR-2 Task 23 — service-level ``on_demand`` flow-group tests (spec §9).

Drives the **REAL** ``SandboxLifecycleService`` + **REAL** ``SandboxProvisioner``
+ **REAL** ``SandboxAttachmentFlusher`` over the established fakes:

* fake sandbox *type* + fake lifecycle UoW / registry — reused from the flight
  facade (``test_sandbox_provision_flight``);
* fake file storage + flusher UoW — reused from the attachment-flush test
  (``test_sandbox_attachment_flush``).

The four flows are the user-visible ``on_demand`` behaviors from spec §9:

1. VNC-first then tool → a SINGLE container provision + attachments visible;
2. a suspended session's stale tool call neither ``resume()``s nor creates;
3. an attachment survives a run restart (fresh provisioner/flusher → full
   idempotent retransfer);
4. the "successful sandbox activation count" = the ``sandbox_lifecycle_log``
   ``old_state="creating" AND new_state="active"`` rows (the authoritative
   single-writer signal; §5.9), across six sub-scenarios.

**VNC path is simulated at the lifecycle level** (``bind_new`` == the VNC-first
direct build; the brief explicitly allows this — wiring the real
``SessionService`` is disproportionate for these service-level flows).

Async runner note: this repo ships **pytest-anyio**, not pytest-asyncio (see
``tests/conftest.py``). The module-level ``pytestmark = pytest.mark.anyio`` + a
local ``anyio_backend`` fixture are authoritative; the brief's illustrative
``@pytest.mark.asyncio`` decorators are intentionally dropped (a bare ``asyncio``
marker under this plugin set would leave the coroutine un-awaited → false-green).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Optional

import pytest

from app.application.services.sandbox_attachment_flush import SandboxAttachmentFlusher
from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.application.services.sandbox_provisioner import SandboxProvisioner
from app.domain.errors.sandbox_lifecycle import (
    SandboxProvisionError,
    SessionSuspendedError,
)
from app.domain.models.event import MessageEvent
from app.domain.models.file import File
from app.domain.models.session import SandboxBinding, SandboxBindingState, Session

from tests.app.application.services.test_sandbox_attachment_flush import (
    _FakeFlushStorage,
    _FakeFlushUoW,
)
from tests.app.application.services.test_sandbox_provision_flight import (
    FakeRegistry,
    FakeSandboxCls,
    FakeUoW,
    _unbound_session,
)
from tests.app.application.services.test_sandbox_provisioner import _FakeMetrics

pytestmark = pytest.mark.anyio

SUSPENDED = SandboxBindingState.SUSPENDED


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── flow-env fakes (thin adapters over the reused facades) ────────────────────


class _FlowSandboxCls(FakeSandboxCls):
    """``FakeSandboxCls`` + a ``create_count`` property = number of PHYSICAL
    container creations (``containers.run`` proxy). Distinct from "activations"
    (audit-log ``creating→active`` rows), which is what §5.9 counts."""

    @property
    def create_count(self) -> int:
        return len(self.create_calls)


class _FlowHandle:
    """Sandbox handle returned by the flow registry's ``acquire_handle``.

    Carries the flusher-facing upload surface (``uploaded_paths``) + the
    acquire-hit ``ensure_sandbox`` readiness probe + ``release``. A SINGLE shared
    instance is returned for every acquire so uploads accumulate on
    ``env.sandbox`` regardless of which acquire (bind_new tail or acquire-hit)
    produced the handle the flusher ran against."""

    def __init__(self) -> None:
        self.uploaded_paths: list[str] = []
        self.upload_calls = 0
        self.upload_success = True
        self.ensure_calls = 0
        self.ensure_raises: Optional[BaseException] = None
        self.released = 0
        self._registry: Optional["_FlowRegistry"] = None

    async def ensure_sandbox(self) -> None:
        self.ensure_calls += 1
        if self.ensure_raises is not None:
            raise self.ensure_raises

    async def upload_file(
        self, *, file_data, filepath, filename=None, refuse_special=False
    ):
        self.upload_calls += 1
        self.uploaded_paths.append(filepath)
        return SimpleNamespace(success=self.upload_success)

    def release(self) -> None:
        self.released += 1
        if self._registry is not None:
            self._registry._open = max(0, self._registry._open - 1)


class _FlowRegistry(FakeRegistry):
    """``FakeRegistry`` whose ``acquire_handle`` returns the shared rich handle
    (the flusher needs ``upload_file`` on it; the base fake handle has only
    ``release``)."""

    def __init__(self, handle: _FlowHandle) -> None:
        super().__init__()
        self.handle = handle

    def acquire_handle(self, session_id: str) -> _FlowHandle:
        self.acquire_handle_calls += 1
        self._open += 1
        return self.handle


class _SpyLifecycle(SandboxLifecycleService):
    """The REAL lifecycle service + a ``resume()`` call counter so a flow can
    assert INV-SPM-9 (the provisioner NEVER calls ``resume``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resume_calls = 0

    async def resume(self, session_id: str):
        self.resume_calls += 1
        return await super().resume(session_id)


class _AuditLogView:
    """``count(old_state=, new_state=)`` over the fake ``sandbox_lifecycle_log``
    rows (``_transition`` writes ``old_state`` / ``new_state`` string values).

    "Successful sandbox activation" (§5.9, G4 gray-launch signal) = rows with
    ``old_state="creating" AND new_state="active"`` — the ``old_state`` filter is
    load-bearing: without it a ``resume()`` (``suspended→active``) would recount
    the SAME container on every resume."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def count(self, *, old_state: str, new_state: str) -> int:
        return sum(
            1
            for r in self._rows
            if r.get("old_state") == old_state and r.get("new_state") == new_state
        )


def _make_flusher_and_provisioner(
    *, lifecycle, session_id: str, user_id: str, flush_uow, storage, metrics
) -> tuple[SandboxAttachmentFlusher, SandboxProvisioner]:
    """Build a fresh REAL flusher + REAL provisioner (the flusher wired as the
    provision hook). ``new_run`` reuses this to model a run restart with a
    zeroed in-process flush ledger."""
    flusher = SandboxAttachmentFlusher(
        uow_factory=lambda: flush_uow, file_storage=storage, metrics=None
    )

    async def _flush_hook(handle) -> None:
        await flusher.flush_all(session_id, handle)

    provisioner = SandboxProvisioner(
        session_id=session_id,
        user_id=user_id,
        lifecycle=lifecycle,
        hooks=[("tool_call", _flush_hook)],
        timeout_seconds=30,
        trigger="tool_call",
        metrics=metrics,
    )
    return flusher, provisioner


@dataclass
class _FlowEnv:
    """Test-facing handle over a fully-wired on_demand run (real lifecycle +
    provisioner + flusher)."""

    session_id: str
    user_id: str
    lifecycle: _SpyLifecycle
    provisioner: SandboxProvisioner
    flusher: SandboxAttachmentFlusher
    sandbox_cls: _FlowSandboxCls
    registry: _FlowRegistry
    sandbox: _FlowHandle
    storage: _FakeFlushStorage
    audit_log: _AuditLogView
    session: Session
    file_rows: dict
    flush_uow: _FakeFlushUoW

    # ── seeding ──

    def seed_message_event(self, attachments: list[str]) -> None:
        """Append a persisted ``MessageEvent`` carrying id-only ``File``
        attachments (producer contract). Each file row's ``filepath`` starts as a
        **MinIO-URL-shaped** string (NOT empty) — DD-20 anti-false-green: the
        flusher rewrites it to ``/home/ubuntu/upload/{filename}``, so an empty
        seed could mask a skipped transfer."""
        atts: list[File] = []
        for fid in attachments:
            if fid not in self.file_rows:
                self.file_rows[fid] = File(
                    id=fid,
                    filename=f"{fid}.pdf",
                    filepath=f"http://minio:9000/bucket/{fid}.pdf",
                )
            atts.append(File(id=fid))  # id-only on the wire
        self.session.events.append(MessageEvent(role="user", attachments=atts))

    # ── flow entry points ──

    async def _bind_new_vnc(self) -> None:
        # VNC-first direct build simulated at the lifecycle level (no provision
        # hooks run — attachments flush only on the first sandbox-facing tool).
        await self.lifecycle.bind_new(self.session_id, user_id=self.user_id)

    async def session_service_get_vnc_url(self) -> None:
        await self._bind_new_vnc()

    async def open_vnc_first_no_tool(self) -> None:
        await self._bind_new_vnc()

    async def invoke_tool(self, name: str) -> None:
        # A sandbox-facing tool call routes through the provisioner: acquire-hit
        # (container already ACTIVE) or bind_new (first touch), then hooks.
        await self.provisioner.get()

    async def finish_chat_only_run(self) -> None:
        # Pure-chat run: never invoked a sandbox tool, so the provisioner was
        # never triggered → zero containers.
        assert self.provisioner.peek() is None

    def new_run(self) -> "_FlowEnv":
        """Model a run restart: a brand-new provisioner + flusher (fresh
        in-process flush ledger) over the SAME lifecycle / registry / session /
        storage / audit log."""
        flusher, provisioner = _make_flusher_and_provisioner(
            lifecycle=self.lifecycle,
            session_id=self.session_id,
            user_id=self.user_id,
            flush_uow=self.flush_uow,
            storage=self.storage,
            metrics=_FakeMetrics(),
        )
        return replace(self, provisioner=provisioner, flusher=flusher)

    # ── activation-count scenario helpers ──

    async def provision_then_suspend_then_resume_and_tool(self) -> None:
        await self.provisioner.get()  # bind_new → creating→active (1 activation)
        await self.lifecycle.suspend(self.session_id)  # active→suspended
        await self.lifecycle.resume(self.session_id)  # suspended→active (filtered)
        # A post-resume tool re-acquires the SAME container (registry hit → NO new
        # creating→active). Drop the cached handle so get() actually re-enters
        # acquire() instead of returning the ready-cached handle.
        self.provisioner.release_held_handle()
        await self.provisioner.get()

    async def provision_hooks_fail_then_retry_ok(self) -> None:
        # A hook that fails once → hooks_failed (handle RETAINED) → the retry
        # reruns ONLY hooks; the container is already ACTIVE so no new activation.
        calls = {"n": 0}

        async def flaky(handle) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("hook boom")

        prov = SandboxProvisioner(
            session_id=self.session_id,
            user_id=self.user_id,
            lifecycle=self.lifecycle,
            hooks=[("tool_call", flaky)],
            timeout_seconds=30,
            trigger="tool_call",
        )
        with contextlib.suppress(SandboxProvisionError):
            await prov.get()  # bind_new (1 activation) + hook fails
        await prov.get()  # retries hooks only → ready

    async def provision_create_ok_then_ensure_fail_swallowed(self) -> None:
        # create OK + ensure_sandbox fails → bind_new compensates (dispose +
        # rollback creating→UNBOUND) → NO creating→active row. A container WAS
        # built (create_count would be 1) but the session never activated.
        self.sandbox_cls.ensure_raises = RuntimeError("supervisord dead")
        with contextlib.suppress(SandboxProvisionError):
            await self.provisioner.get()


def _build_env(*, binding: Optional[str] = None) -> _FlowEnv:
    session_id = "flow-sess"
    user_id = "owner-1"
    session = _unbound_session(session_id, user_id=user_id)
    if binding == "suspended":
        session.sandbox_binding = SandboxBinding(
            state=SUSPENDED, id="sbx-seed", generation=1
        )
    sessions: dict[str, Session] = {session_id: session}
    events: list = []
    audit_rows: list[dict] = []
    file_rows: dict = {}

    handle = _FlowHandle()
    registry = _FlowRegistry(handle)
    handle._registry = registry

    sandbox_cls = _FlowSandboxCls()
    lifecycle_uow = FakeUoW(sessions, events, audit_rows)
    lifecycle = _SpyLifecycle(
        sandbox_cls=sandbox_cls, uow_factory=lambda: lifecycle_uow
    )
    lifecycle._registry = registry  # type: ignore[assignment]

    flush_uow = _FakeFlushUoW(session, file_rows)
    storage = _FakeFlushStorage(file_rows)  # storage rows share the file rows

    flusher, provisioner = _make_flusher_and_provisioner(
        lifecycle=lifecycle,
        session_id=session_id,
        user_id=user_id,
        flush_uow=flush_uow,
        storage=storage,
        metrics=_FakeMetrics(),
    )

    return _FlowEnv(
        session_id=session_id,
        user_id=user_id,
        lifecycle=lifecycle,
        provisioner=provisioner,
        flusher=flusher,
        sandbox_cls=sandbox_cls,
        registry=registry,
        sandbox=handle,
        storage=storage,
        audit_log=_AuditLogView(audit_rows),
        session=session,
        file_rows=file_rows,
        flush_uow=flush_uow,
    )


@pytest.fixture
def flow_env():
    """Factory → a fresh fully-wired ``_FlowEnv`` per call.

    ``mode`` is accepted for spec-fidelity (the flows are the ``on_demand`` ones)
    but the provisioner always operates with on_demand semantics; there is no
    ``Settings`` construction here, so the flows are independent of the PR-2
    ``ALLOWED`` unlock."""

    def _factory(*, mode: str = "on_demand", binding: Optional[str] = None) -> _FlowEnv:
        return _build_env(binding=binding)

    return _factory


# ── flows (spec §9) ───────────────────────────────────────────────────────────


class TestOnDemandFlows:
    async def test_vnc_first_then_tool_single_provision(self, flow_env):
        """VNC-first: session_service directly builds (no hooks) → the first tool
        acquire-hit reruns hooks; a SINGLE container creation + attachments
        visible (INV-SPM-6 VNC-first branch)."""
        env = flow_env(mode="on_demand")
        env.seed_message_event(attachments=["f1"])
        await env.session_service_get_vnc_url()  # bind_new (legal trigger)
        assert env.sandbox_cls.create_count == 1
        await env.invoke_tool("shell_execute")  # acquire-hit → hooks catch up
        assert env.sandbox_cls.create_count == 1  # no second creation
        assert env.sandbox.uploaded_paths == ["/home/ubuntu/upload/f1.pdf"]

    async def test_suspend_stale_tool_call_does_not_resume_or_create(self, flow_env):
        """A stale tool call on a SUSPENDED session passes ``SessionSuspendedError``
        straight through — never ``resume()`` (INV-SPM-9), never create."""
        env = flow_env(mode="on_demand", binding="suspended")
        with pytest.raises(SessionSuspendedError):
            await env.provisioner.get()
        assert env.lifecycle.resume_calls == 0 and env.sandbox_cls.create_count == 0

    async def test_attachment_across_run_restart(self, flow_env):
        """Attachment survives a run restart: upload seed → pure-chat run ends
        (zero containers) → a NEW provisioner/flusher (restart, ledger zeroed) →
        first tool → the attachment is fully retransferred + visible."""
        env = flow_env(mode="on_demand")
        env.seed_message_event(attachments=["f1"])  # MinIO-URL filepath (DD-20)
        await env.finish_chat_only_run()
        assert env.sandbox_cls.create_count == 0
        env2 = env.new_run()  # fresh provisioner/flusher (in-process ledger zeroed)
        await env2.invoke_tool("file_read")
        assert env2.sandbox.uploaded_paths == ["/home/ubuntu/upload/f1.pdf"]

    async def test_sandbox_activation_count_via_audit_log(self, flow_env):
        """"Successful sandbox activation count" = ``sandbox_lifecycle_log`` rows
        with ``old_state="creating" AND new_state="active"`` (§5.9). The
        ``old_state`` filter keeps ``resume()`` (``suspended→active``) from
        recounting the same container. Six sub-scenarios lock exactly-once /
        no-double-count / all-paths."""

        def activations(e: _FlowEnv) -> int:
            return e.audit_log.count(old_state="creating", new_state="active")

        # pure-chat on_demand: zero activations = the saving.
        env = flow_env(mode="on_demand")
        await env.finish_chat_only_run()
        assert activations(env) == 0

        # a sandbox-facing tool: one activation.
        env2 = flow_env(mode="on_demand")
        await env2.invoke_tool("shell_execute")
        assert activations(env2) == 1

        # VNC-first direct build + a later acquire-hit tool: still one (reuse
        # produces no new creating→active).
        env_vnc = flow_env(mode="on_demand")
        await env_vnc.open_vnc_first_no_tool()
        await env_vnc.invoke_tool("shell_execute")
        assert activations(env_vnc) == 1

        # resume (suspended→active): filtered by old_state, not recounted.
        env_rs = flow_env(mode="on_demand")
        await env_rs.provision_then_suspend_then_resume_and_tool()
        assert activations(env_rs) == 1

        # hooks-only retry: container already ACTIVE → no new activation.
        env_hr = flow_env(mode="on_demand")
        await env_hr.provision_hooks_fail_then_retry_ok()
        assert activations(env_hr) == 1

        # post-create failure rollback: container built but session never
        # activated (rollback to UNBOUND) → zero activations.
        env_pc = flow_env(mode="on_demand")
        await env_pc.provision_create_ok_then_ensure_fail_swallowed()
        assert activations(env_pc) == 0
