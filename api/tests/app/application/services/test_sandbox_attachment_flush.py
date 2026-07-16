"""Tests for the SPM PR-1c Task 15 ``SandboxAttachmentFlusher`` (provision hook ①).

The flusher performs a full-idempotent retransfer of this session's persisted
``MessageEvent`` attachments into the freshly-provisioned sandbox. The pending set
is authoritative + replayable (spec DD-20): ALL file ids referenced by persisted
``MessageEvent.attachments`` (ordered dedupe). Per-file the flusher does a
tombstone check → strict download → same-path upload → filepath write-back +
``add_file_if_absent`` — EACH FILE ITS OWN UoW (OQ-2 per-file commit) — with a
read-commit cancel guard after every UoW exit (``db_uow.py:71`` swallows
``CancelledError`` WITHOUT ``uncancel()``).

Async runner note: this repo ships **pytest-anyio**, not pytest-asyncio (see
``tests/conftest.py``). The module-level ``pytestmark = pytest.mark.anyio`` + a
local ``anyio_backend`` fixture are authoritative; the task brief's illustrative
``@pytest.mark.asyncio`` decorators are intentionally dropped.
"""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_attachment_flush import SandboxAttachmentFlusher
from app.application.services.sandbox_provisioner import SandboxProvisioner
from app.domain.errors.sandbox_lifecycle import SandboxProvisionError
from app.domain.models.event import MessageEvent
from app.domain.models.file import File

# Reuse the Task 14 provisioner fakes for the two provisioner-level backstop tests.
from tests.app.application.services.test_sandbox_provisioner import (  # noqa: E501
    _FakeMetrics,
    _FakeProvLifecycle,
    _ProvFakes,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── flusher fakes ─────────────────────────────────────────────────────────────


class _FakeFlushSessionRepo:
    """Duck-typed ``SessionRepository`` slice: ``get_by_id`` (pending read) +
    ``add_file_if_absent`` (new idempotent write) + legacy ``add_file`` (must
    stay untouched — INV-SPM-2)."""

    def __init__(self, session_obj, uow: "_FakeFlushUoW") -> None:
        self._session = session_obj
        self._uow = uow
        self.add_file_if_absent_calls: list[tuple[str, str]] = []
        self.add_file_calls: list[tuple[str, str]] = []

    async def get_by_id(self, session_id: str):
        self._uow._ctx_ops.add("pending_read")
        return self._session

    async def add_file_if_absent(self, session_id: str, file: File) -> None:
        self._uow._ctx_ops.add("perfile_write")
        self.add_file_if_absent_calls.append((session_id, file.id))

    async def add_file(self, session_id: str, file: File) -> None:  # legacy
        self._uow._ctx_ops.add("perfile_write")
        self.add_file_calls.append((session_id, file.id))


class _FakeFlushFileRepo:
    """Duck-typed ``FileRepository`` slice: ``get_by_id`` (tombstone) + ``save``
    (filepath write-back)."""

    def __init__(self, rows: dict[str, File], uow: "_FakeFlushUoW") -> None:
        self.rows = rows
        self._uow = uow

    async def get_by_id(self, file_id: str):
        self._uow._ctx_ops.add("tombstone")
        return self.rows.get(file_id)

    async def save(self, file: File) -> None:
        self._uow._ctx_ops.add("perfile_write")
        self.rows[file.id] = file


class _FakeFlushUoW:
    """Duck-typed UoW async CM. A single instance is reused across every
    ``async with`` block (the ``uow_factory`` returns it); ``__aenter__`` resets
    the per-context op set so ``__aexit__`` knows which commit window this is.

    ``__aexit__`` mirrors ``db_uow.py:71``: an injected ``CancelledError`` landing
    on the commit is swallowed WITHOUT ``uncancel()`` (when
    ``swallow_commit_cancel``), so ``current_task().cancelling()`` stays > 0 and
    the flusher's post-exit guard can honor it."""

    def __init__(self, session_obj, file_rows: dict[str, File]) -> None:
        self._ctx_ops: set[str] = set()
        self.session = _FakeFlushSessionRepo(session_obj, self)
        self.file = _FakeFlushFileRepo(file_rows, self)
        # cancel-window knobs (armed per test)
        self.swallow_commit_cancel = False
        self.hang_on_pending_read_commit: asyncio.Event | None = None
        self.hang_on_perfile_write_commit: asyncio.Event | None = None
        self.pending_read_reached = asyncio.Event()
        self.perfile_write_reached = asyncio.Event()

    async def __aenter__(self) -> "_FakeFlushUoW":
        self._ctx_ops = set()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        # Only the no-exception path commits (matches db_uow.py); an inbound
        # exception rolls back and propagates.
        if exc_type is None:
            if "pending_read" in self._ctx_ops and self.hang_on_pending_read_commit is not None:
                self.pending_read_reached.set()
                try:
                    await self.hang_on_pending_read_commit.wait()
                except asyncio.CancelledError:
                    if not self.swallow_commit_cancel:
                        raise
                    # swallow like db_uow.py:71 — NO uncancel()
            if "perfile_write" in self._ctx_ops and self.hang_on_perfile_write_commit is not None:
                self.perfile_write_reached.set()
                try:
                    await self.hang_on_perfile_write_commit.wait()
                except asyncio.CancelledError:
                    if not self.swallow_commit_cancel:
                        raise
                    # swallow like db_uow.py:71 — NO uncancel()
        return False


class _FakeFlushStorage:
    """Duck-typed ``FileStorage`` — ``download_file(fid) -> (BinaryIO, File)``.

    Mirrors the REAL ``MinioFileStorage.download_file`` cancel-guard: after its
    internal read-UoW commit window (which may swallow a cancel), it honors a
    still-pending cancel BEFORE the MinIO ``get_object`` call, so a cancel there
    never wastes the external I/O."""

    def __init__(self, rows: dict[str, File]) -> None:
        self._rows = rows
        self.download_calls = 0
        self.download_raises: BaseException | None = None
        self.minio_get_object_calls = 0
        self.hang_on_download_read_commit: asyncio.Event | None = None
        self.download_read_reached = asyncio.Event()
        self.swallow_commit_cancel = False

    async def download_file(self, file_id: str):
        self.download_calls += 1
        if self.download_raises is not None:
            raise self.download_raises
        # simulate MinioFileStorage's internal read-UoW commit window
        if self.hang_on_download_read_commit is not None:
            self.download_read_reached.set()
            try:
                await self.hang_on_download_read_commit.wait()
            except asyncio.CancelledError:
                if not self.swallow_commit_cancel:
                    raise
                # swallow like db_uow.py:71 — NO uncancel()
        # storage-layer guard (the real fix in minio_file_storage.py): honor a
        # swallowed cancel AFTER the read-UoW, BEFORE the MinIO get_object.
        t = asyncio.current_task()
        if t is not None and t.cancelling() > 0:
            raise asyncio.CancelledError()
        self.minio_get_object_calls += 1
        file = self._rows[file_id]
        return io.BytesIO(b"payload"), file


class _FakeFlushSandbox:
    """Duck-typed ``SandboxHandle.upload_file`` — records paths + call count."""

    def __init__(self) -> None:
        self.uploaded_paths: list[str] = []
        self.upload_calls = 0
        self.upload_success = True

    async def upload_file(self, *, file_data, filepath, filename=None, refuse_special=False):
        self.upload_calls += 1
        self.uploaded_paths.append(filepath)
        return SimpleNamespace(success=self.upload_success)


class _FakeFlushMetrics:
    def __init__(self) -> None:
        self.skipped: list[tuple[str]] = []

    def record_attachment_skipped(self, *, reason: str) -> None:
        self.skipped.append((reason,))


@dataclass
class _FlushEnv:
    session_id: str
    session_obj: SimpleNamespace
    uow: _FakeFlushUoW
    file_repo: _FakeFlushFileRepo
    session_repo: _FakeFlushSessionRepo
    storage: _FakeFlushStorage
    sandbox: _FakeFlushSandbox
    metrics: _FakeFlushMetrics
    flusher: SandboxAttachmentFlusher

    def seed_message_event(self, attachments: list[str]) -> None:
        """Append a persisted ``MessageEvent`` carrying id-only ``File``
        attachments (producer contract); seed the shared file row + storage row
        for each id (filename = ``{id}.pdf``)."""
        atts: list[File] = []
        for fid in attachments:
            shared = self.file_repo.rows.get(fid)
            if shared is None:
                shared = File(id=fid, filename=f"{fid}.pdf")
                self.file_repo.rows[fid] = shared
                self.storage._rows[fid] = shared
            atts.append(File(id=fid))  # id-only on the wire
        self.session_obj.events.append(MessageEvent(role="user", attachments=atts))


@pytest.fixture
def flusher_env() -> _FlushEnv:
    session_obj = SimpleNamespace(events=[])
    uow = _FakeFlushUoW(session_obj, {})
    storage = _FakeFlushStorage({})
    sandbox = _FakeFlushSandbox()
    metrics = _FakeFlushMetrics()
    flusher = SandboxAttachmentFlusher(
        uow_factory=lambda: uow, file_storage=storage, metrics=metrics
    )
    return _FlushEnv(
        session_id="s1",
        session_obj=session_obj,
        uow=uow,
        file_repo=uow.file,
        session_repo=uow.session,
        storage=storage,
        sandbox=sandbox,
        metrics=metrics,
        flusher=flusher,
    )


# ── provisioner-level backstop env (reuses Task 14 fakes) ─────────────────────


@pytest.fixture
def provisioner_env() -> _ProvFakes:
    lifecycle = _FakeProvLifecycle()
    metrics = _FakeMetrics()
    prov = SandboxProvisioner(
        session_id="s1",
        user_id="u1",
        lifecycle=lifecycle,
        timeout_seconds=0.05,
        trigger="tool_call",
        metrics=metrics,
    )
    return _ProvFakes(
        lifecycle=lifecycle, metrics=metrics, handle=lifecycle.handle, prov=prov
    )


# ── TestFlushAll ──────────────────────────────────────────────────────────────


class TestFlushAll:
    async def test_full_idempotent_retransfer_from_persisted_events(self, flusher_env):
        """INV-SPM-6 core: pending = full set of persisted MessageEvent
        attachments (ordered dedupe); repeated flush is idempotent."""
        env = flusher_env
        env.seed_message_event(attachments=["f1", "f2"])
        env.seed_message_event(attachments=["f2", "f3"])  # f2 duplicate → deduped
        await env.flusher.flush_all(env.session_id, env.sandbox)
        assert env.sandbox.uploaded_paths == [
            "/home/ubuntu/upload/f1.pdf",
            "/home/ubuntu/upload/f2.pdf",
            "/home/ubuntu/upload/f3.pdf",
        ]
        await env.flusher.flush_all(env.session_id, env.sandbox)  # 2nd = same-path overwrite
        assert env.sandbox.upload_calls == 6

    async def test_cancel_at_pending_read_commit(self, flusher_env):
        """Step 1 pending-read UoW commit-window cancel swallowed → flusher guard
        re-raises → zero download/upload."""
        env = flusher_env
        env.seed_message_event(attachments=["f1"])
        env.uow.hang_on_pending_read_commit = asyncio.Event()
        env.uow.swallow_commit_cancel = True
        task = asyncio.create_task(env.flusher.flush_all(env.session_id, env.sandbox))
        await asyncio.wait_for(env.uow.pending_read_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert env.storage.download_calls == 0 and env.sandbox.upload_calls == 0

    async def test_cancel_at_download_uow_commit(self, flusher_env):
        """Step 2 download MinioFileStorage internal read-UoW commit-window cancel
        swallowed → storage-layer guard (post read-UoW, pre MinIO) re-raises →
        MinIO get_object zero calls, zero upload."""
        env = flusher_env
        env.seed_message_event(attachments=["f1"])
        env.storage.hang_on_download_read_commit = asyncio.Event()
        env.storage.swallow_commit_cancel = True
        task = asyncio.create_task(env.flusher.flush_all(env.session_id, env.sandbox))
        await asyncio.wait_for(env.storage.download_read_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert env.storage.minio_get_object_calls == 0 and env.sandbox.upload_calls == 0

    async def test_cancel_at_perfile_write_commit(self, flusher_env):
        """Step 2 write-back file.save/add_file_if_absent UoW commit-window cancel
        swallowed → guard re-raises → does not advance to the next file."""
        env = flusher_env
        env.seed_message_event(attachments=["f1", "f2"])
        env.uow.hang_on_perfile_write_commit = asyncio.Event()
        env.uow.swallow_commit_cancel = True
        task = asyncio.create_task(env.flusher.flush_all(env.session_id, env.sandbox))
        await asyncio.wait_for(env.uow.perfile_write_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert env.sandbox.upload_calls == 1  # only f1 uploaded; never reached f2

    async def test_cancel_after_hook_before_ready_records_cancelled(self, provisioner_env):
        """Provisioner-level backstop: a hook (flusher) whose UoW swallows a cancel
        then returns "success" is caught by the per-hook ``cancelling()>0`` check
        BEFORE ready → provisioner emits ``cancelled`` and does NOT set ready."""
        fakes = provisioner_env
        hook_entered = asyncio.Event()

        async def swallow_hook(handle):
            # Mirror the flusher: a UoW commit inside the hook swallows a cancel
            # (db_uow.py:71 — no uncancel) and the hook returns success.
            hook_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return  # swallow; do NOT re-raise, do NOT uncancel

        prov = fakes.make_provisioner(hooks=[swallow_hook], timeout_seconds=30)
        task = asyncio.create_task(prov.get())
        await asyncio.wait_for(hook_entered.wait(), timeout=2)
        inflight = prov._inflight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.gather(inflight, return_exceptions=True)  # settle detached flight
        assert fakes.metrics.last().outcome == "cancelled"
        assert prov.state != "ready"

    async def test_hard_timeout_records_failed(self, provisioner_env):
        """Hook hard timeout (provision timeout asyncio.timeout → TimeoutError) →
        emit ``failed``, does NOT set ready."""
        fakes = provisioner_env

        async def hang_hook(handle):
            await asyncio.Event().wait()  # never returns → provision timeout fires

        prov = fakes.make_provisioner(hooks=[hang_hook], timeout_seconds=0.05)
        with pytest.raises((TimeoutError, SandboxProvisionError)):
            await prov.get()
        assert fakes.metrics.last().outcome == "failed"
        assert prov.state != "ready"

    async def test_deleted_file_tombstone_skipped_with_metric(self, flusher_env):
        env = flusher_env
        env.seed_message_event(attachments=["gone", "f1"])
        env.file_repo.rows.pop("gone")  # row absent = deleted
        await env.flusher.flush_all(env.session_id, env.sandbox)  # does not raise
        assert env.metrics.skipped == [("deleted",)]
        assert env.sandbox.uploaded_paths == ["/home/ubuntu/upload/f1.pdf"]

    async def test_transfer_failure_raises_strict(self, flusher_env):
        env = flusher_env
        env.seed_message_event(attachments=["f1"])
        env.storage.download_raises = IOError("minio down")  # row present, download fails
        with pytest.raises(IOError):
            await env.flusher.flush_all(env.session_id, env.sandbox)

    async def test_filepath_written_back_and_add_file_called(self, flusher_env):
        env = flusher_env
        env.seed_message_event(attachments=["f1"])
        await env.flusher.flush_all(env.session_id, env.sandbox)
        assert env.file_repo.rows["f1"].filepath == "/home/ubuntu/upload/f1.pdf"
        assert env.session_repo.add_file_if_absent_calls == [("s1", "f1")]
        assert env.session_repo.add_file_calls == []  # legacy untouched (INV-SPM-2)


# ── TestVisionHydrate (runner-level fork) ─────────────────────────────────────


def _b_is_image(block: dict) -> bool:
    return block.get("type") == "image_url"


class _VisionFileRepo:
    def __init__(self) -> None:
        self.rows: dict[str, File] = {}

    async def get_by_id(self, file_id: str):
        return self.rows.get(file_id)


class _VisionUoW:
    """Read-only UoW async CM exposing ``.file`` (pure DB hydrate, no commit)."""

    def __init__(self, file_repo: _VisionFileRepo) -> None:
        self.file = file_repo

    async def __aenter__(self) -> "_VisionUoW":
        return self

    async def __aexit__(self, *a) -> bool:
        return False


class _CountingAccessor:
    """``peek()`` returns a handle only when provisioned; ``get()`` must NEVER be
    called on the vision-hydrate path (zero sandbox supply)."""

    def __init__(self, provisioned: bool) -> None:
        self._provisioned = provisioned
        self.gets = 0

    def peek(self):
        return object() if self._provisioned else None

    async def get(self):
        self.gets += 1
        raise AssertionError("vision hydrate must not call accessor.get()")


@pytest.fixture
def runner_msg_env():
    """Factory: a ``__new__``-built ``AgentTaskRunner`` exposing only the
    attachment-routing + vision-hydrate + image-block surface (mirrors the
    ``test_agent_task_runner_profile`` harness style)."""

    def _build(*, mode: str = "on_demand", provisioned: bool = False):
        from app.domain.services.agent_task_runner import AgentTaskRunner

        runner = AgentTaskRunner.__new__(AgentTaskRunner)
        runner._sandbox_provision_mode = mode
        runner._session_id = "s1"
        runner._attachment_flusher = None
        runner._sandbox_provisioner = None
        runner._supports_vision = True
        runner.profile = None
        runner._image_url_map = {}
        file_repo = _VisionFileRepo()
        runner._uow = _VisionUoW(file_repo)
        runner._file_storage = MagicMock()
        runner._get_image_presigned_url = AsyncMock(return_value="https://s3.example/x")
        accessor = _CountingAccessor(provisioned)
        runner._sandbox_accessor = accessor

        env = SimpleNamespace(runner=runner, file_repo=file_repo, accessor=accessor)

        async def handle_message_event(attachments):
            event = MessageEvent(role="user", attachments=list(attachments))
            await runner._route_message_attachments(event)
            imgs = [a for a in event.attachments if isinstance(a, File)]
            return await runner._build_image_blocks(imgs)

        env.handle_message_event = handle_message_event
        return env

    return _build


class TestVisionHydrate:
    async def test_id_only_image_attachment_hydrated_without_sandbox(self, runner_msg_env):
        """on_demand not-provisioned: an id-only File is hydrated (MIME filled)
        from uow.file.get_by_id BEFORE vision assembly → image enters the blocks;
        zero accessor.get() / zero sandbox touch."""
        env = runner_msg_env(mode="on_demand", provisioned=False)
        env.file_repo.rows["img1"] = File(
            id="img1",
            filename="a.png",
            mime_type="image/png",
            filepath="http://minio:9000/b/a.png",
        )
        blocks = await env.handle_message_event(attachments=[File(id="img1")])  # id-only
        assert any(_b_is_image(b) for b in blocks)  # hydrated → into vision
        assert env.accessor.gets == 0  # zero supply

    async def test_deleted_id_only_attachment_skipped_in_vision(self, runner_msg_env):
        env = runner_msg_env(mode="on_demand", provisioned=False)
        blocks = await env.handle_message_event(attachments=[File(id="ghost")])  # row absent
        assert blocks == [] or not any(_b_is_image(b) for b in blocks)  # silent skip, no crash
        assert env.accessor.gets == 0


# ── TestFlushIncremental ──────────────────────────────────────────────────────


class TestFlushIncremental:
    async def test_ledger_skips_already_flushed(self, flusher_env):
        env = flusher_env
        env.seed_message_event(attachments=["f1"])
        await env.flusher.flush_all(env.session_id, env.sandbox)  # f1 → 1 upload, ledgered
        # f9 needs a downloadable row (incremental takes explicit ids, not events):
        env.file_repo.rows["f9"] = File(id="f9", filename="f9.pdf")
        env.storage._rows["f9"] = env.file_repo.rows["f9"]
        await env.flusher.flush_incremental(env.session_id, env.sandbox, ["f1", "f9"])
        assert env.sandbox.uploaded_paths[-1] == "/home/ubuntu/upload/f9.pdf"
        assert env.sandbox.upload_calls == 2  # f1 once + f9 once; f1 skipped by ledger
