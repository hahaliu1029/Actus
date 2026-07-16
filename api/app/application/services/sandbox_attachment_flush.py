"""SPM PR-1c Task 15: ``SandboxAttachmentFlusher`` — provision hook ①.

Full-idempotent retransfer of THIS session's persisted ``MessageEvent``
attachments into a freshly-provisioned sandbox. The pending set is authoritative
and replayable across run/restart (spec DD-20, zero-predicate): ALL file ids
referenced by persisted ``MessageEvent.attachments`` (ordered dedupe). Because
those events are persisted BEFORE the run, a provision that reruns hooks (e.g. a
``hooks_failed`` retry after an incremental-flush miss) replays the full set and
covers any file the in-run incremental path dropped.

Per-file (``_flush_one``), each step in its OWN UoW (OQ-2 frozen — per-file
commit, so partial success is visible and a retry is idempotent):

1. **tombstone** — ``uow.file.get_by_id`` returns ``None`` (row gone / deleted) →
   ``logger.warning`` + ``metrics.record_attachment_skipped(reason="deleted")`` +
   continue. NEVER fail on a tombstone (a permanently-deleted attachment would
   otherwise poison every future provision).
2. **download** — ``file_storage.download_file`` — any exception is raised as-is
   (strict; the provisioner books it under ``phase="hooks"``).
3. **upload** — ``sandbox.upload_file`` to ``/home/ubuntu/upload/{filename}``
   (idempotent key = same-path overwrite); ``result.success`` false → RuntimeError.
4. **write-back** — ``file.filepath = filepath``; ``uow.file.save`` +
   ``uow.session.add_file_if_absent`` (the new id-idempotent association write;
   legacy ``add_file`` is byte-untouched — INV-SPM-2).

**read-commit cancel guard (r21/R21-U1, r24/R24-CLASS1):** every UoW's ``__aexit__``
empty/write commit also swallows ``CancelledError`` WITHOUT ``uncancel()``
(``db_uow.py:71``), so a last-waiter/hard-timeout cancel landing on a commit would
otherwise let the flusher keep transferring and let the provision reach ready with
an ACTIVE-but-unwatched sandbox. After EVERY UoW exit — and as a second-level
depth guard after ``download_file`` returns — this module honors a still-pending
cancel via ``_raise_if_read_swallowed_cancel`` (same shape as
``SandboxLifecycleService._raise_if_read_swallowed_cancel``). The download's own
nested read-UoW closure point lives INSIDE ``MinioFileStorage.download_file``
(guard added there in the same task), so a cancel swallowed by that inner commit
is honored before the MinIO ``get_object`` — the post-return guard here is only
depth.

**Strict-failure contract:** the flusher raises ``IOError`` / ``RuntimeError`` /
``CancelledError`` only. It must NEVER raise ``SessionSuspendedError`` /
``SessionFinalizedError`` — the provisioner's Suspended/Finalized handler clears
its handle WITHOUT release (it assumes those originate from ``acquire()``), so a
hook raising them would leak the handle.

INV-SPM-2: NEW code only — the always-mode attachment path is unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, BinaryIO, Callable, Protocol, Tuple

from app.domain.models.event import MessageEvent

if TYPE_CHECKING:
    from app.domain.external.sandbox import SandboxHandle
    from app.domain.models.file import File
    from app.domain.repositories.uow import IUnitOfWork

logger = logging.getLogger(__name__)


def _raise_if_read_swallowed_cancel() -> None:
    """Honor a cancellation swallowed by a UoW's ``__aexit__`` commit.

    A read-only or write UoW's ``__aexit__`` commit swallows ``CancelledError``
    and does NOT ``uncancel()`` (``db_uow.py:71``✓), so
    ``current_task().cancelling()`` stays > 0 and is detectable after the fact.
    Call this right after EVERY UoW exits and before the next side effect
    (download / upload / next file) to convert a swallowed cancel into a clean
    abort. The cancel-lands-on-the-``await`` case propagates naturally and never
    reaches here (``cancelling() == 0`` pre-commit). Same shape as
    ``SandboxLifecycleService._raise_if_read_swallowed_cancel``.
    """
    t = asyncio.current_task()
    if t is not None and t.cancelling() > 0:
        raise asyncio.CancelledError()


class _FileStoragePort(Protocol):
    async def download_file(self, file_id: str) -> Tuple[BinaryIO, "File"]: ...


class _AttachmentFlushMetrics(Protocol):
    def record_attachment_skipped(self, *, reason: str) -> None: ...


class SandboxAttachmentFlusher:
    """Provision hook ①：本 session 持久化 MessageEvent 附件全量幂等重传。

    pending 权威集 = session.events 中全部 ``MessageEvent.attachments`` 的 file id
    （先于 run 持久化、跨 run/重启可 replay——spec DD-20 零谓词）。
    """

    def __init__(
        self,
        uow_factory: Callable[[], "IUnitOfWork"],
        file_storage: _FileStoragePort,
        metrics: _AttachmentFlushMetrics | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._metrics = metrics
        # per-process in-run optimization only — NOT a correctness authority.
        self._flushed: dict[str, set[str]] = {}

    async def flush_all(self, session_id: str, sandbox: "SandboxHandle") -> None:
        """Full retransfer: pending = every persisted MessageEvent attachment id
        (ordered dedupe). Idempotent (same-path overwrite). Records each flushed
        id into the ledger so a later ``flush_incremental`` skips it."""
        pending = await self._collect_pending(session_id)
        for file_id in pending:
            await self._flush_one(session_id, sandbox, file_id)

    async def flush_incremental(
        self, session_id: str, sandbox: "SandboxHandle", file_ids: list[str]
    ) -> None:
        """In-run incremental flush of THIS message's attachment ids. Skips any id
        already in the per-process ledger (the full retransfer backstops misses)."""
        seen = self._flushed.get(session_id, set())
        for file_id in dict.fromkeys(file_ids):  # ordered dedupe
            if file_id in seen:
                continue
            await self._flush_one(session_id, sandbox, file_id)

    async def _collect_pending(self, session_id: str) -> list[str]:
        """Authoritative pending set: ordered-dedupe of every persisted
        ``MessageEvent.attachments`` file id."""
        ids: list[str] = []
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
            if session is not None:
                for event in session.events:
                    if isinstance(event, MessageEvent) and event.attachments:
                        ids.extend(f.id for f in event.attachments)
        # pending-read UoW commit-window swallowed-cancel guard (before any I/O)
        _raise_if_read_swallowed_cancel()
        return list(dict.fromkeys(ids))

    async def _flush_one(
        self, session_id: str, sandbox: "SandboxHandle", file_id: str
    ) -> None:
        # 1. tombstone — own read UoW; row gone → warn + metric skip + continue
        async with self._uow_factory() as uow:
            row = await uow.file.get_by_id(file_id)
        _raise_if_read_swallowed_cancel()
        if row is None:
            logger.warning(
                "attachment %s row gone (deleted); skipping full-retransfer", file_id
            )
            if self._metrics is not None:
                self._metrics.record_attachment_skipped(reason="deleted")
            return

        # 2. download — strict: any exception raised as-is (provisioner: phase=hooks)
        file_data, file = await self._file_storage.download_file(file_id)
        # second-level depth guard after download returns (the true nested-read
        # closure point is the guard INSIDE MinioFileStorage.download_file)
        _raise_if_read_swallowed_cancel()

        # 3. upload — idempotent key = same-path overwrite
        filepath = f"/home/ubuntu/upload/{file.filename}"
        result = await sandbox.upload_file(
            file_data=file_data, filepath=filepath, filename=file.filename
        )
        if not getattr(result, "success", False):
            raise RuntimeError(
                f"attachment upload failed for {file_id} → {filepath}"
            )

        # 4. write-back filepath + id-idempotent association — own write UoW
        file.filepath = filepath
        async with self._uow_factory() as uow:
            await uow.file.save(file)
            await uow.session.add_file_if_absent(session_id, file)
        _raise_if_read_swallowed_cancel()

        # 5. ledger (per-process optimization only)
        self._flushed.setdefault(session_id, set()).add(file_id)
