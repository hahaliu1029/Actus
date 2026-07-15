"""C2 v1 LocalFS RollbackSnapshotStore (spec §10.4).

**Process-stable only** — pod crash mid-apply means manual recovery
(operator inspects the audit table + leftover snapshot files). The PR-7
crash-recovery scope can promote this to crash-safe with a persistent
catalog; v1 keeps the simple per-pod local-filesystem layout.

Storage layout::

    {base_dir}/{coordinator_run_id}/{path_hash}

where ``path_hash = sha256(original_path.encode("utf-8")).hexdigest()``
— a fixed 64-char hex string (NAME_MAX-safe on all common
filesystems). The filename is opaque to callers (each ``FileSnapshot``
carries its own absolute ``snapshot_path``, so the on-disk name doesn't
need to be human-decoded). [codex R11 P1] Previously this used
``path.encode("utf-8").hex()`` which produced a 2×N-byte filename and
could exceed NAME_MAX (typically 255 bytes) for legal deep paths.
Defense in depth: even if a malicious ``original_path`` slipped past
``validate_relative_path_strict`` upstream, hashing it before using it
as a filename makes path traversal impossible.

Concurrency model: the PatchApplier holds a Redis lock keyed on
``coordinator:apply:{run_id}`` (spec §10.2), so two snapshot stores for
the same run never execute in parallel. We still defend against
already-exists / already-gone races in ``discard`` because k8s pod
restarts or operator cleanup may race with us.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Protocol


logger = logging.getLogger(__name__)


def _write_bytes(path: str, content: bytes) -> None:
    """Sync helper offloaded via asyncio.to_thread."""
    with open(path, "wb") as f:
        f.write(content)


def _read_bytes(path: str) -> bytes:
    """Sync helper offloaded via asyncio.to_thread."""
    with open(path, "rb") as f:
        return f.read()


def _validate_run_id_segment(coordinator_run_id: str) -> None:
    """[codex R3 P2#4] Defense in depth — ``coordinator_run_id`` is
    used as a path segment under ``base_dir``. Reject any value that
    could escape the base directory (path separators, ``..`` parent
    refs, NUL bytes, absolute paths).

    Normal coordinator run_ids match
    ``f"{session_id}:{step_id_hash16}:a{attempt_ix}"`` — all ASCII
    word chars + colons. This validator is the second wall: even if a
    malicious or buggy producer slipped a traversal value past upstream
    checks, snapshot files remain confined to the snapshot base dir.
    """
    if not coordinator_run_id:
        raise ValueError("coordinator_run_id must not be empty")
    if (
        "/" in coordinator_run_id
        or "\\" in coordinator_run_id
        or "\x00" in coordinator_run_id
        or ".." in coordinator_run_id
        or os.path.isabs(coordinator_run_id)
    ):
        raise ValueError(
            f"coordinator_run_id contains illegal path characters: "
            f"{coordinator_run_id!r}"
        )


@dataclass(frozen=True)
class FileSnapshot:
    """Opaque snapshot handle — caller never inspects ``snapshot_path``.

    ``original_path`` is the parent-sandbox path being snapshotted (used
    during rollback to know where to restore). ``original_digest`` is
    the SHA-256 hex digest the applier observed at snapshot time — the
    rollback path checks this to detect post-apply drift.
    """

    coordinator_run_id: str
    original_path: str
    snapshot_path: str
    original_digest: str


class RollbackSnapshotStore(Protocol):
    """Domain Protocol — applier holds one of these.

    Tests inject an ``AsyncMock``; production binds to
    ``LocalFSRollbackSnapshotStore``.
    """

    async def save(
        self,
        *,
        coordinator_run_id: str,
        path: str,
        content: bytes,
        original_digest: str,
        attempt_token: str | None = None,
    ) -> FileSnapshot: ...
    async def discard(
        self, coordinator_run_id: str, snapshots: Iterable[FileSnapshot],
    ) -> None: ...
    async def load(self, snapshot: FileSnapshot) -> bytes: ...


class LocalFSRollbackSnapshotStore(RollbackSnapshotStore):
    """Local-FS impl backed by ``${base_dir}/${run_id}/${path_hex}``."""

    def __init__(
        self, base_dir: str = "/tmp/actus/coordinator-rollback",
    ) -> None:
        self._base = base_dir

    async def save(
        self,
        *,
        coordinator_run_id: str,
        path: str,
        content: bytes,
        original_digest: str,
        attempt_token: str | None = None,
    ) -> FileSnapshot:
        """Snapshot ``content`` to disk; return an opaque ``FileSnapshot``.

        Idempotent at the OS level — overwriting an existing snapshot
        with the same hex name yields the same bytes (caller's
        responsibility to only snapshot once per path; the Redis lock
        at the applier level enforces this).

        [codex R1 P2#1 fix] Filesystem syscalls are offloaded to a
        thread via ``asyncio.to_thread`` so a large snapshot write
        doesn't block the event loop and starve other coroutines
        (mailbox heartbeats, SSE flushes, etc.).

        [codex R3 P2#4 fix] ``coordinator_run_id`` is used as a path
        segment under ``base_dir``. Even though normal coordinator
        run_ids match ``f"{session_id}:{step_id_hash16}:a{attempt_ix}"``
        (verified safe at the schema layer), defense in depth: reject
        any value containing a path separator or ``..`` so a malicious
        plan can't escape ``base_dir``.
        """
        _validate_run_id_segment(coordinator_run_id)
        # [codex R11 P1] SHA-256 hash (fixed 64 chars) instead of hex
        # of the raw bytes — bounds the on-disk filename length under
        # NAME_MAX regardless of how deep the manifest path goes.
        safe = hashlib.sha256(path.encode("utf-8")).hexdigest()
        target_dir = os.path.join(self._base, coordinator_run_id)
        if attempt_token is not None:
            if not isinstance(attempt_token, str) or not attempt_token:
                raise ValueError("attempt_token must be a non-empty string")
            # The directory, not just the filename, is attempt-scoped. An old
            # owner may lose its lease between replacement-owner mkdir/write;
            # cleaning only its own subdirectory cannot remove the replacement
            # directory in that window.
            token_hash = hashlib.sha256(
                attempt_token.encode("utf-8"),
            ).hexdigest()[:32]
            target_dir = os.path.join(target_dir, token_hash)
        await asyncio.to_thread(os.makedirs, target_dir, exist_ok=True)
        snap_path = os.path.join(target_dir, safe)
        await asyncio.to_thread(_write_bytes, snap_path, content)
        return FileSnapshot(
            coordinator_run_id=coordinator_run_id,
            original_path=path,
            snapshot_path=snap_path,
            original_digest=original_digest,
        )

    async def discard(
        self,
        coordinator_run_id: str,
        snapshots: Iterable[FileSnapshot],
    ) -> None:
        """Remove all snapshot files in ``snapshots``; rmdir if empty.

        Idempotent: missing-file is silently swallowed (concurrent
        cleanup, pod restart, operator manual cleanup all welcome). The
        rmdir is also best-effort — if other snapshots from a
        concurrent run still occupy the directory, ``OSError`` is the
        expected outcome and we ignore it.

        [codex R1 P2#1 fix] All FS syscalls go through
        ``asyncio.to_thread`` so a slow discard doesn't block the
        event loop.

        [codex R3 P2#4 fix] ``coordinator_run_id`` is validated the
        same way as in ``save`` so a malicious run_id can't direct the
        ``rmdir`` outside the snapshot base.
        """
        _validate_run_id_segment(coordinator_run_id)
        snapshots = tuple(snapshots)
        for snap in snapshots:
            try:
                await asyncio.to_thread(os.remove, snap.snapshot_path)
            except FileNotFoundError:
                # Already cleaned up by a prior discard or operator
                # cleanup — idempotent by design.
                pass
        run_dir = os.path.realpath(
            os.path.join(self._base, coordinator_run_id),
        )
        # Attempt-scoped directories are safe to remove independently. Only a
        # direct child of the validated run directory is eligible, so an
        # untrusted/forged FileSnapshot cannot direct rmdir elsewhere.
        attempt_dirs = {
            os.path.realpath(os.path.dirname(snap.snapshot_path))
            for snap in snapshots
        }
        attempt_scoped = False
        for attempt_dir in attempt_dirs:
            if (
                attempt_dir != run_dir
                and os.path.dirname(attempt_dir) == run_dir
            ):
                attempt_scoped = True
                try:
                    await asyncio.to_thread(os.rmdir, attempt_dir)
                except OSError:
                    pass
        if attempt_scoped:
            # The run directory is a shared stable container across owner
            # attempts. Removing it can race a replacement attempt between its
            # parent/child mkdir calls, so attempt cleanup must stop here.
            return
        try:
            await asyncio.to_thread(os.rmdir, run_dir)
        except OSError:
            # Directory non-empty (other in-flight snapshots) or
            # already removed by a concurrent discard. Both are safe
            # to ignore — disk-space cleanup is best-effort, not a
            # correctness signal.
            pass

    async def load(self, snapshot: FileSnapshot) -> bytes:
        """Read snapshotted bytes back for rollback.

        Raises ``FileNotFoundError`` if the snapshot has been
        prematurely discarded (programmer error — the applier's
        rollback path runs before discard, so a missing snapshot means
        a contract violation upstream). The applier's ``_rollback``
        catches this and routes to the rollback-partial branch with a
        ``HealthEvent``.

        [codex R1 P2#1 fix] ``open().read()`` runs in a thread so
        loading a large snapshot doesn't stall the event loop.
        """
        return await asyncio.to_thread(_read_bytes, snapshot.snapshot_path)
