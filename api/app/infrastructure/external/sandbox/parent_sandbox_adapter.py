"""C2 v1 ParentSandboxAdapter — wraps live SandboxHandle (spec §10.5).

Translates the live ``SandboxHandle`` surface (``BinaryIO`` /
``ToolResult`` returns from the sandbox HTTP RPC) into the narrower bytes
/ bool / None contract that ``ParentSandboxPort`` defines.

Composition root (PR-7/8) constructs one of these per coordinator run
holding the parent session's ``SandboxHandle``. The adapter MUST NOT
expose destroy() — see ``ParentSandboxPort`` docstring §10.5 invariant.

**v1 limitations (PR-5 cold-code):**

- *atomic_write_file true atomicity* — **CLOSED by S1.** The sandbox HTTP
  service backing ``SandboxHandle.upload_file`` now writes via
  ``mkstemp + fsync + os.replace`` (``_atomic_write_bytes`` in
  ``sandbox/app/services/file.py``), so a mid-write exception no longer
  leaves a truncated file: the contract's final-path-content
  raise-or-succeed atomicity holds for regular-file / symlink / new-file
  targets and the ``WRITE_IO_ERROR`` rollback skip is sound. (An EXISTING
  special file — FIFO/socket/device — falls back to a non-atomic
  write-through per S1 D12 so the node is not clobbered, which is parity
  with pre-S1; atomicity is meaningless for a stream/device. A patch entry
  carries regular-file content in practice, but the apply path does NOT
  stat-guard the target inode type, so this atomicity guarantee is *scoped
  to regular-file targets* — not a claim that a special-file target is
  unreachable on the apply path.) See the S1 design spec
  (``docs/superpowers/specs/2026-06-16-c2full-s1-atomic-write-design.md``).

- *Path resolution* [codex R8 P1] — still open, **addressed by S1b (Gap B).**
  ``FilePatchEntry.path`` is strict sandbox-relative; the live sandbox HTTP
  API resolves a relative path against the sandbox CWD (``/sandbox``), NOT a
  parent-workspace root. The relative->absolute join is an S1b migration
  (it breaks the locked ``test_g2b_path_roundtrip`` invariant). The
  ``ACTUS_C2_COORDINATOR_ENABLED=false`` flag gate prevents the coordinator
  apply path from going live before S1b lands. (Gap B is orthogonal to
  atomicity — S1 makes the write atomic, S1b makes the path correct.)
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import TYPE_CHECKING, Optional

from app.domain.external.parent_sandbox import ParentSandboxPort

if TYPE_CHECKING:
    from app.domain.external.sandbox import SandboxHandle

logger = logging.getLogger(__name__)


class ParentSandboxAdapter(ParentSandboxPort):
    """Adapter from ``SandboxHandle`` → ``ParentSandboxPort``.

    The constructor takes any object satisfying the ``SandboxHandle``
    Protocol (so DockerSandbox or a future K8s sandbox both fit). Tests
    inject an ``AsyncMock`` directly.
    """

    def __init__(self, sandbox_handle: "SandboxHandle") -> None:
        self._sandbox = sandbox_handle

    async def compute_digest(self, path: str) -> Optional[str]:
        """Return SHA-256 hex of file at ``path``, or ``None`` if missing.

        Per §9.3 step 5 the reducer treats a missing file as a no-drift
        signal — the applier's preflight will catch missing-file as
        ``FILE_MISSING`` before any write happens, so we don't need to
        re-raise here.
        """
        try:
            content = await self.read_file(path)
        except FileNotFoundError:
            return None
        return hashlib.sha256(content).hexdigest()

    async def exists(self, path: str) -> bool:
        """File-exists probe with strict RPC-failure semantics.

        Raises ``OSError`` if the underlying ``check_file_exists`` RPC
        fails (``ToolResult.success == False``).

        [codex R1 P1#3 fix] Previously we collapsed RPC failure to
        ``False``, which would have the applier preflight misreport
        FILE_MISSING for modify/delete (the file might actually exist
        — we just couldn't reach the sandbox) or silently proceed to
        write an ``add`` even though we couldn't verify the path was
        free. Raising surfaces the real fault and lets callers wrap
        if they need a softer signal.

        [codex R12 P1 fix] The live sandbox HTTP service returns
        ``FileCheckResult({filepath, exists})`` as ``result.data`` —
        a dict that is ALWAYS truthy regardless of whether the file
        exists. Previous ``bool(result.data)`` returned True for
        non-existent files, breaking ``add`` preflight (would report
        FILE_EXISTS). Three accepted shapes (in priority order):

        1. dict with ``"exists"`` key (live sandbox response)
        2. object with ``.exists`` attribute (Pydantic FileCheckResult
           if a future from_sandbox preserves the model)
        3. plain bool (test fixtures + degenerate None/empty)
        """
        result = await self._sandbox.check_file_exists(path)
        if not result.success:
            raise OSError(
                f"sandbox check_file_exists failed for {path!r}: "
                f"{result.message!r}",
            )
        data = result.data
        if isinstance(data, dict):
            return bool(data.get("exists", False))
        exists_attr = getattr(data, "exists", None)
        if exists_attr is not None:
            return bool(exists_attr)
        return bool(data)

    async def read_file(self, path: str) -> bytes:
        """Read raw file contents.

        ``SandboxHandle.download_file`` returns ``BinaryIO``; we read
        all bytes and close the stream. A missing-file response from
        the sandbox HTTP layer surfaces as ``httpx.HTTPStatusError``
        (4xx); we translate to ``FileNotFoundError`` so callers can
        match a portable exception type and ``compute_digest`` can
        return None.

        [codex R1 P2#2 fix] Always close the BinaryIO via ``try/finally``.
        The live DockerSandbox impl wraps ``response.content`` in
        ``io.BytesIO`` so close() is essentially a no-op, but the
        ``SandboxHandle.download_file`` Protocol return type is the
        general ``BinaryIO`` — a future K8s/remote impl could hold an
        underlying socket / file handle that leaks if not closed.
        """
        try:
            stream = await self._sandbox.download_file(path)
        except Exception as exc:
            # httpx.HTTPStatusError doesn't subclass OSError; check the
            # response status if available, else re-raise. Conservative:
            # only translate the unambiguous 404 case; bubble other
            # failures (5xx, network, auth) as-is so the orchestrator
            # surfaces the real cause instead of silently dropping them
            # as "file missing".
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                raise FileNotFoundError(path) from exc
            raise
        try:
            return stream.read()
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as close_exc:  # noqa: BLE001
                    # Close failure must not mask the read result, but
                    # we DO want it observable so future remote-stream
                    # / socket implementations don't leak silently.
                    # [codex R3 P2#5] WARN (not ERROR) — the read
                    # itself succeeded; this is hygiene observability.
                    logger.warning(
                        "ParentSandboxAdapter.read_file: stream close "
                        "failed for %r: %s",
                        path, close_exc,
                    )

    async def atomic_write_file(self, path: str, content: bytes) -> None:
        """Write ``content`` to ``path``.

        Per-file atomicity is enforced sandbox-side as of **S1**: the live
        sandbox HTTP service backing ``upload_file`` writes via
        ``mkstemp + fsync + os.replace`` (``_atomic_write_bytes``), so this
        method's contract — final-path-content raise-or-succeed with no
        observable truncated file — holds for regular-file / symlink /
        new-file targets (an EXISTING special file falls back to a non-atomic
        write-through per S1 D12 — parity with pre-S1, the node is not
        clobbered; the apply path does NOT stat-guard the target inode type,
        so the guarantee is *scoped to regular-file targets*, not asserted
        unreachable for special files). The applier's ``WRITE_IO_ERROR``
        branch may therefore soundly skip rollback of the current entry (a
        regular-file write that raised left no partial final-path content). A
        newly-created empty parent directory on a pre-replace failure is a
        benign, pre-existing non-atomic side effect invisible to
        digest/rollback (S1 §1).

        ``SandboxHandle.upload_file`` takes ``BinaryIO`` and returns
        ``ToolResult``. We raise ``OSError`` on ``success=False`` so the
        applier's per-entry try/except catches it and routes to the
        ``WRITE_IO_ERROR`` branch with rollback.
        """
        # [finish-core §5.2 G2b (c)] A bare filename → sandbox os.makedirs("")
        # raises. Reject loudly here so the failure is attributable, not a
        # cryptic FileNotFoundError from the remote service. Path-transparency
        # is preserved: any path with a directory component (relative or
        # absolute) passes through unchanged — no ``/workspace`` join.
        if os.path.dirname(path) == "":
            raise ValueError(
                f"bare filename rejected (no directory component): {path!r}; "
                f"coordinator manifest paths must include a directory "
                f"(e.g. 'workspace/foo.py')"
            )
        result = await self._sandbox.upload_file(io.BytesIO(content), path)
        if not result.success:
            raise OSError(
                f"sandbox upload_file failed for {path!r}: {result.message!r}",
            )

    async def delete_file(self, path: str) -> None:
        """Delete file at ``path``.

        Same failure convention as ``atomic_write_file``: ``success=False``
        from the sandbox RPC raises ``OSError`` so the applier's per-entry
        try/except routes to the failure branch with rollback.
        """
        result = await self._sandbox.delete_file(path)
        if not result.success:
            raise OSError(
                f"sandbox delete_file failed for {path!r}: {result.message!r}",
            )
