"""C2 v1 ParentSandboxAdapter — wraps live SandboxHandle (spec §10.5).

Translates the live ``SandboxHandle`` surface (``BinaryIO`` /
``ToolResult`` returns from the sandbox HTTP RPC) into the narrower bytes
/ bool / None contract that ``ParentSandboxPort`` defines.

Composition root (PR-7/8) constructs one of these per coordinator run
holding the parent session's ``SandboxHandle``. The adapter MUST NOT
expose destroy() — see ``ParentSandboxPort`` docstring §10.5 invariant.

**Known v1 limitations (PR-5 cold-code; addressed before flag flip):**

- *atomic_write_file true atomicity* [codex R8 P1]: ``ParentSandboxPort``
  contracts ``atomic_write_file`` as raise-or-succeed with no observable
  side effect. The live sandbox HTTP service backing
  ``SandboxHandle.upload_file`` currently writes via ``open(path,
  'wb')`` + chunked write (``sandbox/app/services/file.py``); a mid-
  write exception leaves a truncated file. Closing the gap requires
  EITHER (a) sandbox-side tmp+fsync+rename, OR (b) adapter-side
  upload to a temp path + atomic rename via a sandbox rename RPC.
  PR-5 ships the contract + applier logic that relies on it; the
  sandbox-side enforcement is a PR-7 / sandbox-team follow-up.
  Until then operators may observe partial-file leakage on an
  apply that crashes mid-write — symptom is documented in the
  ``failed_reason`` audit column and HealthEvent metrics.

- *Path resolution* [codex R8 P1]: ``FilePatchEntry.path`` is strict
  sandbox-relative (rejects absolute + non-canonical). The live
  sandbox HTTP API currently expects absolute paths
  (``sandbox/app/interfaces/schemas/file.py``). PR-5 cold-code does
  NOT yet join the relative path to a parent-sandbox root prefix —
  that join happens at the PR-7/8 composition root which owns the
  parent sandbox's "patch root" path (typically ``/workspace`` or
  similar). Until composition root wires it, passing a relative
  patch path through this adapter directly to a live sandbox would
  resolve against the sandbox's CWD; PR-5's cold-code gate
  (``ACTUS_C2_COORDINATOR_ENABLED=false``) prevents that today.
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

        Atomicity at the FS level is whatever the sandbox backend
        provides. The applier's snapshot/rollback layer (§10)
        provides the all-or-nothing guarantee across a multi-file
        plan *assuming* per-file atomicity holds. **Today this
        assumption is partial** [codex R8 P1]: the live sandbox HTTP
        service backing ``upload_file`` writes via ``open(path,
        'wb')`` + chunked write, so a mid-write exception leaves a
        truncated file. The applier's ``WRITE_IO_ERROR`` branch
        relies on raise-or-succeed atomicity to skip the current
        entry's rollback — see module docstring "Known v1
        limitations" for the gap + PR-7 mitigation.

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
