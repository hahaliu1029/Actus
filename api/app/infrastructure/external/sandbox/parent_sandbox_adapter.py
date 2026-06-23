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
  special file — FIFO/socket/device — is REFUSED on the
  coordinator apply/seed path as of S1b: ``atomic_write_file`` passes
  ``refuse_special=True`` so the sandbox raises ``OSError(EINVAL)`` instead
  of the D12 non-atomic write-through, surfacing as ``WRITE_IO_ERROR`` +
  rollback rather than a FIFO hang. The 2a preflight additionally rejects a
  direct special target with ``TARGET_SPECIAL_FILE`` before the read. The
  non-atomic D12 write-through is retained ONLY for the agent ``write_file``
  path. The apply-path atomic-or-raise contract now holds for the special
  set too.) See the S1 design spec
  (``docs/superpowers/specs/2026-06-16-c2full-s1-atomic-write-design.md``).

- *Path resolution* — **CLOSED.** ``FilePatchEntry.path`` is strict
  sandbox-relative and the host adapter passes it through unchanged:
  host-side path-transparency — no ``/workspace`` / ``patch_root`` join — is
  locked by ``INV-F2.3`` + ``test_g2b_path_roundtrip`` (C2-finish G2b/PR-F3).
  The relative→absolute anchoring lives sandbox-side: the live sandbox HTTP
  service anchors a relative path under ``workspace_root`` (/home/ubuntu) via
  ``sandbox/app/core/workspace.py`` (the Sandbox Workspace Isolation epic),
  not the process CWD (/sandbox) — the ONLY layer all three writers (agent
  file_write, coordinator seed-install, parent apply) share — so round-trip
  identity holds at /home/ubuntu. (Gap B was orthogonal to atomicity: S1 made
  the write atomic, G2b/INV-F2.3 locked host path-transparency, and the
  Workspace Isolation epic makes the sandbox-side path correct.) S1b hardens
  the special-file target case instead (2a preflight reject + 2b write refuse).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

from app.domain.external.parent_sandbox import (
    ParentSandboxPort,
    SandboxPathCheck,
    WorkspaceScan,
    WorkspaceScanEntry,
)

# R7b P1-i: the 9 ``FileKind`` Literal members, materialized as a runtime set
# so the decode can reject a ``kind`` outside the contract. Mirrors the
# ``FileKind`` Literal in ``app.domain.external.parent_sandbox`` (a Literal has
# no membership API; keep these two in lockstep).
_VALID_FILE_KINDS = frozenset(
    {
        "missing",
        "regular",
        "directory",
        "symlink",
        "fifo",
        "socket",
        "block",
        "char",
        "other",
    }
)
# 64-char lowercase-hex sha256 alphabet (no uppercase per the walker contract).
_HEX_LOWER = frozenset("0123456789abcdef")


def _validate_scan_entry(entry: WorkspaceScanEntry, root: str) -> None:
    """R7b P1-i: enforce per-kind structural invariants on a decoded entry.

    Raises ``OSError`` (fail-CLOSED, consistent with the other decode guards)
    on any violation so a malformed/garbage entry never reaches the differ.
    The rules mirror the sandbox walker's contract (``WorkspaceScanEntry``
    docstring): only ``regular`` carries a sha, only ``symlink`` carries a
    ``link_target``.
    """
    rel_path = entry.rel_path
    kind = entry.kind
    if kind not in _VALID_FILE_KINDS:
        raise OSError(
            f"malformed snapshot_workspace entry {rel_path!r}: "
            f"unknown kind {kind!r} (not a FileKind) for {root!r}",
        )
    sha = entry.sha256
    link_target = entry.link_target
    if kind == "regular":
        if not (
            isinstance(sha, str)
            and len(sha) == 64
            and all(c in _HEX_LOWER for c in sha)
        ):
            raise OSError(
                f"malformed snapshot_workspace entry {rel_path!r}: regular "
                f"requires a 64-char lowercase-hex sha256, got {sha!r} "
                f"for {root!r}",
            )
        if link_target is not None:
            raise OSError(
                f"malformed snapshot_workspace entry {rel_path!r}: regular "
                f"must have link_target=None, got {link_target!r} for {root!r}",
            )
    elif kind == "symlink":
        if sha is not None:
            raise OSError(
                f"malformed snapshot_workspace entry {rel_path!r}: symlink "
                f"must have sha256=None, got {sha!r} for {root!r}",
            )
        if not (isinstance(link_target, str) and link_target != ""):
            raise OSError(
                f"malformed snapshot_workspace entry {rel_path!r}: symlink "
                f"requires a non-empty link_target, got {link_target!r} "
                f"for {root!r}",
            )
    else:
        # Any non-regular, non-symlink kind (directory/fifo/socket/block/
        # char/other/missing): neither a sha nor a link_target is carried.
        if sha is not None:
            raise OSError(
                f"malformed snapshot_workspace entry {rel_path!r}: kind "
                f"{kind!r} must have sha256=None, got {sha!r} for {root!r}",
            )
        if link_target is not None:
            raise OSError(
                f"malformed snapshot_workspace entry {rel_path!r}: kind "
                f"{kind!r} must have link_target=None, got {link_target!r} "
                f"for {root!r}",
            )

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

    async def check_path(self, path: str) -> SandboxPathCheck:
        """Inode-typed existence probe (S1b 2a).

        Parses ``{exists, kind}`` from the sandbox ``check_file_exists`` RPC.
        **Mixed-version legacy fallback**: an OLD sandbox image returns no
        ``kind``. Map a kindless EXISTING target to ``"regular"`` (NOT
        ``"other"``): pre-S1b behavior was ``exists()``-based and treated every
        existing target as a regular file, and PR-4's kind-invariant guards
        (dispatch seed-enrichment + ``PatchApplier`` preflight) now fail-CLOSED
        on any non-``"regular"`` kind — so an ``"other"`` default would break
        normal typed modify/delete/add on an old sandbox (the very
        "break all applies" regression this fallback exists to avoid).
        ``"other"`` stays reserved for a NEW sandbox that EXPLICITLY reports an
        indeterminate inode → correctly fail-closed. (Shell-mode capture needs a
        NEW sandbox — ``snapshot_workspace`` is a new RPC — so this legacy
        regular-fallback never weakens shell-capture protection.) Raises
        ``OSError`` on RPC failure, matching ``exists()``.
        """
        result = await self._sandbox.check_file_exists(path)
        if not result.success:
            raise OSError(
                f"sandbox check_file_exists failed for {path!r}: "
                f"{result.message!r}",
            )
        data = result.data
        if isinstance(data, dict):
            exists = bool(data.get("exists", False))
            kind = data.get("kind")
        else:
            exists = bool(getattr(data, "exists", False))
            kind = getattr(data, "kind", None)
        if not kind:
            # [codex PR-4 R2 P1] legacy (kindless) sandbox: existing → "regular"
            # (pre-S1b bool semantics), absent → "missing".
            kind = "missing" if not exists else "regular"
        return SandboxPathCheck(exists=exists, kind=kind)

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
        new-file targets.

        **S1b 2b (coordinator-only refuse):** this adapter passes
        ``refuse_special=True`` to ``upload_file`` so an EXISTING special
        file (FIFO / socket / device) at the target is **refused** — the
        sandbox raises ``EINVAL`` (surfaced here as ``OSError``) rather than
        doing the D12 non-atomic write-through that could block on a FIFO
        open (the TOCTOU hang the 2a preflight cannot fully close). The
        agent ``write_file`` path keeps the default ``refuse_special=False``
        and its D12 write-through is unchanged. The applier's
        ``WRITE_IO_ERROR`` branch may therefore soundly skip rollback of the
        current entry: a regular-file write that raised left no partial
        final-path content (``os.replace`` is the only mutating step), and a
        refused special-file write never touched the target node at all. A
        newly-created empty parent directory on a pre-replace failure is a
        benign, pre-existing non-atomic side effect invisible to
        digest/rollback (S1 §1).

        ``SandboxHandle.upload_file`` takes ``BinaryIO`` and returns
        ``ToolResult``. We raise ``OSError`` on ``success=False`` so the
        applier's per-entry try/except catches it and routes to the
        ``WRITE_IO_ERROR`` branch with rollback.
        """
        # [Sandbox Workspace Isolation §3.8 / single-path contract] Reject a bare
        # filename as coordinator manifest HYGIENE — manifest paths must carry a
        # directory component (e.g. 'workspace/foo.py'). As of the single-path
        # contract this is now a should-never-fire BACKSTOP: the bare-filename
        # rejection is enforced FORWARD at the lease boundary
        # (``_build_work_units_from_requests`` -> ``validate_coordinator_path``)
        # and the manifest wire schema
        # (``FilePatchEntry`` -> ``validate_directory_qualified_relative_path``),
        # so a bare path is rejected before children spawn / at envelope
        # construction rather than late here as ``write_io_error``. Kept as
        # defense-in-depth (the sandbox service anchors a bare name to
        # /home/ubuntu/<name>, so a slipped-through bare apply would otherwise
        # silently land at the workspace root). Path transparency is preserved:
        # any path WITH a directory component passes through unchanged.
        if os.path.dirname(path) == "":
            raise ValueError(
                f"bare filename rejected (no directory component): {path!r}; "
                f"coordinator manifest paths must include a directory "
                f"(e.g. 'workspace/foo.py')"
            )
        result = await self._sandbox.upload_file(
            io.BytesIO(content), path, refuse_special=True
        )
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

    async def kill_all_shell_sessions(self) -> None:
        """[S2 PR-4 §3.2] Quiesce: kill every tracked shell session's process
        group sandbox-side before the POST snapshot scan. Same failure
        convention as ``delete_file`` — ``success=False`` from the sandbox RPC
        raises ``OSError`` so the bounded finalizer routes the failure to
        ``_finalize_failed`` (a non-quiescent capture must not silently SUCCESS)."""
        result = await self._sandbox.kill_all_shell_sessions()
        if not result.success:
            raise OSError(
                f"sandbox kill_all_shell_sessions failed: {result.message!r}",
            )

    async def snapshot_workspace(
        self,
        root: str = "/home/ubuntu",
        *,
        max_paths: int,
        max_files: int,
        max_total_bytes: int,
        max_seconds: float,
    ) -> WorkspaceScan:
        """Decode the sandbox ``snapshot_workspace`` RPC into a typed scan.

        Fails CLOSED: a payload missing ``truncated`` (old image / malformed)
        is treated as ``truncated=True`` so the differ zero-applies rather than
        trusting an incomplete walk (opposite of ``check_path``'s fail-open).
        Raises ``OSError`` on RPC failure, matching ``exists()``/``check_path``.
        """
        result = await self._sandbox.snapshot_workspace(
            root,
            max_paths=max_paths,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
            max_seconds=max_seconds,
        )
        if not result.success:
            raise OSError(
                f"sandbox snapshot_workspace failed for {root!r}: "
                f"{result.message!r}",
            )
        data = result.data
        if isinstance(data, dict):
            # R3: get the raw value AS-IS (no ``or {}``). A FALSEY non-mapping
            # value (``[]`` / ``""`` / ``0`` / ``False``) must NOT be coerced
            # to an empty dict — that would decode a malformed payload to a
            # "complete empty scan" (fail-OPEN, differ skips group-zero-apply).
            raw_entries = data.get("entries")
            truncated = data.get("truncated")
        else:
            raw_entries = getattr(data, "entries", None)
            truncated = getattr(data, "truncated", None)
        # Fail closed: decode ``truncated`` strictly.
        # - absent/None (old image / missing key) => treat as truncated (the
        #   established lenient default for a MISSING flag, fail-CLOSED).
        # - a REAL bool => use it as-is. ``isinstance(truncated, bool)`` is the
        #   correct guard: although ``bool`` is an ``int`` subclass, isinstance
        #   accepts ONLY ``True``/``False`` and rejects ``0``/``1`` ints.
        # - R4: present but NON-bool (``0`` / ``""`` / ``[]`` / ``{}`` / etc.)
        #   is a malformed payload — a falsey value must NOT be coerced via
        #   ``bool(truncated)`` to ``False`` ("complete scan"). That is the same
        #   fail-OPEN class the R3 ``entries`` fix closed: the differ would
        #   trust an incomplete/garbage walk and proceed to group-zero-apply.
        #   Fail CLOSED as OSError, consistent with the entries-not-a-mapping
        #   branch below.
        if truncated is None:
            truncated_bool = True
        elif isinstance(truncated, bool):
            truncated_bool = truncated
        else:
            raise OSError(
                f"malformed snapshot_workspace payload: truncated not a "
                f"bool ({type(truncated).__name__}) for {root!r}",
            )
        # R2-B / R3: a malformed-but-plausible payload (entries not a mapping,
        # or an entry dict/object missing required keys) must fail CLOSED as
        # OSError per this method's documented "Raises OSError on RPC failure"
        # contract, NOT raise a raw KeyError/AttributeError that escapes the
        # fail-closed envelope. The strict ``Mapping`` check rejects
        # ``[]``/``""``/``0``/``False``/``None``/absent (none are Mapping) as
        # fail-CLOSED, while a genuine empty scan ``{}`` (a Mapping) still
        # passes and decodes to an empty WorkspaceScan.
        if not isinstance(raw_entries, Mapping):
            raise OSError(
                f"malformed snapshot_workspace payload: entries not a "
                f"mapping ({type(raw_entries).__name__}) for {root!r}",
            )
        entries: dict[str, WorkspaceScanEntry] = {}
        for rel_path, e in raw_entries.items():
            try:
                if isinstance(e, dict):
                    entry = WorkspaceScanEntry(
                        rel_path=e["rel_path"],
                        kind=e["kind"],
                        sha256=e.get("sha256"),
                        size=int(e["size"]),
                        mode=int(e["mode"]),
                        link_target=e.get("link_target"),
                    )
                else:
                    entry = WorkspaceScanEntry(
                        rel_path=getattr(e, "rel_path"),
                        kind=getattr(e, "kind"),
                        sha256=getattr(e, "sha256", None),
                        size=int(getattr(e, "size")),
                        mode=int(getattr(e, "mode")),
                        link_target=getattr(e, "link_target", None),
                    )
                # R7b P1-h: the mapping KEY is what the differ keys changes by,
                # so it must be a ``str`` AND agree with the entry's own
                # ``rel_path``. A non-str key (defensive — JSON keys are always
                # strings, but an object-shaped payload need not be) or a
                # key≠rel_path would have the differ key a change by a path that
                # contradicts the entry it carries. Fail CLOSED.
                if not isinstance(rel_path, str) or rel_path != entry.rel_path:
                    raise OSError(
                        f"malformed snapshot_workspace entry: key {rel_path!r} "
                        f"!= rel_path {entry.rel_path!r} for {root!r}",
                    )
                # R7b P1-i: per-kind structural invariants (sha only on
                # regular, link_target only on symlink, valid FileKind).
                _validate_scan_entry(entry, root)
                entries[rel_path] = entry
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                raise OSError(
                    f"malformed snapshot_workspace entry {rel_path!r}: {exc}",
                ) from exc
        return WorkspaceScan(entries=entries, truncated=truncated_bool)
