"""C2 v1 ParentSandboxPort (spec §10.5).

Domain Protocol the coordinator + PatchApplier use to operate the **parent
session's** sandbox. Intentionally narrow:

- **Read-side**: ``compute_digest`` / ``exists`` / ``read_file`` — needed
  by the reducer's digest-drift check and the applier's preflight.
- **Write-side**: ``atomic_write_file`` / ``delete_file`` — needed by the
  applier's apply-and-rollback path.

**M1 type-level invariant (spec §10.5): NO destroy(). NO suspend().**
Sandbox lifecycle (create, destroy, suspend, recover) stays exclusively
with ``MailboxSupervisor.*Handler`` (C3). Exposing destroy() on the
coordinator's port would let a buggy PatchApplier deallocate the parent's
sandbox mid-write — guarding against that at the type level is cheaper
than a runtime guard. Adapters MUST NOT expose destroy() either, even if
the underlying SandboxHandle supports it.

The adapter (``infrastructure/external/sandbox/parent_sandbox_adapter``)
wraps the live ``SandboxHandle`` (which returns ``BinaryIO`` /
``ToolResult``) and translates to the bytes / bool / None contract here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Protocol


FileKind = Literal[
    "missing", "regular", "directory", "symlink",
    "fifo", "socket", "block", "char", "other",
]


@dataclass(frozen=True)
class SandboxPathCheck:
    """Result of an inode probe over the parent sandbox (domain-pure).

    Mirrors the sandbox-side ``FileCheckResult{exists, kind}`` value set;
    cannot share a module (api vs sandbox package). ``exists`` keeps
    ``os.path.exists`` semantics (follows symlinks); ``kind`` is the
    ``os.lstat`` inode type (does NOT follow the final symlink).
    """

    exists: bool
    kind: FileKind


@dataclass(frozen=True)
class WorkspaceScanEntry:
    """One non-directory inode captured by ``snapshot_workspace`` (S2 §3.1).

    ``kind`` is the ``os.lstat`` inode type and is NEVER ``"directory"`` —
    directories are traversed, not emitted. ``sha256`` is the content digest,
    set ONLY when ``kind == "regular"``; ``link_target`` is the ``os.readlink``
    referent, set ONLY when ``kind == "symlink"``. The diff identity tuple the
    differ compares is ``(kind, sha256, size, mode, link_target)``.
    """

    rel_path: str
    kind: "FileKind"
    sha256: Optional[str]
    size: int
    mode: int
    link_target: Optional[str]


@dataclass(frozen=True)
class WorkspaceScan:
    """Capped, directory-excluded snapshot of a workspace subtree (S2 §3.1).

    ``entries`` is keyed by ``rel_path`` (workspace-relative,
    directory-qualified); directories are EXCLUDED. ``truncated`` is True iff
    any cap (paths / files / total-bytes / seconds) aborted the walk — the
    differ treats a truncated scan as a hard fail-closed signal (group
    zero-apply).
    """

    entries: Mapping[str, WorkspaceScanEntry]
    truncated: bool


class ParentSandboxPort(Protocol):
    """Narrow read/write surface over the parent session's sandbox.

    This port is a **neutral, read-only file-surface used over BOTH the
    parent and the child handle** (S2 §3.1 [B9]): the same adapter wraps the
    parent session's ``SandboxHandle`` and a coordinator child's handle
    (``coordinator_child_runner_starter.py:246``). ``snapshot_workspace`` is
    read-only and (per M1) MUST NOT expose any lifecycle op.

    Method semantics are spec-defined (§10.5):

    - ``compute_digest(path)`` returns SHA-256 hex of the file at ``path``,
      or ``None`` if the file does not exist. Used by the reducer's
      drift check (§9.3 step 5) — missing-file => no drift signal, the
      applier preflight will catch it via ``exists()``.
    - ``exists(path)`` is a thin existence probe — preflight uses this
      before opening modify/delete operations.
    - ``check_path(path)`` is the inode-typed probe the applier preflight
      uses to reject a direct special-file target (FIFO/socket/block/char)
      before the read (S1b 2a). ``exists()`` is kept for the bool-only
      callers + the ``add`` branch.
    - ``read_file(path) -> bytes`` returns raw bytes; the applier reads
      to snapshot original content for rollback.
    - ``atomic_write_file(path, content)`` writes ``content`` to ``path``
      atomically. **Contract** [codex R7 P1 + R8 P1]: this method
      either succeeds (file at ``path`` now has the new contents) OR
      raises with NO observable final-path-content side effect on the
      sandbox. The applier's rollback path is built around this
      contract. Backends emulate via tmp + fsync + rename.

      *v1 status*: **enforced sandbox-side as of S1** — the live
      ``SandboxHandle.upload_file`` HTTP RPC writes via
      ``mkstemp + fsync + os.replace`` (see
      ``sandbox/app/services/file.py`` ``_atomic_write_bytes`` and the
      S1 design spec); a mid-write exception no longer leaves a
      truncated file for regular-file / symlink / new-file targets. (An
      EXISTING special file — FIFO/socket/device — uses a non-atomic
      write-through per S1 D12 on the AGENT path; the coordinator apply/seed
      path refuses a special target with ``OSError(EINVAL)`` per S1b 2b.
      Path-transparency is the permanent contract — Gap B path resolution is
      closed by G2b.)
    - ``delete_file(path)`` removes the file at ``path``. **Idempotent
      as of S1**: deleting an already-absent file is success
      (terminal-absent) — the live RPC uses ``os.remove`` catching
      ENOENT; other OSError still surface.

    Failure modes for ``read_file`` / ``atomic_write_file`` /
    ``delete_file`` are intentionally NOT specified at the Protocol level
    — the adapter raises whatever the sandbox client raises (typically
    OSError-family). Callers are expected to catch and translate.
    """

    async def compute_digest(self, path: str) -> Optional[str]: ...
    async def exists(self, path: str) -> bool: ...
    async def check_path(self, path: str) -> "SandboxPathCheck": ...
    async def read_file(self, path: str) -> bytes: ...
    async def atomic_write_file(self, path: str, content: bytes) -> None: ...
    async def delete_file(self, path: str) -> None: ...
    async def snapshot_workspace(
        self,
        root: str = "/home/ubuntu",
        *,
        max_paths: int,
        max_files: int,
        max_total_bytes: int,
        max_seconds: float,
    ) -> "WorkspaceScan":
        """Capped, directory-excluded content snapshot of ``root`` (S2 §3.1).

        Walks ``root`` (excluding ``/sandbox`` + ``.memory``), emitting one
        ``WorkspaceScanEntry`` per non-directory inode keyed by
        directory-qualified workspace-relative path. Aborts the walk and sets
        ``truncated=True`` if any cap is exceeded. Indeterminate-kind inodes
        fail CLOSED (treated as a scan failure / truncation), opposite of
        ``check_path``'s fail-open. Raises ``OSError`` on RPC failure.
        """
        ...
