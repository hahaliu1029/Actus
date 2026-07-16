"""SPM PR-1b: lazy ``ParentSandboxPort`` for the ``on_demand`` × coordinator path.

r18/codex R17-Q1 — In ``on_demand`` mode the flow's ``self._sandbox_accessor``
holds NO raw handle until the first provision. The coordinator's
``_build_config`` historically wrapped the raw handle eagerly via
``parent_sandbox_adapter_factory(handle)``; doing that with an accessor (or with
``None``) would crash on the FIRST coordinator parent I/O (seed / extract).
``spec §4`` forbids only ``off × coordinator`` — ``on_demand × coordinator`` is
supported (the coordinator needs the parent sandbox to seed / extract). This
port defers the wrap: every ``ParentSandboxPort`` method awaits
``accessor.get()`` (provision-if-needed) and delegates to a per-generation
cached adapter.

r21/codex R21-U6 — The adapter MUST come from the injected
``parent_sandbox_adapter_factory`` (``SandboxHandle -> ParentSandboxPort``);
constructing ``ParentSandboxAdapter`` directly would bypass the coordinator DI
invariant (``coordinator_runtime_deps.py`` docstring) and break the injection
structure test (``test_coordinator_parent_sandbox_adapter_injection``).

INV-SPM-2 — NEW code only. In PR-1b the flow builds this ONLY for a non-Eager
accessor (``peek()`` is ``None``); the Eager (``always`` / child) path keeps the
direct ``factory(handle)`` wrap so ``always`` behavior stays byte-identical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from app.domain.external.parent_sandbox import ParentSandboxPort

if TYPE_CHECKING:
    from app.domain.external.parent_sandbox import SandboxPathCheck, WorkspaceScan
    from app.domain.external.sandbox import SandboxAccessor, SandboxHandle


class LazyParentSandboxPort(ParentSandboxPort):
    """Deferred ``ParentSandboxPort`` bound to a ``SandboxAccessor`` + factory.

    Provision (via ``accessor.get()``) happens on the FIRST parent I/O, not at
    construction — so an ``on_demand`` coordinator that never touches the parent
    sandbox (pure chat) never provisions. The wrapped adapter is cached per
    sandbox *generation*: if the underlying handle identity changes (teardown +
    re-provision) ``get()`` returns a new handle and the adapter is rebuilt via
    the same injected factory.
    """

    def __init__(
        self,
        accessor: "SandboxAccessor",
        adapter_factory: "Callable[[SandboxHandle], ParentSandboxPort]",
    ) -> None:
        self._accessor = accessor
        self._factory = adapter_factory
        # Per-generation cache: keyed by handle identity so a same-generation
        # re-entry reuses the wrapped adapter (factory NOT re-invoked).
        self._cached_handle: Optional["SandboxHandle"] = None
        self._cached_adapter: Optional[ParentSandboxPort] = None

    async def _delegate(self) -> ParentSandboxPort:
        """Provision-if-needed, then return the per-generation cached adapter."""
        handle = await self._accessor.get()
        if handle is not self._cached_handle or self._cached_adapter is None:
            self._cached_adapter = self._factory(handle)
            self._cached_handle = handle
        return self._cached_adapter

    async def compute_digest(self, path: str) -> Optional[str]:
        return await (await self._delegate()).compute_digest(path)

    async def exists(self, path: str) -> bool:
        return await (await self._delegate()).exists(path)

    async def check_path(self, path: str) -> "SandboxPathCheck":
        return await (await self._delegate()).check_path(path)

    async def read_file(self, path: str) -> bytes:
        return await (await self._delegate()).read_file(path)

    async def atomic_write_file(self, path: str, content: bytes) -> None:
        await (await self._delegate()).atomic_write_file(path, content)

    async def delete_file(self, path: str) -> None:
        await (await self._delegate()).delete_file(path)

    async def kill_all_shell_sessions(self) -> None:
        await (await self._delegate()).kill_all_shell_sessions()

    async def snapshot_workspace(
        self,
        root: str = "/home/ubuntu",
        *,
        max_paths: int,
        max_files: int,
        max_total_bytes: int,
        max_seconds: float,
    ) -> "WorkspaceScan":
        return await (await self._delegate()).snapshot_workspace(
            root,
            max_paths=max_paths,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
            max_seconds=max_seconds,
        )
