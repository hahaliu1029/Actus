"""SandboxHandle concrete implementation with __getattr__ whitelist proxy.

Caller code should never import this module directly — they import and annotate
with ``SandboxHandle`` (Protocol) from ``app.domain.external.sandbox``.
``SandboxLifecycleService.acquire()`` returns ``cast(SandboxHandle, impl)``.

See spec §8.2 for design rationale.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Any

from app.domain.errors.sandbox_lifecycle import SandboxPoisonedError
from app.domain.external.sandbox import Sandbox

if TYPE_CHECKING:
    from app.infrastructure.external.sandbox.sandbox_registry import SandboxRegistry

logger = logging.getLogger(__name__)

# All Sandbox Protocol async methods that should be forwarded through
# generation-checked dispatch. Kept as class-level constant for
# ``hasattr``-free feature detection (spec §8.2 caveat).
SANDBOX_FORWARDED_METHODS: frozenset[str] = frozenset(
    {
        "exec_command",
        "read_shell_output",
        "wait_process",
        "write_shell_input",
        "resize_shell_session",
        "kill_process",
        "write_file",
        "read_file",
        "check_file_exists",
        "delete_file",
        "list_files",
        "replace_in_file",
        "search_in_file",
        "find_files",
        "upload_file",
        "download_file",
        "ensure_sandbox",
        "renew_timeout_lease",
        "get_browser",
        "snapshot_workspace",
        # [S2 PR-4 §3.2] shell quiesce before the POST snapshot scan.
        "kill_all_shell_sessions",
    }
)


class SandboxHandleImpl:
    """Lifecycle-aware sandbox wrapper with generation checking.

    Forwards Sandbox Protocol methods through a ``__getattr__`` whitelist
    proxy. Each forwarded call:

    1. Enrolls the current ``asyncio.Task`` in the registry's inflight set
    2. Checks generation (raises ``SandboxPoisonedError`` if stale)
    3. Delegates to the underlying ``Sandbox`` instance
    4. Discharges the task in ``finally``

    Properties (id, cdp_url, shell_ws_url, vnc_url) are cached at acquire
    time — no generation check needed since URLs don't change.
    """

    __slots__ = (
        "_sandbox",
        "_session_id",
        "_generation",
        "_registry",
        "_released",
        # Cached properties from underlying sandbox
        "_id",
        "_cdp_url",
        "_shell_ws_url",
        "_vnc_url",
    )

    def __init__(
        self,
        sandbox: Sandbox,
        session_id: str,
        generation: int,
        registry: SandboxRegistry,
    ) -> None:
        self._sandbox = sandbox
        self._session_id = session_id
        self._generation = generation
        self._registry = registry
        self._released = False
        # Cache read-only properties at acquire time
        self._id = sandbox.id
        self._cdp_url = sandbox.cdp_url
        self._shell_ws_url = sandbox.shell_ws_url
        self._vnc_url = sandbox.vnc_url

    # ── Read-only properties (no generation check) ──

    @property
    def id(self) -> str:
        return self._id

    @property
    def cdp_url(self) -> str:
        return self._cdp_url

    @property
    def shell_ws_url(self) -> str:
        return self._shell_ws_url

    @property
    def vnc_url(self) -> str:
        return self._vnc_url

    @property
    def generation(self) -> int:
        return self._generation

    # ── Generation check ──

    def _check_generation(self) -> None:
        """Raise SandboxPoisonedError if this handle's generation is stale."""
        current_gen = self._registry.get_generation(self._session_id)
        if current_gen is None or current_gen != self._generation:
            raise SandboxPoisonedError(
                session_id=self._session_id,
                expected_generation=self._generation,
                actual_generation=current_gen if current_gen is not None else -1,
            )

    # ── Checked call with inflight tracking (spec §8.5) ──

    async def _checked_call(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Forward a method call with generation check + inflight enrollment."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError(
                "SandboxHandle methods must be called inside an asyncio Task"
            )

        self._registry.enroll_inflight(self._session_id, task)
        try:
            self._check_generation()
            bound = getattr(self._sandbox, method_name)
            return await bound(*args, **kwargs)
        finally:
            self._registry.discharge_inflight(self._session_id, task)

    # ── __getattr__ whitelist proxy ──

    def __getattr__(self, name: str) -> Any:
        if name in SANDBOX_FORWARDED_METHODS:
            return functools.partial(self._checked_call, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    # ── Lifecycle ──

    def release(self) -> None:
        """Release this handle from the registry. Idempotent."""
        if not self._released:
            self._released = True
            self._registry.release_handle(self._session_id, self)

    async def __aenter__(self) -> SandboxHandleImpl:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.release()
