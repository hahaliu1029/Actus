"""C2 PR-3 §8.3 + §14.3.1 — CoordinatorChildRunner skeleton.

PR-3 scope: ``StopReason`` carrier + ``request_stop`` sole-entry semantics only.
PR-4 fleshes out:
  - finalizers (``_finalize_success`` / ``_finalize_failed`` / ...)
  - 8-checkpoint cancel propagation
  - PatchManifest emission

PR-6 fleshes out:
  - ``_finalize_by_stop_reason`` routing
  - budget watchdogs (token + wallclock)

Constructor accepts envelope-publish deps + mailbox_subscriber at the skeleton
level so PR-4's full implementation does not need a breaking ctor change.
"""
from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any, Optional


class StopReason(StrEnum):
    """C2 PR-3 §14.3.1 — sole authoritative stop classifier."""

    PARENT_CANCEL = "parent_cancel"
    TOKEN_BUDGET = "token_budget"
    WALLCLOCK_BUDGET = "wallclock_budget"


class CoordinatorChildRunner:
    """Skeleton — PR-4 fleshes out finalizers + worker contract.

    PR-3 contract:
      - ``request_stop(reason)`` is the SOLE entry that sets the cancel
        event and ``_stop_reason``. First setter wins (subsequent calls
        do not overwrite reason).
      - ``run_work_unit`` raises NotImplementedError until PR-4 lands the
        full child loop.
    """

    def __init__(
        self,
        *,
        cancel_event: asyncio.Event,
        envelope_factory: Any = None,
        parent_session_id: str = "",
        coordinator_run_id: str = "",
        mailbox_subscriber: Any = None,
    ) -> None:
        self._cancel_event = cancel_event
        self._stop_reason: Optional[StopReason] = None
        self._envelope_factory = envelope_factory
        self._parent_session_id = parent_session_id
        self._coordinator_run_id = coordinator_run_id
        self._mailbox_subscriber = mailbox_subscriber

    def request_stop(self, reason: StopReason) -> None:
        """Sole entry — sets cancel_event + records first-wins stop reason."""
        if self._stop_reason is None:
            self._stop_reason = reason
        self._cancel_event.set()

    async def run_work_unit(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "CoordinatorChildRunner.run_work_unit is a PR-4 deliverable; "
            "PR-3 ships skeleton only."
        )
