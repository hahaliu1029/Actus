"""Provision flight registry + deletion tombstones (spec §5.2c, DD-17).

Lifespan-scoped, in-memory, single-event-loop. Owned exclusively by
SandboxLifecycleService — no other writer (INV-SPM-10).
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Optional

FlightOutcome = Literal["destroy", "delete", "quiesce"]

_TOMBSTONE_CAP = 10_000


@dataclass
class ProvisionFlight:
    attempt: str = field(default_factory=lambda: uuid.uuid4().hex)
    invalidated: Optional[FlightOutcome] = None
    container_disposer: Optional[Callable[[], Awaitable[None]]] = None


class ProvisionFlightTable:
    def __init__(self) -> None:
        self._flights: dict[str, ProvisionFlight] = {}
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def begin(self, session_id: str) -> ProvisionFlight:
        if session_id in self._flights:
            raise RuntimeError(
                f"provision flight already active for session {session_id}"
            )
        flight = ProvisionFlight()
        self._flights[session_id] = flight
        return flight

    def get(self, session_id: str) -> Optional[ProvisionFlight]:
        return self._flights.get(session_id)

    def finish(self, session_id: str) -> None:
        self._flights.pop(session_id, None)

    def invalidate(self, session_id: str, outcome: FlightOutcome) -> bool:
        if outcome == "delete":
            # delete 恒 tombstone：destroy 返回后、session 行硬删前，任何后续
            # bind_new（含全新 flight 的第二次）都必须被入口检查拒绝。
            self._tombstones[session_id] = None
            self._tombstones.move_to_end(session_id)
            while len(self._tombstones) > _TOMBSTONE_CAP:
                self._tombstones.popitem(last=False)
        flight = self._flights.get(session_id)
        if flight is not None:
            # FIX-J: delete is STICKY. A later quiesce/destroy must not DOWNGRADE
            # an earlier delete mark — the tombstone still blocks rebinding, but
            # SandboxProvisionInvalidated.invalidation_outcome would otherwise
            # misreport a deleted session as quiesced/destroyed. Upgrades TO
            # "delete" are always allowed; downgrades FROM "delete" are ignored.
            # Still return True (an active flight was — already — marked).
            if flight.invalidated != "delete":
                flight.invalidated = outcome
            return True
        return False

    def is_tombstoned(self, session_id: str) -> bool:
        return session_id in self._tombstones
