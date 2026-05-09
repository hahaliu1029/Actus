"""Supervisor admission and FSM rejection errors."""

from __future__ import annotations


class SupervisorContractError(Exception):
    """Raised when supervisor admission or transition is rejected."""

    __slots__ = ("rejection_code", "from_phase", "to_phase", "reason")

    def __init__(
        self,
        rejection_code: str,
        from_phase: str,
        to_phase: str,
        reason: str,
    ) -> None:
        super().__init__(f"[{rejection_code}] {reason} ({from_phase} -> {to_phase})")
        self.rejection_code = rejection_code
        self.from_phase = from_phase
        self.to_phase = to_phase
        self.reason = reason
