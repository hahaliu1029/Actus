"""C1a spawn capacity constants and exception.

Constants are module-level defaults. Runtime values can be overridden by
SubagentLimitsConfig (core/config.py); see service_dependencies.get_subagent_limits.
"""
from __future__ import annotations

from typing import Final, Literal

MAX_SUBAGENT_DEPTH: Final[int] = 1
MAX_DESCENDANTS_PER_ROOT: Final[int] = 10

SpawnKind = Literal["depth", "descendants"]


class SpawnCapExceeded(ValueError):
    """Raised by SessionService.create_session_with_parent when spawn caps would
    be exceeded. Endpoint layer maps to 429 (descendants cap) or 422 (depth cap)."""

    def __init__(self, kind: SpawnKind, current: int, cap: int) -> None:
        if kind not in ("depth", "descendants"):
            raise ValueError(
                f"SpawnCapExceeded.kind must be 'depth' or 'descendants', got {kind!r}"
            )
        self.kind: SpawnKind = kind
        self.current = current
        self.cap = cap
        super().__init__(
            f"spawn cap exceeded: kind={kind} current={current} cap={cap}"
        )
