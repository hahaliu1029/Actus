"""Execution health monitoring: dual-layer watchdog + cooperative control.

Provides:
- ``ExecutionWatchdog``: Dual timeout monitor (total + idle).
- ``ExecutionControl``: Shared mutable control object injected via configurable.
- ``WatchdogVerdict``: Health evaluation result enum.
- ``_is_progress_event``: Classify domain events as progress signals.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Hashable

from app.domain.models.event import (
    BaseEvent,
    FinishingEvent,
    MessageEvent,
    PlanEvent,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
    ToolEventStatus,
)


class WatchdogVerdict(str, Enum):
    HEALTHY = "healthy"
    SOFT_RECOVER = "soft_recover"
    HARD_TERMINATE = "hard_terminate"


@dataclass
class ExecutionWatchdog:
    """Dual-layer timeout monitor. Not thread-safe; bound to one asyncio loop.

    - **Total timeout**: hard cap on entire graph execution.
    - **Idle timeout**: detects stalled execution (no progress events).

    Call ``evaluate()`` once per timeout cycle (it advances internal state).
    Use ``status`` property for read-only snapshots.
    """

    total_timeout_seconds: float = 0.0
    idle_timeout_seconds: float = 120.0

    _start_time: float = field(default_factory=time.monotonic)
    _last_progress_time: float = field(default_factory=time.monotonic)
    _last_node: str | None = None
    _idle_warnings: int = 0
    _idle_pause_keys: set[Hashable] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_timeout_seconds):
            raise ValueError("total_timeout_seconds must be finite")
        if self.total_timeout_seconds <= 0:
            self.total_timeout_seconds = 0.0
        if (
            not math.isfinite(self.idle_timeout_seconds)
            or self.idle_timeout_seconds <= 0
        ):
            raise ValueError("idle_timeout_seconds must be finite and > 0")

    def record_progress(self, node_name: str | None = None) -> None:
        """Called when a user-visible progress signal arrives."""
        self._last_progress_time = time.monotonic()
        self._idle_warnings = 0
        if node_name:
            self._last_node = node_name

    def pause_idle(self, key: Hashable) -> None:
        """Pause idle evaluation for one structured lifecycle owner.

        Keys form a set rather than a boolean so overlapping coordinator runs
        cannot resume one another accidentally. Re-registering a key is
        idempotent.
        """
        self._idle_pause_keys.add(key)

    def resume_idle(self, key: Hashable) -> None:
        """Release one idle-pause owner, resetting idle on the last release."""
        if key not in self._idle_pause_keys:
            return
        self._idle_pause_keys.remove(key)
        if not self._idle_pause_keys:
            self._last_progress_time = time.monotonic()
            self._idle_warnings = 0

    def _total_exceeded(self) -> bool:
        """Check if total_timeout has been exceeded.

        ``total_timeout_seconds <= 0`` means unlimited (no total cap).
        """
        if self.total_timeout_seconds <= 0:
            return False
        return (time.monotonic() - self._start_time) >= self.total_timeout_seconds

    def evaluate(self) -> WatchdogVerdict:
        """Evaluate health and advance state machine.

        Call once per timeout cycle (typically from idle TimeoutError branch).
        This method has side effects (increments ``_idle_warnings``).
        """
        if self._total_exceeded():
            return WatchdogVerdict.HARD_TERMINATE
        if self._idle_pause_keys:
            return WatchdogVerdict.HEALTHY
        now = time.monotonic()
        idle = now - self._last_progress_time
        if idle >= self.idle_timeout_seconds:
            self._idle_warnings += 1
            if self._idle_warnings >= 2:
                return WatchdogVerdict.HARD_TERMINATE
            return WatchdogVerdict.SOFT_RECOVER
        return WatchdogVerdict.HEALTHY

    def check_total_only(self) -> bool:
        """Returns True if total_timeout has been exceeded.

        Use this on every event (not just idle) so total_timeout is checked
        even when the graph is still producing output. Zero side effects.
        """
        return self._total_exceeded()

    @property
    def status(self) -> WatchdogVerdict:
        """Read-only health snapshot. No side effects."""
        if self._total_exceeded():
            return WatchdogVerdict.HARD_TERMINATE
        if self._idle_pause_keys:
            return WatchdogVerdict.HEALTHY
        now = time.monotonic()
        idle = now - self._last_progress_time
        if idle >= self.idle_timeout_seconds:
            if self._idle_warnings >= 2:
                return WatchdogVerdict.HARD_TERMINATE
            if self._idle_warnings >= 1:
                return WatchdogVerdict.SOFT_RECOVER
        return WatchdogVerdict.HEALTHY

    @property
    def last_node(self) -> str | None:
        return self._last_node

    @property
    def idle_seconds(self) -> float:
        if self._idle_pause_keys:
            return 0.0
        return time.monotonic() - self._last_progress_time

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time


@dataclass
class ExecutionControl:
    """Shared mutable control object. Injected via ``configurable``.

    Nodes read ``should_terminate`` from config to cooperatively exit.
    Bridge writes ``should_terminate`` when watchdog triggers.
    """

    should_terminate: bool = False
    idle_recovery_hint: str | None = None


def _is_progress_event(event: BaseEvent) -> bool:
    """Classify whether an event signals user-visible progress.

    Progress events reset the idle watchdog timer. Internal maintenance
    events (context compaction, error, control transfer) do not.
    """
    if isinstance(event, (MessageEvent, TitleEvent, FinishingEvent)):
        return True
    if isinstance(event, ToolEvent) and event.status == ToolEventStatus.CALLED:
        return True
    if isinstance(event, StepEvent) and event.status == StepEventStatus.COMPLETED:
        return True
    if isinstance(event, PlanEvent):
        return True
    return False


def _should_terminate(config: dict) -> bool:
    """Helper to check should_terminate from config.configurable."""
    control = config.get("configurable", {}).get("execution_control")
    return bool(control and control.should_terminate)
