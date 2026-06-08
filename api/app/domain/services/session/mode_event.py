"""A4-2: canonical SessionModeChangedEvent builder + sink type.

Single construction authority for SessionModeChangedEvent. The SSM's
emit_session_mode_changed (session_state_machine.py) is the only caller of
build_session_mode_changed_event; every one of the 8 logical control-mode emits
routes through it (enforced by the production-only single-construction AST guard
in test_mode_change_emit_guard.py). Domain-only imports (no FastAPI/SQLAlchemy)."""
from __future__ import annotations

from typing import Awaitable, Callable, Optional

from app.domain.models.event import SessionModeChangedEvent
from app.domain.models.session import SessionStatus

# A sink consumes (session_id, event) and performs the caller-owned runtime emit
# (live-stream put and/or DB add_event and/or seq-stamp/idle-touch). Returns None.
ModeChangedEventSink = Callable[[str, SessionModeChangedEvent], Awaitable[None]]


def build_session_mode_changed_event(
    *,
    to: SessionStatus | str,
    from_mode: Optional[str],
    reason: str,
    mode_revision: Optional[int],
) -> SessionModeChangedEvent:
    """Single canonical constructor for SessionModeChangedEvent.

    ``to`` accepts a SessionStatus (runner / inline sites pass the enum) or a
    str (agent_service helper sites pass ``.value``); normalized to the event's
    str field. ``from_mode`` is ALWAYS str|None at all 8 call sites (runner
    "running"; helper sites a str; inline "takeover"/None) — no SessionStatus
    normalization needed. All payload fields are server-fixed (INV-6)."""
    return SessionModeChangedEvent(
        to=to.value if isinstance(to, SessionStatus) else to,
        from_mode=from_mode,
        reason=reason,
        mode_revision=mode_revision,
    )
