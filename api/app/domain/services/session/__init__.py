"""Session state machine subpackage.

Per-session control mode owner. Owns sessions.status + sessions.mode_revision
writes (INV-4-hard ships in A4-1; PE-0 enforces INV-4-soft).
"""

from app.domain.services.session.session_state_machine import (  # noqa: F401
    SessionStateMachine,
)
