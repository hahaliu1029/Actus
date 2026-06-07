"""Session state machine subpackage.

Per-session control mode owner. Owns sessions.status + sessions.mode_revision
writes (INV-4-hard SHIPPED in A4-1: the SSM subpackage is the sole CALLER of
the sessions.status repo mutators, enforced by Gate A/B in
tests/invariants/test_inv4_ssm_single_writer.py).
"""

from app.domain.services.session.session_state_machine import (  # noqa: F401
    SessionStateMachine,
)
