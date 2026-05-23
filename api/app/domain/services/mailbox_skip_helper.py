"""Helper gate used by AgentService to skip suspend/destroy for mailbox-plane children.

C3 spec §11.4 + R3 P1.2 — when a session is a mailbox-plane subagent, the
AgentService cancel/complete/stop paths MUST NOT call
``sandbox_lifecycle_service.suspend()`` or ``.destroy()``. MailboxSupervisor
owns the destroy path for those sessions (single-writer M1 invariant); a
legacy ``suspend()`` here races the supervisor's ``destroy()`` and corrupts
the binding state machine.

The function is intentionally tiny and pure so the AST CI gate
(``tests/ci/test_no_direct_sandbox_destroy.py``, spec §13.3 + R3 P2.2) can
syntactically detect call sites that reference it. Callers must keep the
identifier ``_should_skip_mailbox_lifecycle`` literally present in the
function body that wraps a ``.suspend(...)`` / ``.destroy(...)`` call.
"""

from __future__ import annotations

from app.domain.models.session import Session


def _should_skip_mailbox_lifecycle(session: Session) -> bool:
    """True when AgentService must NOT suspend/destroy this session.

    Conditions (BOTH must hold):
      1. ``session.worker_type == 'subagent'``
      2. ``session.subagent_control_plane == 'mailbox'``

    Root sessions and legacy/None-plane children fall through to the existing
    suspend path. Phase-1 pre-PR-5 callers see ``False`` for every row
    because ``SessionService.create_session_with_parent`` persists
    ``subagent_control_plane='legacy'`` whenever
    ``settings.mailbox_supervisor_enabled`` is False (the .env.example
    default until PR-5). ``None`` is the legacy value for pre-C3 rows
    that pre-date the column; it is still treated as legacy here for
    defense in depth.
    """
    return (
        getattr(session, "worker_type", None) == "subagent"
        and getattr(session, "subagent_control_plane", None) == "mailbox"
    )
