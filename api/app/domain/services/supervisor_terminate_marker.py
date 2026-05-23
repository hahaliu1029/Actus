"""C3 PR-4.5 — supervisor-terminate marker primitive (domain layer).

codex r16 [R16-2, HIGH ARCH] — earlier rounds wired this set inside
``app.interfaces.service_dependencies`` and had ``agent_task_runner``
(domain) import it. That reverse import violates the Clean
Architecture rule in CLAUDE.md ("Domain 层禁止导入 FastAPI/SQLAlchemy
... interfaces"). The marker primitive is purely in-process state
with no framework dependencies, so it belongs in the domain layer;
both the supervisor callback bridge
(``service_dependencies._pr4_5_agent_service_callback``) and the
agent task runner (``agent_task_runner._maybe_stop_child_publisher``)
import from here.

Purpose: when the supervisor's ``CancelRequestHandler`` TERMINATE
path drives a child's terminal status via the callback →
``AgentService.stop_session`` bridge, the supervisor will synthesize
its own ``CANCEL_ACK(force_terminated)``. The child runner MUST NOT
publish its own ``CANCEL_ACK(cancelled)`` for the same correlation
or ``CancelAckHandler`` double-fires destroy and audit shows two
terminal envelopes. The bridge calls
``add_supervisor_terminate_marker`` before ``stop_session``; the
runner consumes it idempotently inside
``_maybe_stop_child_publisher`` to short-circuit its terminal
publish.
"""

from __future__ import annotations


_SUPERVISOR_TERMINATE_SIDS: set[str] = set()


def add_supervisor_terminate_marker(session_id: str) -> None:
    """Mark ``session_id`` as supervisor-initiated terminate.

    Idempotent: a second call is a no-op. Defensive cap of 10000
    entries prevents unbounded growth if a marker is never consumed
    (e.g. runner crashed before terminal path).

    codex r23 [R23-3, LOW PERF] — clear BEFORE adding so the
    just-registered session is never wiped by its own add. The
    earlier ``add + check + clear`` ordering would lose the marker
    for the current call when the cap was hit.
    """
    if len(_SUPERVISOR_TERMINATE_SIDS) >= 10_000:
        _SUPERVISOR_TERMINATE_SIDS.clear()
    _SUPERVISOR_TERMINATE_SIDS.add(session_id)


def consume_supervisor_terminate_marker(session_id: str) -> bool:
    """Pop ``session_id`` from the marker set.

    Returns ``True`` iff the session was present (the callback
    already drove its terminal status via stop_session, so the
    runner should skip its own CANCEL_ACK emission). Idempotent: a
    duplicate call returns False.
    """
    try:
        _SUPERVISOR_TERMINATE_SIDS.remove(session_id)
        return True
    except KeyError:
        return False


def _reset_for_tests() -> None:
    """Test helper — clear all markers between tests."""
    _SUPERVISOR_TERMINATE_SIDS.clear()
