"""C3 PR-3c / PR-4 / PR-4.5 — fail-closed gate for ``build_supervisor_registry``
(codex r4 [HIGH CONTRACT], codex r2 [R2-1, HIGH ARCH]).

The supervisor's PR-3a stub dispatch table ACKed ``RESULT_READY`` /
``CANCEL_ACK`` envelopes without calling
``SandboxLifecycleService.destroy`` → silent sandbox leak on every
terminal envelope. ``build_supervisor_registry`` is gated on the
module-level constant ``_PR4_TERMINAL_HANDLERS_READY``.

PR-4 ships the real terminal handlers + cascade pipeline behind the gate
(monkeypatched ``True`` in tests). The flag itself stays ``False`` until
PR-4.5 replaces ``_pr3c_noop_callback`` with a real
``AgentService.stop_session`` bridge — otherwise CancelRequestHandler
TERMINATE step 1 (`agent_service_callback`) silently does nothing and
step 2 destroys a still-running asyncio.Task, violating spec §7.6
"stop → destroy" ordering.

These tests lock both halves of the gate:

* While the callback remains ``_pr3c_noop_callback`` (the PR-3c
  placeholder), :func:`build_supervisor_registry` MUST raise
  ``RuntimeError``. The flag is the single edit site PR-4.5 will flip
  alongside wiring the real callback.
* The factory monkeypatched ``True`` (PR-4.5 + future) still constructs
  the registry with the same ``_pr3c_noop_callback`` shape today; that
  combination is intentionally noisy at the gate (False default) so a
  premature flip without callback wiring fails closed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.interfaces import service_dependencies


def test_build_supervisor_registry_fails_closed_while_callback_is_noop() -> None:
    """codex r2 [R2-1] — current production state is the noop-callback +
    flag-False combination. Lifespan MUST refuse to construct the registry
    so operators can't silently land in the half-baked "destroy but never
    stop" path documented in spec §7.6.
    """
    assert service_dependencies._PR4_TERMINAL_HANDLERS_READY is False, (
        "_PR4_TERMINAL_HANDLERS_READY must be False while "
        "_pr3c_noop_callback (line ~660 in service_dependencies.py) is "
        "still wired as agent_service_callback. PR-4.5 flips this to True "
        "in the same commit that replaces the no-op with a real "
        "AgentService.stop_session bridge. If a refactor sneaks this "
        "constant to True without wiring the callback, CancelRequest "
        "TERMINATE step 1 (stop) silently no-ops and step 2 destroys a "
        "still-running task → spec §7.6 ordering violated."
    )


def test_build_supervisor_registry_fails_closed_when_pr4_flag_false(
    monkeypatch,
) -> None:
    """Regression — confirm the gate raises with operator-readable message
    so the misconfiguration is visible at lifespan startup.
    """
    monkeypatch.setattr(
        service_dependencies, "_PR4_TERMINAL_HANDLERS_READY", False
    )

    redis_client_stub = MagicMock()
    publisher_stub = MagicMock()
    lifecycle_stub = MagicMock()

    with pytest.raises(RuntimeError) as exc_info:
        service_dependencies.build_supervisor_registry(
            redis_client=redis_client_stub,
            publisher=publisher_stub,
            sandbox_lifecycle_service=lifecycle_stub,
        )

    msg = str(exc_info.value).lower()
    # Pin keywords so the operator-facing error stays load-bearing.
    assert "pr-4" in msg
    assert "stub" in msg
    assert "destroy" in msg or "leak" in msg
    # Codex r4 [R4-5, MEDIUM DOC] — the message now points operators at the
    # PR-4.5 flip site (callback swap + flag flip in the SAME commit) so
    # operators surfacing this error don't go looking for a missing PR-4
    # follow-up. Lock the PR-4.5 reference + the gate-test pointer so a
    # future copy-edit doesn't silently regress this guidance.
    assert "pr-4.5" in msg, (
        "operator-facing message must direct ops to PR-4.5 (the planned "
        "callback swap + flag flip site); otherwise the error reads as "
        "'wait for PR-4' which already shipped."
    )
    assert "stop_session" in msg.replace("_", "").replace("`", "") or (
        "stop_session" in msg
    ), (
        "operator-facing message must name AgentService.stop_session as "
        "the planned PR-4.5 replacement for _pr3c_noop_callback so ops "
        "know what the half-baked state looks like."
    )
    assert "spec §7.6" in msg or "spec section 7.6" in msg, (
        "operator-facing message must cite spec §7.6 (stop → destroy) so "
        "ops can find the contract that the gate is enforcing."
    )


def test_callback_is_still_noop_pending_pr_4_5() -> None:
    """codex r2 [R2-1] — explicit assertion that the registry factory is
    still wired to ``_pr3c_noop_callback``. PR-4.5 is the planned site to
    flip both this callback AND ``_PR4_TERMINAL_HANDLERS_READY``; flipping
    only one half causes silent contract violations. If PR-4.5 replaces
    the no-op, update this test to reference the new bridge function name
    rather than ``_pr3c_noop_callback``.
    """
    # Locate the function defined in service_dependencies and confirm it
    # is the documented PR-3c no-op placeholder. Without this lock, a
    # diff that swaps the callback name without also flipping the flag
    # (or vice versa) would skip the gate test.
    callback = service_dependencies._pr3c_noop_callback
    assert callback.__name__ == "_pr3c_noop_callback"
    assert "PR-3c placeholder" in (callback.__doc__ or ""), (
        "_pr3c_noop_callback docstring must keep flagging itself as the "
        "PR-3c placeholder so PR-4.5 has a clear edit site."
    )
