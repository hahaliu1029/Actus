"""C3 PR-3c / PR-4 / PR-4.5 — readiness gate for ``build_supervisor_registry``
(codex r4 [HIGH CONTRACT], codex r2 [R2-1, HIGH ARCH], codex r1 [R1-3]).

The supervisor's PR-3a stub dispatch table ACKed ``RESULT_READY`` /
``CANCEL_ACK`` envelopes without calling
``SandboxLifecycleService.destroy`` → silent sandbox leak on every
terminal envelope. ``build_supervisor_registry`` is gated on the
module-level constant ``_PR4_TERMINAL_HANDLERS_READY``.

**PR-4.5 landed** — the readiness gate is now ``True`` and the registry
factory wires ``_pr4_5_agent_service_callback`` (the real
``AgentService.stop_session`` bridge) instead of ``_pr3c_noop_callback``.

These tests now lock the POST-PR-4.5 state:

* ``_PR4_TERMINAL_HANDLERS_READY`` must be ``True`` so the factory
  constructs the registry when ``MAILBOX_SUPERVISOR_ENABLED=true``.
* The factory MUST wire ``_pr4_5_agent_service_callback`` (the spec §7.6
  stop-before-destroy bridge), NOT ``_pr3c_noop_callback``.
* The legacy gate (flag forced False) still raises with the operator-
  readable message intact, so an inadvertent regression that flips the
  flag back to False without removing the gate is still caught.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.interfaces import service_dependencies


def test_pr_4_5_readiness_gate_is_open() -> None:
    """codex r1 [R1-3, HIGH CONTRACT] — PR-4.5 flipped the gate to True
    in the same commit that replaced ``_pr3c_noop_callback`` with
    ``_pr4_5_agent_service_callback``. Both halves moved together per
    spec §7.6.
    """
    assert service_dependencies._PR4_TERMINAL_HANDLERS_READY is True, (
        "_PR4_TERMINAL_HANDLERS_READY must be True after PR-4.5. The "
        "factory now wires _pr4_5_agent_service_callback (the real "
        "AgentService.stop_session bridge) as ctx.agent_service_callback. "
        "If a refactor flips this back to False without also restoring "
        "the noop callback, the gate semantics break."
    )


def test_factory_wires_pr_4_5_callback_bridge() -> None:
    """codex r1 [R1-3] / r14 [R14-5] — verify the registry factory
    actually constructs SupervisorContext with
    ``_pr4_5_agent_service_callback`` (NOT the legacy noop). The
    earlier round only checked the symbol existed; that passed even
    if the factory secretly switched back to ``_pr3c_noop_callback``.
    This test captures ``ctx.agent_service_callback`` via monkeypatch
    and asserts identity equality.
    """
    from unittest.mock import patch, MagicMock

    callback = service_dependencies._pr4_5_agent_service_callback
    assert callable(callback)
    assert callback.__name__ == "_pr4_5_agent_service_callback"

    captured: dict[str, object] = {}

    def _capture_context(ctx, **kwargs):  # noqa: ANN001
        captured["agent_service_callback"] = ctx.agent_service_callback
        return MagicMock()

    redis_client = MagicMock()
    redis_client.client = MagicMock()
    publisher = MagicMock()
    lifecycle = MagicMock()

    # Stub postgres + audit repo so the factory body doesn't try to
    # touch the real DB at import-time. ``DbMailboxEnvelopeAuditRepository``
    # only stores the session_factory at __init__ (no I/O), but
    # ``get_postgres()`` raises when the global pool isn't initialized.
    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    with patch(
        "app.application.services.mailbox_supervisor.MailboxSupervisor",
        side_effect=_capture_context,
    ), patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ):
        registry = service_dependencies.build_supervisor_registry(
            redis_client=redis_client,
            publisher=publisher,
            sandbox_lifecycle_service=lifecycle,
            # PR-9b-A4 INV-A1/A2 — composition-root must always populate
            # these coordinator slots; the gate test passes MagicMocks
            # because it only locks the agent_service_callback identity,
            # not the coordinator port wiring (covered by
            # tests/application/composition/test_coordinator_composition_root_wiring.py).
            coordinator_envelope_store=MagicMock(),
            cost_rollup_service=MagicMock(),
        )
        # Trigger the inner _factory closure by spawning a supervisor.
        registry._factory("root-test")  # type: ignore[attr-defined]

    assert captured["agent_service_callback"] is callback, (
        "build_supervisor_registry must wire "
        "_pr4_5_agent_service_callback as ctx.agent_service_callback; "
        f"got {captured.get('agent_service_callback')!r}"
    )


def test_pr_3c_noop_callback_kept_for_backwards_compat() -> None:
    """The PR-3c placeholder is preserved as a named symbol for tests and
    code paths that still reference it by name. It must remain a no-op
    so accidental wiring doesn't reintroduce the leak."""
    callback = service_dependencies._pr3c_noop_callback
    assert callback.__name__ == "_pr3c_noop_callback"


def test_legacy_gate_message_still_load_bearing(monkeypatch) -> None:
    """Regression — when an operator (or buggy refactor) forces the gate
    back to False, the error message must still cite spec §7.6 and the
    callback-swap contract so they know what's missing.
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
            # PR-9b-A4 INV-A1/A2 — required kwargs after signature
            # extension. This test asserts the gate-False RuntimeError
            # fires BEFORE the factory body runs, so the values are
            # irrelevant; they must just satisfy the signature.
            coordinator_envelope_store=MagicMock(),
            cost_rollup_service=MagicMock(),
        )

    msg = str(exc_info.value).lower()
    assert "pr-4" in msg
    assert "destroy" in msg or "leak" in msg
    assert "spec §7.6" in msg or "spec section 7.6" in msg, (
        "operator-facing error must cite spec §7.6 (stop → destroy)"
    )
