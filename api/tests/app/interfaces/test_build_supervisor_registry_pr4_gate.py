"""C3 PR-3c — fail-closed gate for ``build_supervisor_registry`` (codex r4 [HIGH CONTRACT]).

If an operator flips ``MAILBOX_SUPERVISOR_ENABLED=True`` at PR-3c
(before PR-4 ships real terminal handlers), the supervisor's PR-3a
stub dispatch table would ACK ``RESULT_READY`` / ``CANCEL_ACK``
envelopes without calling ``SandboxLifecycleService.destroy`` → silent
sandbox leak on every terminal envelope.

``build_supervisor_registry`` is gated on the module-level constant
``_PR4_TERMINAL_HANDLERS_READY``. PR-4 will flip it to ``True`` in the
same commit that wires real terminal handlers; until then, the
function MUST raise so lifespan fails closed on misconfiguration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.interfaces import service_dependencies


def test_build_supervisor_registry_fails_closed_when_pr4_not_ready(monkeypatch) -> None:
    """Operator flipping the flag at PR-3c MUST trigger a hard failure
    with a message that names the PR4 dependency, so lifespan crashes
    visibly instead of silently draining terminal envelopes.
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


def test_build_supervisor_registry_constant_defaults_to_false() -> None:
    """Pin the safe default. PR-4 flips this; if a future PR flips it
    early without shipping real handlers, this test fails fast.
    """
    assert service_dependencies._PR4_TERMINAL_HANDLERS_READY is False, (
        "_PR4_TERMINAL_HANDLERS_READY must default False until PR-4 "
        "ships ResultReadyHandler + CancelAckHandler with real destroy "
        "side-effects (spec §7.3)."
    )
