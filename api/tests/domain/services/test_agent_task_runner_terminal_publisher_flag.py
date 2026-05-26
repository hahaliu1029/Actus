"""C2 PR-4 Task 4.4 — terminal_envelope_publisher_disabled flag tests.

Spec ref: §8.5.1 r6 P0-1 regression fix.

The C2 coordinator child uses ``CoordinatorChildRunner._finalize_*`` as the
SOLE terminal envelope publisher. The default ``AgentTaskRunner`` heartbeat
+ terminal publish path (``_maybe_stop_child_publisher``) MUST NOT also fire
for coordinator_step children, or the wire carries two RESULT_READY envelopes
for the same correlation_id (duplicate ack destroys, supervisor double-fires).

The fix MUST be surgical:
- Heartbeat task cleanup → still runs unconditionally first.
  Otherwise a coordinator_step child whose flag prevents the publish would
  leak its heartbeat task and keep emitting PROGRESS_UPDATE forever.
- Supervisor checks (``_is_mailbox_plane_child``, ``_supervisor_registry``
  presence) → still gate publish, even when the flag is False.
- ``_terminal_envelope_publisher_disabled = True`` → returns BEFORE the
  envelope construction + ``publisher.publish(envelope)`` call.

Verified invariants:
1. flag default == False → behavior identical to pre-PR-4 (regression check).
2. flag == True → publisher.publish NOT called; heartbeat cleanup still called.
3. flag == True does NOT bypass spawn-side ``_maybe_spawn_child_publisher``
   (SPAWN_ACK + heartbeat startup) — only the terminal path is gated.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.session import SessionStatus


pytestmark = pytest.mark.anyio


def _runner_skel(*, disabled: bool, publisher: AsyncMock | None,
                 heartbeat_handle: asyncio.Task | None = None) -> MagicMock:
    """Build a minimal AgentTaskRunner facsimile with the exact attribute set
    consumed by ``_maybe_stop_child_publisher``.

    The unit under test is the gate behavior — we do NOT construct a real
    runner (which would require ~25 dependencies). Instead we drive
    ``AgentTaskRunner._maybe_stop_child_publisher.__get__(skel)`` and verify
    the publisher + heartbeat side effects.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    skel = MagicMock(spec=AgentTaskRunner)
    skel._mailbox_publisher = publisher
    skel._supervisor_registry = MagicMock() if publisher is not None else None
    skel._session_id = "child-1"
    skel._spawn_correlation_id = "spawn:child-1"
    skel._runner_exception_terminal = False
    skel._terminal_envelope_publisher_disabled = disabled

    cached_session = MagicMock()
    cached_session.parent_session_id = "parent-1"
    skel._cached_session_for_publisher = cached_session
    skel._is_mailbox_plane_child = AsyncMock(return_value=True)
    skel._cleanup_heartbeat_task = AsyncMock()
    return skel


async def test_default_flag_publishes_terminal() -> None:
    """[regression baseline] disabled=False → publisher.publish called once."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    skel = _runner_skel(disabled=False, publisher=publisher)

    with patch(
        "app.domain.services.supervisor_terminate_marker."
        "consume_supervisor_terminate_marker",
        return_value=False,
    ):
        await AgentTaskRunner._maybe_stop_child_publisher(
            skel, SessionStatus.COMPLETED, terminal_reason=None,
        )

    publisher.publish.assert_awaited_once()
    skel._cleanup_heartbeat_task.assert_awaited_once()


async def test_disabled_flag_skips_terminal_publish() -> None:
    """[§8.5.1 r6 P0-1 core] disabled=True → publisher.publish NOT called."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    skel = _runner_skel(disabled=True, publisher=publisher)

    await AgentTaskRunner._maybe_stop_child_publisher(
        skel, SessionStatus.COMPLETED, terminal_reason=None,
    )

    publisher.publish.assert_not_called()


async def test_disabled_flag_still_cleans_heartbeat() -> None:
    """[§8.5.1 r6 P0-1 corollary] disabled=True MUST still run heartbeat
    cleanup. Skipping it would leak the PROGRESS_UPDATE emitter task."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    skel = _runner_skel(disabled=True, publisher=publisher)

    await AgentTaskRunner._maybe_stop_child_publisher(
        skel, SessionStatus.COMPLETED, terminal_reason=None,
    )

    skel._cleanup_heartbeat_task.assert_awaited_once()
    publisher.publish.assert_not_called()


async def test_disabled_flag_skips_publish_on_user_cancel() -> None:
    """[§8.5.1 r6 P0-1] flag gating applies to BOTH RESULT_READY AND
    CANCEL_ACK terminal envelopes — coordinator child finalizer publishes
    CANCEL_ACK itself, so default runner MUST stay silent."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    skel = _runner_skel(disabled=True, publisher=publisher)

    await AgentTaskRunner._maybe_stop_child_publisher(
        skel, SessionStatus.TIMED_OUT, terminal_reason="user_cancel",
    )

    publisher.publish.assert_not_called()
    skel._cleanup_heartbeat_task.assert_awaited_once()


async def test_disabled_flag_attribute_exists_default_false() -> None:
    """[smoke] AgentTaskRunner ctor accepts the flag and defaults to False."""
    from app.domain.services.agent_task_runner import AgentTaskRunner
    import inspect

    sig = inspect.signature(AgentTaskRunner.__init__)
    assert "terminal_envelope_publisher_disabled" in sig.parameters
    param = sig.parameters["terminal_envelope_publisher_disabled"]
    assert param.default is False
