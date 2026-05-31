"""F2.2 — child workspace seed-install (G1d) + seed leak-guard (INV-F1.4b, F1.5).

[finish-core §5.1.4] Before the child ReAct loop runs, the parent's base file
bytes must be installed (seeded) into the child sandbox so the child reads real
content, with digest verification. A seed-install failure raises
``_SeedInstallError`` which ``run_work_unit`` routes to ``_finalize_failed``
(publishes RESULT_READY(FAILED), never self-destroys — M1 invariant).
"""
import asyncio
import hashlib

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.application.services import coordinator_child_runner as runner_module
from app.application.services.coordinator_child_runner import (
    ChildRunResult,
    CoordinatorChildRunner,
    ResultReadyOutcome,
    _OutOfLeaseWriteError,
    _SeedInstallError,
)
from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.work_unit import PathLease, WorkUnit


@pytest.fixture
def anyio_backend():
    return "asyncio"


pytestmark = pytest.mark.anyio


def _wu(leases):
    return WorkUnit(
        work_unit_id="wu-1",
        objective="o",
        phase="write",
        allowed_tools=["file_write"],
        write_lease=leases,
    )


def _runner(child_sandbox, artifact):
    return CoordinatorChildRunner(
        cancel_event=asyncio.Event(),
        child_sandbox=child_sandbox,
        artifact_storage=artifact,
        coordinator_run_id="run-1",
    )


async def test_install_seed_writes_modify_lease_bytes_and_verifies_digest():
    seed = b"orig content"
    digest = hashlib.sha256(seed).hexdigest()
    child_sandbox = MagicMock()
    child_sandbox.atomic_write_file = AsyncMock()
    child_sandbox.compute_digest = AsyncMock(return_value=digest)
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(return_value=seed)
    r = _runner(child_sandbox, artifact)
    wu = _wu(
        [PathLease(path="a.py", op="modify", base_digest=digest, seed_content_ref="ref-1")]
    )
    await r._install_seed(wu)
    child_sandbox.atomic_write_file.assert_awaited_once_with("a.py", seed)


async def test_install_seed_raises_on_digest_mismatch():
    seed = b"orig content"
    child_sandbox = MagicMock()
    child_sandbox.atomic_write_file = AsyncMock()
    child_sandbox.compute_digest = AsyncMock(return_value="deadbeef")  # mismatch
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(return_value=seed)
    r = _runner(child_sandbox, artifact)
    wu = _wu(
        [PathLease(path="a.py", op="modify", base_digest="cafe", seed_content_ref="ref-1")]
    )
    with pytest.raises(_SeedInstallError):
        await r._install_seed(wu)


async def test_install_seed_skips_add_lease():
    child_sandbox = MagicMock()
    child_sandbox.atomic_write_file = AsyncMock()
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    r = _runner(child_sandbox, artifact)
    wu = _wu([PathLease(path="new.py", op="add")])  # add ⇒ no seed
    await r._install_seed(wu)
    child_sandbox.atomic_write_file.assert_not_called()
    artifact.get_bytes.assert_not_called()


async def test_install_seed_raises_when_seeded_lease_missing_base_digest():
    # Fail-closed guard: a seeded modify lease with no base_digest cannot be
    # verified, so it must raise instead of silently skipping the digest check
    # (PathLease permits op="modify" + base_digest=None).
    child_sandbox = MagicMock()
    child_sandbox.atomic_write_file = AsyncMock()
    child_sandbox.compute_digest = AsyncMock(return_value=None)
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(return_value=b"x")
    r = _runner(child_sandbox, artifact)
    wu = _wu(
        [PathLease(path="a.py", op="modify", base_digest=None, seed_content_ref="ref-1")]
    )
    with pytest.raises(_SeedInstallError):
        await r._install_seed(wu)


async def test_run_work_unit_reaps_listener_when_seed_install_fails():
    # [finish-core deviation] The seed-failure branch wraps the
    # _finalize_failed return in try/finally: _safe_listener_shutdown so the
    # cancel listener is reaped (not leaked) even though run_work_unit returns
    # early. Lock in BOTH effects: listener.shutdown awaited + one
    # RESULT_READY(FAILED) published.
    inner_runner = MagicMock()
    inner_runner.invoke_until_done = AsyncMock()  # satisfies the Protocol guard

    publisher = MagicMock()
    publisher.publish = AsyncMock()

    envelope_factory = MagicMock()
    sentinel_envelope = object()
    envelope_factory.make_result_ready = MagicMock(return_value=sentinel_envelope)

    # Fake listener: start()/shutdown() awaitable, ready_event already set so
    # run_work_unit's `await listener.ready_event.wait()` returns immediately.
    fake_listener = MagicMock()
    fake_listener.start = AsyncMock()
    fake_listener.shutdown = AsyncMock()
    ready = asyncio.Event()
    ready.set()
    fake_listener.ready_event = ready

    cancel_event = asyncio.Event()  # NOT set → does not short-circuit pre-seed
    r = CoordinatorChildRunner(
        cancel_event=cancel_event,
        inner_runner=inner_runner,
        publisher=publisher,
        envelope_factory=envelope_factory,
        mailbox_subscriber=MagicMock(),
        coordinator_run_id="run-1",
    )
    # Force the seed-install step to fail.
    r._install_seed = AsyncMock(side_effect=_SeedInstallError("boom"))

    wu = _wu([PathLease(path="a.py", op="modify", base_digest="cafe", seed_content_ref="ref-1")])

    with patch.object(
        runner_module, "CoordinatorChildCancelListener", return_value=fake_listener
    ):
        result = await r.run_work_unit(
            coordinator_run_id="run-1",
            work_unit=wu,
            child_session_id="child-1",
            spawn_manifest=MagicMock(),
            cancel_event=cancel_event,
            root_session_id="root-1",
        )

    # Listener was reaped on the seed-failure exit path.
    fake_listener.shutdown.assert_awaited_once()
    # _finalize_failed ran → exactly one RESULT_READY(FAILED) published.
    publisher.publish.assert_awaited_once_with(sentinel_envelope)
    assert envelope_factory.make_result_ready.call_count == 1
    published_payload = envelope_factory.make_result_ready.call_args.kwargs["payload"]
    assert published_payload.outcome is ResultReadyOutcome.FAILED
    # The inner ReAct loop must NOT have run — seed failure short-circuits.
    inner_runner.invoke_until_done.assert_not_awaited()
    assert result is published_payload


# ---------------------------------------------------------------------------
# [F2 P0 + P1] run_work_unit-level terminal-envelope + listener-reap guards.
# ---------------------------------------------------------------------------


def _fake_listener():
    """start()/shutdown() awaitable; ready_event pre-set so the
    ``await listener.ready_event.wait()`` gate returns immediately."""
    listener = MagicMock()
    listener.start = AsyncMock()
    listener.shutdown = AsyncMock()
    ready = asyncio.Event()
    ready.set()
    listener.ready_event = ready
    return listener


def _wire_runner(inner_runner, *, child_sandbox=None, artifact=None):
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    envelope_factory = MagicMock()
    sentinel_envelope = object()
    envelope_factory.make_result_ready = MagicMock(return_value=sentinel_envelope)
    cancel_event = asyncio.Event()
    runner = CoordinatorChildRunner(
        cancel_event=cancel_event,
        inner_runner=inner_runner,
        publisher=publisher,
        child_sandbox=child_sandbox,
        artifact_storage=artifact,
        envelope_factory=envelope_factory,
        mailbox_subscriber=MagicMock(),
        coordinator_run_id="run-1",
    )
    return runner, cancel_event, publisher, envelope_factory, sentinel_envelope


def _write_tool(path):
    return ToolEvent(
        tool_call_id="tc", tool_name="file", function_name="file_write",
        function_args={"filepath": path, "content": "x"},
        status=ToolEventStatus.CALLING,
    )


async def test_run_work_unit_out_of_lease_write_publishes_needs_authorization():
    # [F2 P0a] A write-phase child whose extraction raises _OutOfLeaseWriteError
    # (tool wrote a path with no matching lease) must NOT crash run_work_unit:
    # it publishes exactly one RESULT_READY(NEEDS_AUTHORIZATION, out_of_path_lease)
    # and reaps the listener.
    # child_sandbox must support seed-install (async) so the run reaches the
    # extraction step where the out-of-lease write is caught. Seed digest is
    # made to match the lease base_digest so _install_seed succeeds.
    base = "b" * 64
    child_sandbox = MagicMock()
    child_sandbox.atomic_write_file = AsyncMock()
    child_sandbox.compute_digest = AsyncMock(return_value=base)
    child_sandbox.read_file = AsyncMock(return_value=b"x")
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(return_value=b"seed")
    artifact.put_content_addressed_bytes = AsyncMock(return_value="r")
    inner_runner = MagicMock()
    # Child wrote evil.py, but the only lease is for a.py → out-of-lease.
    inner_runner.invoke_until_done = AsyncMock(
        return_value=ChildRunResult(done_event=MagicMock(), tool_calls=(_write_tool("evil.py"),))
    )
    runner, cancel_event, publisher, envf, sentinel = _wire_runner(
        inner_runner, child_sandbox=child_sandbox, artifact=artifact,
    )
    wu = _wu([PathLease(path="a.py", op="modify", base_digest=base, seed_content_ref="ref-1")])
    fake_listener = _fake_listener()
    with patch.object(
        runner_module, "CoordinatorChildCancelListener", return_value=fake_listener
    ):
        result = await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=wu, child_session_id="child-1",
            spawn_manifest=MagicMock(), cancel_event=cancel_event, root_session_id="root-1",
        )
    # Did NOT raise; exactly one terminal envelope published; listener reaped.
    publisher.publish.assert_awaited_once_with(sentinel)
    assert envf.make_result_ready.call_count == 1
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome is ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details is not None
    assert payload.needs_authorization_details.reason == "out_of_path_lease"
    fake_listener.shutdown.assert_awaited_once()
    assert result is payload


async def test_run_work_unit_patch_build_error_publishes_failed():
    # [F2 P0b] When patch build raises (here: extraction raises a
    # pydantic.ValidationError, modeling a FilePatchEntry schema violation such
    # as the deferred absolute-path contract), run_work_unit must publish
    # exactly one RESULT_READY(FAILED), NOT raise, and reap the listener.
    import pydantic

    inner_runner = MagicMock()
    inner_runner.invoke_until_done = AsyncMock(
        return_value=ChildRunResult(done_event=MagicMock(), tool_calls=())
    )
    runner, cancel_event, publisher, envf, sentinel = _wire_runner(inner_runner)
    # Force the manifest/extraction build to raise a ValidationError.
    validation_error = pydantic.ValidationError.from_exception_data("FilePatchEntry", line_errors=[])
    runner._extract_patch_files_from_history = AsyncMock(side_effect=validation_error)
    wu = _wu([PathLease(path="a.py", op="modify", base_digest="b" * 64, seed_content_ref="ref-1")])
    fake_listener = _fake_listener()
    with patch.object(
        runner_module, "CoordinatorChildCancelListener", return_value=fake_listener
    ):
        result = await runner.run_work_unit(
            coordinator_run_id="run-1", work_unit=wu, child_session_id="child-1",
            spawn_manifest=MagicMock(), cancel_event=cancel_event, root_session_id="root-1",
        )
    publisher.publish.assert_awaited_once_with(sentinel)
    assert envf.make_result_ready.call_count == 1
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome is ResultReadyOutcome.FAILED
    fake_listener.shutdown.assert_awaited_once()
    assert result is payload


async def test_run_work_unit_reaps_listener_when_seed_install_cancelled():
    # [F2 P1] An asyncio.CancelledError raised DURING _install_seed (parent
    # cancels mid-seed) is NOT a _SeedInstallError — it must propagate, but the
    # cancel listener must still be reaped exactly once by the outer finally.
    inner_runner = MagicMock()
    inner_runner.invoke_until_done = AsyncMock()  # satisfies the Protocol guard
    runner, cancel_event, publisher, envf, _ = _wire_runner(inner_runner)
    runner._install_seed = AsyncMock(side_effect=asyncio.CancelledError())
    wu = _wu([PathLease(path="a.py", op="modify", base_digest="cafe" * 16, seed_content_ref="ref-1")])
    fake_listener = _fake_listener()
    with patch.object(
        runner_module, "CoordinatorChildCancelListener", return_value=fake_listener
    ):
        with pytest.raises(asyncio.CancelledError):
            await runner.run_work_unit(
                coordinator_run_id="run-1", work_unit=wu, child_session_id="child-1",
                spawn_manifest=MagicMock(), cancel_event=cancel_event, root_session_id="root-1",
            )
    # Listener reaped exactly once despite the cancel propagating.
    fake_listener.shutdown.assert_awaited_once()
    # No terminal envelope on a cancel-during-seed (it propagates, not finalizes).
    inner_runner.invoke_until_done.assert_not_awaited()
