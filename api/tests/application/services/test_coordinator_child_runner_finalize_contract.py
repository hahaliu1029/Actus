"""[S2 §5 invariant 5] Behavior-contract lock for the coordinator child's
natural-done finalizer dispatch (Task 2.4a, runs AFTER Task 2.8).

This pins the premise the worker_node SUCCESS-demotion logic relies on:

    coordinator SUCCESS terminal  ⇒  a non-None PatchManifest was built and
                                      routed through the BUILD-ONLY helper

and its dual:

    an exploration child finalizes NEEDS_AUTHORIZATION and NEVER reaches the
    SUCCESS build helper.

The old Task 2.4 / Task 2.5 drafts asserted "exploration SUCCESS + no manifest
stays SUCCESS" — a state the producer **cannot emit** (an exploration child
finalizes NEEDS_AUTHORIZATION, never SUCCESS — coordinator_child_runner.py:628).
This test pins the real invariant directly so the demotion premise is grounded.

Post-Task-2.8 path: ``_finalize_success`` builds its SUCCESS payload through
``_build_manifest_payload_inline_or_ref`` (BUILD-only), so this test spies that
helper and asserts the built manifest flows through it — NOT the pre-2.8 inline
``patch_manifest=patch_manifest`` payload literal (which no longer lives in
``_finalize_success``). Because this task runs AFTER 2.8, all assertions are
GREEN.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.mailbox_envelope import (
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.patch_manifest import PatchManifest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _runner() -> "CoordinatorChildRunner":
    # Construct without __init__ (mirror the producer test's _runner pattern):
    # only the collaborators the finalizer touches are stubbed. The write path
    # touches _extract_patch_files_from_history + _build_manifest_payload_inline_or_ref
    # + _publish_result_ready; the exploration path touches
    # _extract_proposed_write_plan + _publish_result_ready.
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )

    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    return runner


def _wu(wu_id: str) -> object:
    wu = MagicMock()
    wu.work_unit_id = wu_id
    return wu


@pytest.mark.anyio
async def test_finalize_success_builds_non_none_manifest_and_routes_through_helper() -> None:
    """[S2 §5 invariant 5 / F20] _finalize_success (write-phase natural done)
    constructs a non-None PatchManifest even when the child wrote nothing
    (files=() ⇒ file_count=0, apply skipped — NOT None) and routes it through
    _build_manifest_payload_inline_or_ref (post-Task-2.8 path) before the single
    publish. This asserts the BEHAVIOR (manifest built + routed via the helper),
    NOT the pre-2.8 inline ``patch_manifest=patch_manifest`` payload shape, so it
    stays green at the PR-2 boundary after Task 2.8 lands."""
    runner = _runner()
    # F20 success-with-no-writes: extraction yields zero files.
    runner._extract_patch_files_from_history = AsyncMock(return_value=[])
    # Spy the post-2.8 BUILD-ONLY helper; return a placeholder SUCCESS payload.
    # _finalize_success then calls the (separately stubbed) _publish_result_ready
    # OUTSIDE its try with that payload.
    helper = AsyncMock(
        return_value=ResultReadyPayload(
            summary="completed wu1", outcome=ResultReadyOutcome.SUCCESS,
        )
    )
    runner._build_manifest_payload_inline_or_ref = helper
    runner._publish_result_ready = AsyncMock()

    out = await runner._finalize_success("r1", _wu("wu1"), "c1", done_event=None)

    assert out.outcome == ResultReadyOutcome.SUCCESS
    helper.assert_awaited_once()
    # Build-only helper signature is (run_id, wu, patch_manifest, *, summary), so
    # the built manifest is positional arg #3 (index 2): run_id, wu, manifest.
    call = helper.await_args
    built_manifest = call.args[2]
    assert isinstance(built_manifest, PatchManifest)
    assert built_manifest is not None  # non-None even for the empty write set
    assert built_manifest.work_unit_id == "wu1"
    assert len(built_manifest.files) == 0  # F20: empty manifest, apply skipped
    # The single publish happens OUTSIDE the try with the built payload (F2 P0).
    runner._publish_result_ready.assert_awaited_once_with("c1", out)


@pytest.mark.anyio
async def test_finalize_exploration_never_routes_through_success_helper() -> None:
    """[S2 §5 invariant 5] An exploration child's natural done finalizes via
    _finalize_exploration_proposal (NEEDS_AUTHORIZATION) and NEVER reaches the
    SUCCESS build helper — so a SUCCESS worker is never an exploration worker,
    and the only `manifest_required=False` dispatch can never be SUCCESS."""
    runner = _runner()
    runner._extract_proposed_write_plan = lambda done_event: None
    runner._publish_result_ready = AsyncMock()
    # If exploration ever wrongly routed through the success build helper this spy
    # would record a call — assert it stays untouched.
    runner._build_manifest_payload_inline_or_ref = AsyncMock()

    out = await runner._finalize_exploration_proposal(
        "r1", _wu("wu1"), "c1", done_event=None,
    )

    assert out.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    runner._build_manifest_payload_inline_or_ref.assert_not_awaited()
    runner._publish_result_ready.assert_awaited_once()


def _routing_runner(monkeypatch) -> "CoordinatorChildRunner":
    """[codex PR-2 R4 P1] Build a run_work_unit-drivable runner with every
    collaborator BEFORE the phase-routing branch stubbed, so the two tests below
    assert the REAL await (exploration → _finalize_exploration_proposal, write →
    _finalize_success) instead of a source-string presence check (which would
    pass even with INVERTED routing). ``_budget=None`` skips the wallclock
    watchdog; the cancel listener is monkeypatched to a no-op whose
    ``ready_event`` is pre-set so ``ready_event.wait()`` returns immediately.
    """
    import asyncio

    import app.application.services.coordinator_child_runner as ccr_mod
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )

    class _FakeListener:
        def __init__(self, **_kwargs) -> None:
            self.ready_event = asyncio.Event()

        async def start(self) -> None:
            self.ready_event.set()

    monkeypatch.setattr(ccr_mod, "CoordinatorChildCancelListener", _FakeListener)

    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    runner._stop_reason = None            # skip the pre-run + post-invoke finalize
    runner._budget = None                 # skip the wallclock watchdog
    runner._mailbox_subscriber = MagicMock()
    runner._inner_runner = MagicMock()
    runner._inner_runner.invoke_until_done = AsyncMock(return_value="done_event")
    runner._install_seed = AsyncMock()
    runner._build_child_prompt = MagicMock(return_value="prompt")
    runner._safe_listener_shutdown = AsyncMock()
    runner._finalize_success = AsyncMock(return_value="SUCCESS_RESULT")
    runner._finalize_exploration_proposal = AsyncMock(return_value="EXPLORE_RESULT")
    return runner


@pytest.mark.anyio
async def test_run_work_unit_routes_write_phase_to_finalize_success(
    monkeypatch,
) -> None:
    """[codex PR-2 R4 P1 / S2 §5 invariant 5] A write-phase natural-done run
    routes to ``_finalize_success`` (NOT exploration). Behavioral — catches an
    inverted routing the old source-string lock could not."""
    import asyncio

    runner = _routing_runner(monkeypatch)
    write_wu = MagicMock()
    write_wu.phase = "write"
    write_wu.work_unit_id = "wu1"

    out = await runner.run_work_unit(
        coordinator_run_id="r1",
        work_unit=write_wu,
        child_session_id="c1",
        spawn_manifest=MagicMock(),
        cancel_event=asyncio.Event(),
        root_session_id="root1",
    )

    assert out == "SUCCESS_RESULT"
    runner._finalize_success.assert_awaited_once()
    runner._finalize_exploration_proposal.assert_not_awaited()


@pytest.mark.anyio
async def test_run_work_unit_routes_exploration_phase_to_exploration_finalizer(
    monkeypatch,
) -> None:
    """[codex PR-2 R4 P1 / S2 §5 invariant 5] An exploration-phase natural-done
    run routes to ``_finalize_exploration_proposal`` (NOT _finalize_success), so
    a SUCCESS terminal can never originate from an exploration unit. Behavioral
    counterpart to the write-phase test — together they pin the routing both
    ways."""
    import asyncio

    runner = _routing_runner(monkeypatch)
    explore_wu = MagicMock()
    explore_wu.phase = "exploration"
    explore_wu.work_unit_id = "wu1"

    out = await runner.run_work_unit(
        coordinator_run_id="r1",
        work_unit=explore_wu,
        child_session_id="c1",
        spawn_manifest=MagicMock(),
        cancel_event=asyncio.Event(),
        root_session_id="root1",
    )

    assert out == "EXPLORE_RESULT"
    runner._finalize_exploration_proposal.assert_awaited_once()
    runner._finalize_success.assert_not_awaited()
