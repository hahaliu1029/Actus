import pytest

from app.domain.models.mailbox_envelope import ResultReadyOutcome
from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_worker_result_manifest_required_defaults_false() -> None:
    wr = WorkerResult(
        work_unit_id="wu1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
    )
    assert wr.manifest_required is False


def test_worker_result_manifest_required_settable() -> None:
    wr = WorkerResult(
        work_unit_id="wu1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        manifest_required=True,
    )
    assert wr.manifest_required is True


# ── Task 2.4: worker_node manifest-by-ref resolution + fail-closed demotion ──
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.runnables import RunnableConfig

from app.domain.models.mailbox_envelope import (
    MailboxEnvelopeType,
    ResultReadyPayload,
)
from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest
from app.domain.services.graphs.parallel_execution_subgraph import worker_node

_SHA = hashlib.sha256(b"a").hexdigest()


def _manifest() -> PatchManifest:
    return PatchManifest(
        patch_id="r1:wu1:p",
        coordinator_run_id="r1",
        work_unit_id="wu1",
        files=(
            FilePatchEntry(
                path="d/a.py",
                op="add",
                new_digest=_SHA,
                content_ref="ref-1",
                content_size=1,
            ),
        ),
    )


def _envelope(payload: ResultReadyPayload) -> MagicMock:
    env = MagicMock()
    env.type = MailboxEnvelopeType.RESULT_READY
    env.payload = payload
    return env


def _config(*, artifact_storage) -> RunnableConfig:
    waiter = AsyncMock()
    return {  # type: ignore[return-value]
        "configurable": {
            "terminal_waiter": waiter,
            "cancel_event": __import__("asyncio").Event(),
            "artifact_storage": artifact_storage,
        }
    }


def _send(*, manifest_required: bool) -> dict:
    return {
        "work_unit_id": "wu1",
        "child_session_id": "c1",
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
        "manifest_required": manifest_required,
    }


async def test_worker_node_resolves_manifest_by_ref() -> None:
    pm = _manifest()
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(
        return_value=json.dumps(pm.model_dump(mode="json")).encode("utf-8")
    )
    payload = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest_ref="coordinator/r1/wu1/manifest/abc",
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.SUCCESS
    assert wr.patch_manifest is not None
    assert wr.patch_manifest.work_unit_id == "wu1"
    artifact.get_bytes.assert_awaited_once_with("coordinator/r1/wu1/manifest/abc")


async def test_worker_node_inline_manifest_unchanged() -> None:
    pm = _manifest()
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    payload = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest=pm,
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.SUCCESS
    assert wr.patch_manifest is not None
    artifact.get_bytes.assert_not_awaited()


async def test_worker_node_demotes_success_no_manifest_when_required() -> None:
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    payload = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.FAILED
    assert wr.patch_manifest is None


async def test_worker_node_demotes_unresolvable_ref_when_required() -> None:
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(side_effect=KeyError("missing ref"))
    payload = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest_ref="coordinator/r1/wu1/manifest/gone",
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.FAILED
    assert wr.patch_manifest is None


async def test_worker_node_manifest_NOT_required_no_demotion_path() -> None:
    # [S2 §3.2 C1 / §5 invariant 5] When manifest_required=False the demotion
    # guard is INERT — worker_node leaves the outcome alone. This does NOT
    # assert that a "SUCCESS + no manifest" state is producible in production:
    # a coordinator SUCCESS is always either a write-phase child carrying a
    # (possibly empty) manifest, or it never reaches SUCCESS at all (exploration
    # finalizes NEEDS_AUTHORIZATION — see Task 2.4a below). This case only locks
    # that the guard fires SOLELY on manifest_required=True, so a future caller
    # that (wrongly) routed a non-write worker through here would not get an
    # unexpected demotion — the demotion is gated exclusively on the flag.
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    payload = ResultReadyPayload(
        summary="explored",
        outcome=ResultReadyOutcome.SUCCESS,
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    out = await worker_node(_send(manifest_required=False), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.SUCCESS
    assert wr.patch_manifest is None


# ── Task 2.7: cross-component regression — demoted SUCCESS must NOT zero-apply ─
from app.application.services.patch_reducer_service import PatchReducerService
from app.domain.models.patch_apply_plan import GroupOutcome


async def test_dropped_manifest_success_does_not_zero_apply() -> None:
    # [S2 §5 invariant 5 / §8 regression] End-to-end lock across two
    # components: a write-phase child that returned SUCCESS but whose
    # patch_manifest did not survive (envelope-store truncation marker —
    # persisted payload came back as a bare {"outcome": "success"} with no
    # manifest, no ref) must NOT silently zero-apply.
    #
    # Chain under test:
    #   worker_node(manifest_required=True, SUCCESS, no manifest)
    #     → fail-closed demotion to FAILED (Tasks 2.4 + 2.6)
    #     → PatchReducerService.reduce sees a FAILED worker
    #     → Step 2 worker-priority short-circuit
    #     → group_outcome=FAILED, apply_plan=None  (no empty SUCCESS plan)
    #
    # Had this run against pre-2.4 code the worker would have stayed
    # SUCCESS-with-None-manifest; the reducer's SUCCESS-no-manifest branch
    # would have produced a SUCCESS group outcome with an empty plan — the
    # silent zero-apply this PR exists to prevent.
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    payload = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    worker_out = await worker_node(_send(manifest_required=True), config)
    wr = worker_out["worker_results"][0]
    # worker_node demoted it — the reducer must therefore NOT see SUCCESS.
    assert wr.outcome == ResultReadyOutcome.FAILED
    assert wr.patch_manifest is None

    reducer = PatchReducerService()
    # ``parent_sandbox=None`` is safe here: a FAILED worker short-circuits
    # at Step 2 (worker-priority), which returns BEFORE the optional
    # Step 5 digest-drift probe that is the sole parent_sandbox consumer.
    out = await reducer.reduce(
        coordinator_run_id="r1",
        work_unit_ids_expected=frozenset({"wu1"}),
        worker_results=[wr],
        parent_sandbox=None,
    )
    # A FAILED worker → non-SUCCESS group outcome → NO apply plan (the
    # zero-apply guard: never hand the applier an empty SUCCESS plan).
    assert out.group_outcome != GroupOutcome.SUCCESS
    assert out.group_outcome == GroupOutcome.FAILED
    assert out.apply_plan is None


async def test_worker_node_malformed_terminal_envelope_fails_closed() -> None:
    """[codex PR-2 R4 P0] A malformed terminal envelope must fail-closed, never
    crash worker_node.

    The waiter's predicate matches the terminal on envelope-level fields
    (type/child/correlation) over a RAW dict, and the subscriber XACKs the
    matched entry BEFORE the waiter deep-validates it via
    ``MailboxEnvelope.model_validate`` (which runs the typed ResultReadyPayload
    schema). A malformed child payload therefore surfaces as a pydantic
    ``ValidationError`` out of ``await_terminal`` AFTER the entry is already
    acked/lost. worker_node must catch it and return a FAILED WorkerResult so
    the reducer's completeness invariant holds (every Send yields a
    worker_result) and one malformed child envelope cannot crash the whole
    parallel superstep or strand the run.
    """
    from pydantic import ValidationError

    # Build the real ValidationError the waiter would surface on a bad payload.
    try:
        ResultReadyPayload.model_validate({"outcome": "not_a_real_outcome"})
        raise AssertionError("expected ValidationError")  # pragma: no cover
    except ValidationError as exc:
        err = exc

    config = _config(artifact_storage=MagicMock())
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        side_effect=err
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.FAILED
    assert wr.work_unit_id == "wu1"
    assert wr.child_session_id == "c1"
    assert wr.manifest_required is True  # preserved for reducer/diagnostics
    assert wr.patch_manifest is None


async def test_worker_node_terminal_wait_timeout_fails_closed() -> None:
    """[codex PR-2 R5 P0] ``await_terminal`` raising ``asyncio.TimeoutError``
    (the child never emitted a terminal within the 600s waiter window) must
    fail-closed to a TIMED_OUT WorkerResult, never propagate.

    A raise here crashes the whole LangGraph ``Send`` fan-out superstep — only
    ``CoordinatorPathContractError`` is caught upstream in
    ``_run_parallel_backend`` and ``executor_node`` has no retry policy — so it
    would lose EVERY sibling worker's result and strand the run. S2 shell-mode
    children do heavier work and are likelier to hit this. worker_node must
    always produce a WorkerResult so the reducer's completeness invariant holds
    and a single slow child cannot kill the batch.
    """
    import asyncio

    config = _config(artifact_storage=MagicMock())
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        side_effect=asyncio.TimeoutError("no terminal after 600s")
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.TIMED_OUT
    assert wr.work_unit_id == "wu1"
    assert wr.child_session_id == "c1"
    assert wr.manifest_required is True
    assert wr.patch_manifest is None


# ── codex PR-4 R1 P0: flag-flip kill-switch on the PENDING/worker_node path ──


async def test_worker_node_shell_replay_kill_demotes_success_with_manifest() -> None:
    # [codex PR-4 R1 P0] A pending shell-intended child whose SUCCESS terminal
    # surfaces in worker_node under flag OFF (shell_replay_kill=True) must be
    # demoted to FAILED with its captured manifest discarded — twin of the
    # already-terminal kill-switch. Even a fully-resolvable manifest must NOT
    # reach the reducer/apply.
    pm = _manifest()
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    payload = ResultReadyPayload(
        summary="captured under flag ON",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest=pm,
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    send = _send(manifest_required=True)
    send["shell_replay_kill"] = True
    out = await worker_node(send, config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.FAILED
    assert wr.patch_manifest is None
    assert "flag OFF" in (wr.summary or "")


async def test_worker_node_no_kill_when_shell_replay_kill_false() -> None:
    # Differential: same SUCCESS+manifest, but shell_replay_kill absent/False
    # (flag ON, or a non-shell unit) ⇒ the manifest IS kept (no demotion). Proves
    # the kill is gated strictly on the bit, not always-on.
    pm = _manifest()
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock()
    payload = ResultReadyPayload(
        summary="ok",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest=pm,
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    out = await worker_node(_send(manifest_required=True), config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.SUCCESS
    assert wr.patch_manifest is not None


async def test_worker_node_shell_replay_kill_demotes_byref_success() -> None:
    # [codex PR-4 R3 P2] pending-path kill-switch must fire on a BY-REF resolved
    # manifest too (resolve-then-demote), not just inline. The ref IS resolved
    # (get_bytes awaited) THEN the flag-off demotion discards it → FAILED/None.
    pm = _manifest()
    artifact = MagicMock()
    artifact.get_bytes = AsyncMock(
        return_value=json.dumps(pm.model_dump(mode="json")).encode("utf-8")
    )
    payload = ResultReadyPayload(
        summary="captured under flag ON",
        outcome=ResultReadyOutcome.SUCCESS,
        patch_manifest_ref="coordinator/r1/wu1/manifest/abc",
    )
    config = _config(artifact_storage=artifact)
    config["configurable"]["terminal_waiter"].await_terminal = AsyncMock(
        return_value=_envelope(payload)
    )
    send = _send(manifest_required=True)
    send["shell_replay_kill"] = True
    out = await worker_node(send, config)
    wr = out["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.FAILED
    assert wr.patch_manifest is None
    artifact.get_bytes.assert_awaited_once()  # ref resolved, THEN demoted
