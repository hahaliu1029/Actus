"""Unit tests for _rehydrate_dispatch (PR-7 Task 7.5)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_rehydrate_service import (
    AlreadyAppliedInfo,
    RehydrateResult,
    TerminalEnvelopeRecord,
)
from app.application.services.coordinator_parent_execution_lease import (
    CoordinatorParentExecutionLease,
    CoordinatorParentPhase,
)
from app.application.services.coordinator_wait_guard import CoordinatorWaitGuard
from app.domain.models.mailbox_envelope import ResultReadyOutcome
from app.domain.models.work_unit import WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
    WorkerResult,
    _build_pre_results_from_terminal,
    _rehydrate_dispatch,
    dispatch_node,
    reducer_node,
)

pytestmark = pytest.mark.anyio


def _make_wu(wu_id: str = "wu1") -> WorkUnit:
    # phase="exploration" because WorkUnit validator forbids phase="write"
    # without a non-empty write_lease (api/app/domain/models/work_unit.py
    # lines 70-86). Tests only care about the work_unit_id surface that
    # _rehydrate_dispatch reads.
    return WorkUnit(
        work_unit_id=wu_id,
        objective="x",
        phase="exploration",
        allowed_tools=[],
        write_lease=[],
        expected_result_schema=None,
    )


def _make_state(
    *, parent_session_id: str = "p1", root_session_id: str = "r1"
) -> dict:
    return {
        "parent_session_id": parent_session_id,
        "root_session_id": root_session_id,
        "work_unit_requests": [],
        "step_id": "s1",
        "user_id": "u1",
    }


def _make_config(
    *,
    mailbox_publisher=None,
    envelope_factory=None,
    mailbox_subscriber=None,
) -> dict:
    if mailbox_subscriber is None:
        mailbox_subscriber = AsyncMock()
        mailbox_subscriber.subscribe = AsyncMock()
    cfg: dict = {"mailbox_subscriber": mailbox_subscriber}
    if mailbox_publisher is not None:
        cfg["mailbox_publisher"] = mailbox_publisher
    if envelope_factory is not None:
        cfg["envelope_factory"] = envelope_factory
    return {"configurable": cfg}


# Pin anyio backend to asyncio to avoid trio dep / param-explosion.
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -- _build_pre_results_from_terminal --------------------------------------

# [S2 §3.2 C1] _build_pre_results_from_terminal is now async and requires an
# artifact_storage + work_units_by_id. These TestBuildPreResults cases carry no
# patch_manifest_ref, so the resolver never touches storage; an AsyncMock keeps
# an accidental call observable instead of crashing with TypeError.
_FAKE_ARTIFACT_STORAGE = AsyncMock()


class TestBuildPreResults:
    async def test_result_ready_constructs_worker_result(self) -> None:
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "success", "summary": "done"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert len(out) == 1
        assert isinstance(out[0], WorkerResult)
        assert out[0].work_unit_id == "wu1"
        assert out[0].outcome == ResultReadyOutcome.SUCCESS
        assert out[0].summary == "done"

    async def test_jsonb_patch_manifest_dict_coerced_to_model(self) -> None:
        """[codex R2 P1] JSONB round-trip lands patch_manifest as dict;
        downstream reducer does attribute access (pm.coordinator_run_id)
        which fails on dict. Builder must coerce dict -> PatchManifest.
        """
        from app.domain.models.patch_manifest import PatchManifest
        pm_dict = {
            "patch_id": "r1:wu1:p",
            "coordinator_run_id": "r1",
            "work_unit_id": "wu1",
            "files": [],
        }
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "success", "patch_manifest": pm_dict},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert isinstance(out[0].patch_manifest, PatchManifest)
        assert out[0].patch_manifest.coordinator_run_id == "r1"
        assert out[0].patch_manifest.work_unit_id == "wu1"

    async def test_invalid_patch_manifest_dict_degrades_to_none(self) -> None:
        """Defensive: a malformed manifest dict (missing required fields)
        does not raise -- WorkerResult ends up with patch_manifest=None.
        """
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "success", "patch_manifest": {"bad": "shape"}},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert out[0].patch_manifest is None
        # ... but the rest of WorkerResult still populated
        assert out[0].outcome == ResultReadyOutcome.SUCCESS

    async def test_cancel_ack_cancelled_maps_correctly(self) -> None:
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="CANCEL_ACK",
                payload={"final_state": "cancelled", "summary": "user cancel"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert out[0].outcome == ResultReadyOutcome.CANCELLED

    async def test_cancel_ack_force_terminated_maps_to_timed_out(self) -> None:
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="CANCEL_ACK",
                payload={"final_state": "force_terminated"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert out[0].outcome == ResultReadyOutcome.TIMED_OUT

    async def test_unknown_envelope_type_logged_and_skipped(self) -> None:
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="MYSTERY",
                payload={},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert out == []

    async def test_invalid_outcome_string_degrades_to_failed(self) -> None:
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "not_a_real_outcome"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal, artifact_storage=_FAKE_ARTIFACT_STORAGE, work_units_by_id={}
        )
        assert out[0].outcome == ResultReadyOutcome.FAILED

    def _shell_write_unit(self, wu_id: str = "wu1") -> WorkUnit:
        from app.domain.models.work_unit import TreeLease

        return WorkUnit(
            work_unit_id=wu_id, objective="o", phase="write", shell_mode=True,
            allowed_tools=["file_write"],
            write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
        )

    def _success_manifest_terminal(self, wu_id: str = "wu1") -> dict:
        pm_dict = {
            "patch_id": f"r1:{wu_id}:p",
            "coordinator_run_id": "r1",
            "work_unit_id": wu_id,
            "files": [],
        }
        return {
            wu_id: TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "success", "patch_manifest": pm_dict,
                         "summary": "done"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }

    async def test_flag_flip_kill_switch_demotes_shell_replay_success(self) -> None:
        # [S2 PR-4 kill-switch] flag OFF + a shell-intended wu_id in the kill set
        # ⇒ the persisted SUCCESS terminal's captured manifest is NOT replayed;
        # the WorkerResult is demoted to FAILED with patch_manifest None.
        wu = self._shell_write_unit("wu1")
        out = await _build_pre_results_from_terminal(
            self._success_manifest_terminal("wu1"),
            artifact_storage=_FAKE_ARTIFACT_STORAGE,
            work_units_by_id={"wu1": wu},
            shell_replay_kill_ids={"wu1"},
        )
        assert out[0].outcome == ResultReadyOutcome.FAILED
        assert out[0].patch_manifest is None
        assert "flag OFF" in (out[0].summary or "")

    async def test_no_kill_switch_replays_success_when_id_absent(self) -> None:
        # Empty kill set (flag ON, or a non-shell unit) ⇒ the SUCCESS terminal's
        # manifest IS replayed (resolved to a PatchManifest), no demotion. Proves
        # the kill-switch is gated strictly on membership, not always-on.
        from app.domain.models.patch_manifest import PatchManifest

        wu = self._shell_write_unit("wu1")
        out = await _build_pre_results_from_terminal(
            self._success_manifest_terminal("wu1"),
            artifact_storage=_FAKE_ARTIFACT_STORAGE,
            work_units_by_id={"wu1": wu},
            shell_replay_kill_ids=set(),
        )
        assert out[0].outcome == ResultReadyOutcome.SUCCESS
        assert isinstance(out[0].patch_manifest, PatchManifest)

    async def test_kill_switch_demotes_byref_resolved_shell_replay_success(
        self,
    ) -> None:
        # [codex PR-4 R3 P2] The kill-switch must fire even when the manifest is
        # resolved BY-REF (PR-2), not just inline: the builder RESOLVES the ref
        # (artifact_storage.get_bytes) THEN the flag-off demotion discards it.
        # Proves a large by-ref shell manifest cannot slip past the kill-switch.
        import json as _json

        from app.domain.models.patch_manifest import PatchManifest

        wu = self._shell_write_unit("wu1")
        pm = PatchManifest(
            patch_id="r1:wu1:p", coordinator_run_id="r1", work_unit_id="wu1",
            files=(),
        )
        artifact = AsyncMock()
        artifact.get_bytes = AsyncMock(
            return_value=_json.dumps(pm.model_dump(mode="json")).encode("utf-8")
        )
        terminal = {
            "wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={
                    "outcome": "success",
                    "patch_manifest_ref": "coordinator/r1/wu1/manifest/abc",
                    "summary": "done",
                },
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            ),
        }
        out = await _build_pre_results_from_terminal(
            terminal,
            artifact_storage=artifact,
            work_units_by_id={"wu1": wu},
            shell_replay_kill_ids={"wu1"},
        )
        assert out[0].outcome == ResultReadyOutcome.FAILED
        assert out[0].patch_manifest is None
        artifact.get_bytes.assert_awaited_once()  # ref resolved, THEN demoted


# -- _rehydrate_dispatch branches ------------------------------------------


class TestRehydrateDispatchBranches:
    async def test_already_applied_short_circuits_to_end(self) -> None:
        from langgraph.graph import END
        probe_quota = AsyncMock()
        config = _make_config()
        config["configurable"]["probe_quota"] = probe_quota
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=AlreadyAppliedInfo(status="success", audit_id=42),
        )
        cmd = await _rehydrate_dispatch(
            _make_state(),
            config,
            existing,
            "r1",
            [_make_wu("wu1")],
        )
        assert cmd.goto == END
        assert cmd.update["step_result_candidate"] == "ALREADY_APPLIED:success:42"
        assert cmd.update["group_outcome"] is None
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
            coordinator_run_id="r1",
        )
        probe_quota.acquire_coordinator_concurrency.assert_not_awaited()

    async def test_already_applied_release_failure_preserves_terminal_outcome(
        self,
    ) -> None:
        from langgraph.graph import END

        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock(
            side_effect=RuntimeError("redis unavailable")
        )
        config = _make_config()
        config["configurable"]["probe_quota"] = probe_quota
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=AlreadyAppliedInfo(status="success", audit_id=42),
        )

        cmd = await _rehydrate_dispatch(
            _make_state(), config, existing, "r1", [_make_wu("wu1")]
        )

        assert cmd.goto == END
        assert cmd.update["step_result_candidate"] == "ALREADY_APPLIED:success:42"
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
            coordinator_run_id="r1",
        )

    async def test_already_applied_stops_guard_before_guard_owned_release(
        self,
    ) -> None:
        probe_quota = AsyncMock()
        wait_guard = AsyncMock()
        wait_guard.resume_run = AsyncMock(return_value=True)
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": wait_guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=AlreadyAppliedInfo(status="success", audit_id=42),
        )

        await _rehydrate_dispatch(
            _make_state(), config, existing, "r1", [_make_wu("wu1")]
        )

        wait_guard.resume_run.assert_awaited_once_with("s1", "r1")
        probe_quota.release_coordinator_quotas.assert_not_awaited()

    async def test_in_progress_recent_does_not_direct_release(self) -> None:
        probe_quota = AsyncMock()
        config = _make_config()
        config["configurable"]["probe_quota"] = probe_quota
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=AlreadyAppliedInfo(
                status="in_progress_recent", audit_id=42,
            ),
        )

        cmd = await _rehydrate_dispatch(
            _make_state(), config, existing, "r1", [_make_wu("wu1")]
        )

        assert cmd.update["step_result_candidate"] == (
            "ALREADY_APPLIED:in_progress_recent:42"
        )
        probe_quota.release_coordinator_quotas.assert_not_awaited()
        probe_quota.acquire_coordinator_concurrency.assert_not_awaited()

    async def test_in_progress_recent_does_not_touch_real_guard(self) -> None:
        probe_quota = AsyncMock()

        async def release_quota() -> None:
            await probe_quota.release_coordinator_quotas(
                user_id="u1", coordinator_run_id="r1",
            )

        guard = CoordinatorWaitGuard()
        await guard.enter_run("s1", "r1", release_quota=release_quota)
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=AlreadyAppliedInfo(
                status="in_progress_recent", audit_id=42,
            ),
        )

        await _rehydrate_dispatch(
            _make_state(), config, existing, "r1", [_make_wu("wu1")]
        )

        assert "r1" in guard._runs_by_step["s1"]
        probe_quota.release_coordinator_quotas.assert_not_awaited()

        assert await guard.resume_run("s1", "r1") is True
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1", coordinator_run_id="r1",
        )

    async def test_unknown_already_applied_status_fails_closed_without_release(
        self,
    ) -> None:
        probe_quota = AsyncMock()
        wait_guard = AsyncMock()
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": wait_guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=AlreadyAppliedInfo(
                status="future_status", audit_id=42,
            ),
        )

        await _rehydrate_dispatch(
            _make_state(), config, existing, "r1", [_make_wu("wu1")]
        )

        wait_guard.resume_run.assert_not_awaited()
        probe_quota.release_coordinator_quotas.assert_not_awaited()

    async def test_terminal_only_routes_to_reducer(self) -> None:
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
            already_applied=None,
        )
        cmd = await _rehydrate_dispatch(
            _make_state(),
            _make_config(),
            existing,
            "r1",
            [_make_wu("wu1")],
        )
        assert cmd.goto == "reducer_node"
        assert len(cmd.update["worker_results"]) == 1

    async def test_terminal_only_rehydrate_starts_real_reducing_parent_lease(
        self,
    ) -> None:
        touched = AsyncMock()
        handles: list[CoordinatorParentExecutionLease] = []

        class LivenessMustNotBeRead:
            async def get_lease(self, _child_id):
                raise AssertionError("REDUCING must not depend on child liveness")

            def is_stale(self, _lease):
                raise AssertionError("REDUCING must not inspect child liveness")

        def factory(**kwargs):
            handle = CoordinatorParentExecutionLease(
                liveness_service=LivenessMustNotBeRead(),
                interval_seconds=60,
                touch_parent_activity=touched,
                **kwargs,
            )
            handles.append(handle)
            return handle

        guard = CoordinatorWaitGuard(parent_lease_factory=factory)
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
        )
        config = _make_config()
        config["configurable"]["coordinator_wait_guard"] = guard

        async with guard.step_scope("s1"):
            cmd = await _rehydrate_dispatch(
                _make_state(), config, existing, "r1", [_make_wu("wu1")],
            )
            for _ in range(20):
                if touched.await_count:
                    break
                await asyncio.sleep(0)
            assert cmd.goto == "reducer_node"
            assert len(handles) == 1
            assert handles[0].phase is CoordinatorParentPhase.REDUCING
            context = touched.await_args.args[0]
            assert context.child_session_ids == ("c1",)
            assert context.phase is CoordinatorParentPhase.REDUCING

        assert handles[0].done

    async def test_pending_only_sends_to_workers(self) -> None:
        from langgraph.types import Send
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            pending=["wu1", "wu2"],
            terminal={},
            already_applied=None,
        )
        subscriber = AsyncMock()
        subscriber.subscribe = AsyncMock()
        cmd = await _rehydrate_dispatch(
            _make_state(),
            _make_config(mailbox_subscriber=subscriber),
            existing,
            "r1",
            [_make_wu("wu1"), _make_wu("wu2")],
        )
        assert isinstance(cmd.goto, list)
        assert len(cmd.goto) == 2
        assert all(isinstance(s, Send) for s in cmd.goto)
        assert subscriber.subscribe.await_count == 2

    async def test_startup_lease_failure_retains_quota_for_pending_child(
        self,
    ) -> None:
        probe_quota = AsyncMock()
        liveness = AsyncMock()
        liveness.record_startup_lease = AsyncMock(
            side_effect=RuntimeError("lease registration failed")
        )
        guard = CoordinatorWaitGuard()
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_liveness_service": liveness,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=["wu1"],
            terminal={},
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="lease registration failed"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1", [_make_wu("wu1")],
            )

        probe_quota.release_coordinator_quotas.assert_not_awaited()
        assert guard._runs_by_step == {}

    async def test_mixed_terminal_and_pending(self) -> None:
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            pending=["wu2"],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
            already_applied=None,
        )
        cmd = await _rehydrate_dispatch(
            _make_state(),
            _make_config(),
            existing,
            "r1",
            [_make_wu("wu1"), _make_wu("wu2")],
        )
        assert len(cmd.update["worker_results"]) == 1
        # goto should be a list of 1 Send (for wu2 only)
        assert len(cmd.goto) == 1

    async def test_unexpected_child_publishes_cancel_request(self) -> None:
        # work_units = [wu1] (expected); existing has wu1 AND wu_extra. The
        # wu_extra entry must trigger CANCEL_REQUEST. (Both wu1 and wu_extra
        # are terminal so no Send fan-out is required.)
        from datetime import datetime, timezone
        term = TerminalEnvelopeRecord(
            envelope_type="RESULT_READY",
            payload={"outcome": "success"},
            child_session_id="c1",
            received_at=datetime.now(timezone.utc),
        )
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1", "wu_extra": "c_extra"},
            pending=[],
            terminal={"wu1": term},
            already_applied=None,
        )
        publisher = AsyncMock()
        publisher.publish = AsyncMock()
        envelope_factory = MagicMock()
        envelope_factory.make_cancel_request = MagicMock(return_value="ENV")
        await _rehydrate_dispatch(
            _make_state(),
            _make_config(
                mailbox_publisher=publisher,
                envelope_factory=envelope_factory,
            ),
            existing,
            "r1",
            [_make_wu("wu1")],
        )
        publisher.publish.assert_awaited_once_with("ENV")

    async def test_unexpected_child_publish_failure_swallowed(self) -> None:
        from datetime import datetime, timezone
        term = TerminalEnvelopeRecord(
            envelope_type="RESULT_READY",
            payload={"outcome": "success"},
            child_session_id="c1",
            received_at=datetime.now(timezone.utc),
        )
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1", "wu_extra": "c_extra"},
            pending=[],
            terminal={"wu1": term},
            already_applied=None,
        )
        publisher = AsyncMock()
        publisher.publish = AsyncMock(side_effect=RuntimeError("publish down"))
        envelope_factory = MagicMock()
        envelope_factory.make_cancel_request = MagicMock(return_value="ENV")
        # Must NOT raise — best-effort cancel
        cmd = await _rehydrate_dispatch(
            _make_state(),
            _make_config(
                mailbox_publisher=publisher,
                envelope_factory=envelope_factory,
            ),
            existing,
            "r1",
            [_make_wu("wu1")],
        )
        assert cmd is not None

    async def test_missing_publisher_skips_cancel_with_warning(self) -> None:
        from datetime import datetime, timezone
        term = TerminalEnvelopeRecord(
            envelope_type="RESULT_READY",
            payload={"outcome": "success"},
            child_session_id="c1",
            received_at=datetime.now(timezone.utc),
        )
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1", "wu_extra": "c_extra"},
            pending=[],
            terminal={"wu1": term},
            already_applied=None,
        )
        # config has NO publisher/envelope_factory keys
        cmd = await _rehydrate_dispatch(
            _make_state(),
            _make_config(),
            existing,
            "r1",
            [_make_wu("wu1")],
        )
        # Successful return -- no raise
        assert cmd is not None

    async def test_missing_child_raises_runtime_error(self) -> None:
        """[codex R1 P1] v1 contract: a non-empty missing_wu_ids is unrecoverable
        and must raise rather than silently fall through to reducer with an
        incomplete worker_results set. PR-7+ will replace this with idempotent
        re-spawn once the partial-unique INSERT is wired."""
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},  # wu1 exists; wu2 missing
            pending=["wu1"],
            terminal={},
            already_applied=None,
        )
        with pytest.raises(RuntimeError, match="cannot resume run"):
            await _rehydrate_dispatch(
                _make_state(),
                _make_config(),
                existing,
                "r1",
                [_make_wu("wu1"), _make_wu("wu2")],
            )

    async def test_missing_child_releases_before_guard_transfer(self) -> None:
        probe_quota = AsyncMock()
        guard = CoordinatorWaitGuard()
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="cannot resume run"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1",
                [_make_wu("wu1"), _make_wu("wu2")],
            )

        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1", coordinator_run_id="r1",
        )
        assert guard._runs_by_step == {}

    async def test_pending_missing_child_retains_shared_run_quota(self) -> None:
        probe_quota = AsyncMock()
        guard = CoordinatorWaitGuard()
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=["wu1"],
            terminal={},
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="cannot resume run"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1",
                [_make_wu("wu1"), _make_wu("wu2")],
            )

        probe_quota.release_coordinator_quotas.assert_not_awaited()

    async def test_existing_guard_without_quota_callback_is_not_handoff(
        self,
    ) -> None:
        probe_quota = AsyncMock()
        guard = CoordinatorWaitGuard()
        await guard.enter_run("s1", "r1")
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=["wu1"],
            terminal={},
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="quota ownership transfer failed"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1", [_make_wu("wu1")],
            )

        # A pending child may still be renewing this shared run-id quota from
        # another backend, so local setup failure must not remove the member.
        probe_quota.release_coordinator_quotas.assert_not_awaited()
        assert guard._runs_by_step == {}

    async def test_terminal_existing_guard_without_callback_releases_exactly(
        self,
    ) -> None:
        probe_quota = AsyncMock()
        guard = CoordinatorWaitGuard()
        await guard.enter_run("s1", "r1")
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="quota ownership transfer failed"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1", [_make_wu("wu1")],
            )

        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1", coordinator_run_id="r1",
        )
        assert guard._runs_by_step == {}

    async def test_terminal_fatal_uses_existing_guard_release_only_once(
        self,
    ) -> None:
        probe_quota = AsyncMock()

        async def release_quota() -> None:
            await probe_quota.release_coordinator_quotas(
                user_id="u1", coordinator_run_id="r1",
            )

        guard = CoordinatorWaitGuard()
        await guard.enter_run("s1", "r1", release_quota=release_quota)
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="cannot resume run"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1",
                [_make_wu("wu1"), _make_wu("wu2")],
            )

        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1", coordinator_run_id="r1",
        )
        assert guard._runs_by_step == {}

    async def test_terminal_only_release_failure_preserves_preflight_error(
        self,
    ) -> None:
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock(
            side_effect=RuntimeError("redis unavailable")
        )
        guard = CoordinatorWaitGuard()
        config = _make_config()
        config["configurable"].update({
            "probe_quota": probe_quota,
            "coordinator_wait_guard": guard,
        })
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={
                "wu1": TerminalEnvelopeRecord(
                    envelope_type="RESULT_READY",
                    payload={"outcome": "success"},
                    child_session_id="c1",
                    received_at=datetime.now(timezone.utc),
                ),
            },
            already_applied=None,
        )

        with pytest.raises(RuntimeError, match="cannot resume run"):
            await _rehydrate_dispatch(
                _make_state(), config, existing, "r1",
                [_make_wu("wu1"), _make_wu("wu2")],
            )

        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1", coordinator_run_id="r1",
        )

    async def test_limbo_child_raises_runtime_error(self) -> None:
        """[codex R4 P1] A wu_id with a child row that is NEITHER in
        existing.pending NOR existing.terminal (e.g. child status is
        ``completed`` but no envelope persisted because PR-7's best-effort
        PROLOGUE swallowed an exception) is a 'limbo' state: silently
        routing to reducer with an incomplete worker_results set produces
        a degraded INCOMPLETE outcome.
        """
        existing = RehydrateResult(
            # wu1 exists as a child row, but is NOT in pending (rehydrate
            # service filtered it out because status is not pending/running)
            # AND is NOT in terminal (envelope persistence failed earlier).
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={},
            already_applied=None,
        )
        with pytest.raises(RuntimeError, match="non-pending non-running"):
            await _rehydrate_dispatch(
                _make_state(),
                _make_config(),
                existing,
                "r1",
                [_make_wu("wu1")],
            )

    async def test_partial_limbo_with_some_terminals_still_raises(self) -> None:
        """Mixed scenario: wu1 has a real terminal envelope; wu2 is in
        limbo. The presence of wu1's clean path must NOT mask wu2's
        limbo state — operator needs to see the limbo failure."""
        from datetime import datetime, timezone
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            pending=[],
            terminal={"wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "success"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            )},  # wu2 has NO terminal envelope
            already_applied=None,
        )
        with pytest.raises(RuntimeError, match=r"work_units \['wu2'\]"):
            await _rehydrate_dispatch(
                _make_state(),
                _make_config(),
                existing,
                "r1",
                [_make_wu("wu1"), _make_wu("wu2")],
            )

    async def test_dispatch_node_threads_rehydrate_emit_closure(
        self, monkeypatch,
    ) -> None:
        """[finish-core R3] On the crash-recovery branch (peek attempt >= 1 +
        event_queue present), dispatch_node must build a per-run async emit
        closure and pass it into rehydrate_service.detect_existing_run via
        ``emit_event=``. Guards parallel_execution_subgraph.py:257-267 — if the
        kwarg/closure is dropped, the HealthEvent alert never reaches the queue.
        """
        import asyncio

        captured: dict = {}

        async def _detect(**kwargs):
            captured.update(kwargs)
            return RehydrateResult(
                child_session_ids={}, pending=[], terminal={},
                already_applied=None,
            )

        rehydrate_service = MagicMock()
        rehydrate_service.detect_existing_run = AsyncMock(side_effect=_detect)

        session_service = MagicMock()
        # peek returns >= 1 -> crash-recovery branch
        session_service.peek_coordinator_attempt = AsyncMock(return_value=1)
        # bump must NOT be needed on this branch, but stub defensively.
        session_service.bump_coordinator_attempt = AsyncMock(return_value=1)

        # Patch the post-detection router so we isolate the wiring under test
        # (we only care that detect_existing_run got the closure; routing is
        # covered by the TestRehydrateDispatchBranches suite above).
        routed: dict = {}

        async def _stub_rehydrate_dispatch(
            state, config, existing, run_id, wus, **kwargs
        ):
            routed["called"] = True
            from langgraph.graph import END
            from langgraph.types import Command
            return Command(goto=END, update={})

        monkeypatch.setattr(
            "app.domain.services.graphs.parallel_execution_subgraph."
            "_rehydrate_dispatch",
            _stub_rehydrate_dispatch,
        )

        event_queue: asyncio.Queue = asyncio.Queue()
        config = {
            "configurable": {
                "session_service": session_service,
                "rehydrate_service": rehydrate_service,
                "event_queue": event_queue,
            }
        }

        await dispatch_node(_make_state(), config)

        # detect_existing_run was reached and threaded a non-None emitter.
        assert rehydrate_service.detect_existing_run.await_count == 1
        emit = captured.get("emit_event")
        assert emit is not None
        assert asyncio.iscoroutinefunction(emit)
        # Bonus: invoking the captured emitter lands the event on the queue.
        sentinel_event = object()
        await emit(sentinel_event)
        assert event_queue.get_nowait() is sentinel_event
        # And the post-detection router was actually reached.
        assert routed.get("called") is True

    async def test_rehydrate_reducer_never_records_run_terminal(self) -> None:
        """[C2b rollout WS1b §3.4 double-count guard — LOCKING TEST] The
        rehydrate path (``_rehydrate_dispatch``) never sets
        ``dispatch_started_monotonic`` on the state. So when reducer_node runs
        on a rehydrated run (state has NO stamp), the run-level metrics recorder
        is NEVER called — preventing a double-count of the monotonic
        run_cost_usd counter on crash recovery. Pins the invariant against a
        future refactor that lets _rehydrate_dispatch set the stamp."""
        import asyncio
        from unittest.mock import MagicMock

        from app.application.services.patch_reducer_service import (
            ReducerDiagnostics,
            ReducerOutput,
        )
        from app.domain.models.patch_apply_plan import GroupOutcome

        recorder = MagicMock()
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        queue: asyncio.Queue = asyncio.Queue()
        # State shaped like a rehydrate handoff: NO dispatch_started_monotonic.
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_make_wu("wu1")],
            "worker_results": [],
            "child_session_ids": {"wu1": "c1"},
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "event_queue": queue,
            "coordinator_metrics_recorder": recorder,
        }}

        await reducer_node(state, config)

        recorder.record_run_terminal.assert_not_called()

    async def test_missing_child_raises_even_when_only_terminals(self) -> None:
        """Defense-in-depth: a wu_id in work_units with NO corresponding
        child_session_ids entry (even if all known children are terminal)
        raises. Prevents latent silent-failure regression."""
        from datetime import datetime, timezone
        existing = RehydrateResult(
            child_session_ids={"wu1": "c1"},
            pending=[],
            terminal={"wu1": TerminalEnvelopeRecord(
                envelope_type="RESULT_READY",
                payload={"outcome": "success"},
                child_session_id="c1",
                received_at=datetime.now(timezone.utc),
            )},
            already_applied=None,
        )
        with pytest.raises(RuntimeError, match="cannot resume run"):
            await _rehydrate_dispatch(
                _make_state(),
                _make_config(),
                existing,
                "r1",
                [_make_wu("wu1"), _make_wu("wu2_missing")],
            )
