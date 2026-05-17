"""Phase 10, Task 10.2: interrupt_helper drives pe.commit_resume + writes pe_resume_outcomes.

Uses the same approach as test_react_graph_pe_dispatch.py: build a minimal
react_graph and extract the interrupt_helper closure from the compiled graph.
Tests are adapted from the plan (no graph_runner_with_fake_pe_in_interrupt_helper
fixture — we call the closure directly with controlled state/config).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.domain.models.tool_result import AllowSuccess, Denied, DecisionReason

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_interrupt_helper_fn():
    """Build a minimal react_graph and extract the interrupt_helper closure."""
    from langchain_core.tools import tool as lc_tool
    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        return f"wrote {path}"

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [file_write])
    # interrupt_helper is registered as a graph node; extract the async closure.
    return graph.nodes["interrupt_helper"].bound.afunc


def _make_state_with_pending(tool_call_id: str = "tc1", tool_name: str = "file_write") -> dict:
    """Build a minimal state dict with pending_ask_* fields set."""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": tool_call_id, "name": tool_name, "args": {"path": "/x"}, "type": "tool_call"}],
            )
        ],
        "llm_input_messages": [],
        "step_description": "test",
        "original_request": "test",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "events": [],
        "should_interrupt": False,
        "soft_hint_sent": False,
        "attempt_count": 0,
        "failure_count": 0,
        "completed_tool_call_prefix": [],
        "approved_tool_call_ids": [],
        "pe_resume_outcomes": {},
        "pending_ask_outcome": {"variant": "asked", "content": "waiting"},
        "pending_ask_tool_call_id": tool_call_id,
        "pending_ask_artifact": {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_source": {"source": "native", "category": "file", "canonical_name": tool_name},
            "outcome": {"variant": "asked", "content": "waiting"},
        },
        "pending_ask_tool_args": {"path": "/x"},
    }


def _make_fake_ssm(mode=None, revision: int = 1):
    """Return an AsyncMock SSM that returns (mode, revision)."""
    from app.domain.models.session import SessionStatus
    mode = mode or SessionStatus.RUNNING
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(mode, revision))
    return ssm


def _make_config(fake_pe, fake_ssm, *, user_id="u", session_id="s"):
    """Build a minimal RunnableConfig with PE + SSM."""
    return {
        "configurable": {
            "permission_engine": fake_pe,
            "session_state_machine": fake_ssm,
            "permission_engine_native_enabled": True,
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }


# ---------------------------------------------------------------------------
# Task 10.2: PE path — commit_resume called, pe_resume_outcomes updated
# ---------------------------------------------------------------------------

class TestInterruptHelperPeCommit:
    async def test_pe_path_calls_commit_resume_and_updates_state(self):
        """interrupt_helper calls pe.commit_resume and writes pe_resume_outcomes."""
        import asyncio
        from langgraph.types import interrupt as _interrupt

        interrupt_helper_fn = _build_interrupt_helper_fn()
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            return_value=AllowSuccess(content="ok", data={"via": "user_click"}),
        )
        fake_ssm = _make_fake_ssm()
        state = _make_state_with_pending("tc1")
        config = _make_config(fake_pe, fake_ssm)

        # The resume_payload that interrupt() returns on second invocation.
        resume_payload = {
            "tool_call_id": "tc1",
            "action": "approve",
            "scope": "session",
            "claim_nonce": "n" * 32,
        }

        # interrupt() raises GraphInterrupt on first call; we mock it to return
        # the resume payload directly so we can exercise the resume branch.
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # pe.commit_resume must have been called
        fake_pe.commit_resume.assert_awaited_once()
        call_args = fake_pe.commit_resume.call_args
        assert call_args is not None

        # pe_resume_outcomes must contain the outcome for tc1
        assert "pe_resume_outcomes" in cmd.update
        assert "tc1" in cmd.update["pe_resume_outcomes"]
        outcome_dict = cmd.update["pe_resume_outcomes"]["tc1"]
        assert outcome_dict.get("variant") == "allow_success" or "content" in outcome_dict

        # Routed to tool_node
        assert cmd.goto == "tool_node"

        # pending_ask fields cleared
        assert cmd.update.get("pending_ask_outcome") is None
        assert cmd.update.get("pending_ask_tool_call_id") is None
        assert cmd.update.get("pending_ask_artifact") is None
        assert cmd.update.get("pending_ask_tool_args") is None

    async def test_legacy_path_used_when_pe_absent(self):
        """interrupt_helper falls back to legacy path (approved_tool_call_ids) when PE is None."""
        interrupt_helper_fn = _build_interrupt_helper_fn()
        state = _make_state_with_pending("tc2")

        # No PE in config
        config = {
            "configurable": {
                "user_id": "u",
                "session_id": "s",
                "thread_id": "s",
            }
        }
        resume_payload = {
            "action": "approve",
            "tool_call_id": "tc2",
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Legacy path: approved_tool_call_ids must contain tc2
        assert "tc2" in cmd.update.get("approved_tool_call_ids", [])
        # pe_resume_outcomes NOT written by legacy path
        assert "tc2" not in (cmd.update.get("pe_resume_outcomes") or {})
        assert cmd.goto == "tool_node"

    async def test_pe_path_falls_back_on_missing_claim_nonce(self):
        """interrupt_helper falls back to legacy path when claim_nonce is absent."""
        interrupt_helper_fn = _build_interrupt_helper_fn()
        fake_pe = AsyncMock()
        fake_ssm = _make_fake_ssm()
        state = _make_state_with_pending("tc3")
        config = _make_config(fake_pe, fake_ssm)

        # No claim_nonce in resume payload
        resume_payload = {
            "action": "approve",
            "scope": "session",
            "tool_call_id": "tc3",
            # claim_nonce intentionally absent
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Falls back to legacy: approved_tool_call_ids written, NOT pe_resume_outcomes
        assert "tc3" in cmd.update.get("approved_tool_call_ids", [])
        # commit_resume must NOT have been called
        fake_pe.commit_resume.assert_not_called()

    async def test_p1_2_arg_digest_restored_from_queue(self):
        """P1#2: interrupt_helper reads arg_digest/primary_arg/dir_arg from queue.

        When _rehydrate_call_spec builds a ToolCallSpec without these digest fields
        (because ToolArtifact schema doesn't carry them), interrupt_helper must read
        them from pe._queue to ensure commit_resume receives the canonical arg_digest
        and doesn't raise arg_digest_mismatch.
        """
        from langgraph.types import Command
        from app.domain.models.tool_result import AllowSuccess
        from app.domain.services.permission.tool_call_spec import ToolCallSpec
        from app.domain.services.permission.confirmation_queue import ConfirmationDetail
        import time

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # Set up the queue mock: read() returns a ConfirmationDetail with known arg_digest
        expected_arg_digest = "sha256-test-digest-abc123"
        expected_primary_arg = "/workspace/test.txt"
        expected_dir_arg = "/workspace"

        mock_detail = ConfirmationDetail(
            session_id="s",
            tool_call_id="tc-digest",
            user_id="u",
            tool_name="file_write",
            tool_args={"path": "/workspace/test.txt"},
            risk_level="medium",
            arg_digest=expected_arg_digest,
            primary_arg=expected_primary_arg,
            dir_arg=expected_dir_arg,
            matched_patterns=[],
            deadline_ts=time.time() + 300,
        )

        fake_queue = AsyncMock()
        fake_queue.read = AsyncMock(return_value=mock_detail)

        # PE with a queue attribute (mirroring DefaultPermissionEngine._queue)
        fake_pe = AsyncMock()
        fake_pe._queue = fake_queue
        commit_result = AllowSuccess(content="ok", data={"via": "user_click"})
        fake_pe.commit_resume = AsyncMock(return_value=commit_result)

        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-digest", "file_write")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-digest",
            "action": "approve",
            "scope": "session",
            "claim_nonce": "a" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # commit_resume must have been called with the arg_digest from the queue
        fake_pe.commit_resume.assert_awaited_once()
        call_args = fake_pe.commit_resume.call_args
        call_spec: ToolCallSpec = call_args[0][0]  # positional arg 0
        assert call_spec.arg_digest == expected_arg_digest, (
            f"P1#2: arg_digest must be read from queue, got {call_spec.arg_digest!r}"
        )
        assert call_spec.primary_arg == expected_primary_arg, (
            f"P1#2: primary_arg must be read from queue, got {call_spec.primary_arg!r}"
        )
        assert call_spec.dir_arg == expected_dir_arg, (
            f"P1#2: dir_arg must be read from queue, got {call_spec.dir_arg!r}"
        )
        assert cmd.goto == "tool_node"

    async def test_interrupt_helper_handles_session_mode_violation(self):
        """P2#3: interrupt_helper catches SessionModeViolation from pe.commit_resume.

        When the session transitions to FINISHING/COMPLETED/TIMED_OUT between
        preflight and graph commit, DefaultPermissionEngine.commit_resume raises
        SessionModeViolation.  interrupt_helper must catch it and return an error
        Command (not bubble it to AgentService.drive's rollback handler).
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage
        from app.domain.services.permission.errors import SessionModeViolation

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # PE that raises SessionModeViolation from commit_resume
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            side_effect=SessionModeViolation("session entered FINISHING between preflight and commit")
        )
        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-smv")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-smv",
            "action": "approve",
            "scope": "session",
            "claim_nonce": "b" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Must NOT bubble — must return a Command with an error ToolMessage
        assert isinstance(cmd, Command), (
            "P2#3 FAIL: SessionModeViolation was not caught; interrupt_helper did not return a Command"
        )
        assert cmd.goto == "tool_node", (
            f"P2#3 FAIL: expected goto='tool_node', got {cmd.goto!r}"
        )

        # pending_ask fields must be cleared
        assert cmd.update.get("pending_ask_outcome") is None
        assert cmd.update.get("pending_ask_tool_call_id") is None

        # pe_resume_outcomes must NOT contain the tool_call_id (outcome was not committed)
        pe_outcomes = cmd.update.get("pe_resume_outcomes") or {}
        assert "tc-smv" not in pe_outcomes, (
            "P2#3 FAIL: pe_resume_outcomes was written despite SessionModeViolation"
        )

    async def test_p1_1_ssm_failure_with_claim_nonce_fails_closed(self):
        """P1#1 (Codex round-10): SSM failure + claim_nonce present → fail closed.

        When HTTP preflight has already written a claim_nonce (PE path), an SSM
        failure during graph-layer interrupt_helper must NOT fall back to the
        legacy path.  Legacy approve writes approved_tool_call_ids which bypasses
        commit_resume nonce/mode validation, executes the tool without writing a
        grant/audit, and leaves the Redis confirmation stuck in 'processing'.

        Expected: _build_resume_error_command is returned (error ToolMessage +
        goto='tool_node') rather than the legacy approved_tool_call_ids update.
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # SSM that raises on get_mode_with_revision
        fake_ssm = AsyncMock()
        fake_ssm.get_mode_with_revision = AsyncMock(
            side_effect=RuntimeError("DB connection pool exhausted")
        )

        # PE that should NOT be reached (commit_resume must not be called)
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(return_value=None)

        state = _make_state_with_pending("tc-ssm-fail")
        config = _make_config(fake_pe, fake_ssm)

        # resume_payload carries claim_nonce → PE path was already claimed
        resume_payload = {
            "tool_call_id": "tc-ssm-fail",
            "action": "approve",
            "scope": "once",
            "claim_nonce": "c" * 32,  # non-None → PE claim active
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Must return a Command (fail-closed error path), not raise
        assert isinstance(cmd, Command), (
            "P1#1 FAIL: SSM failure with claim_nonce must return an error Command"
        )

        # Must NOT write approved_tool_call_ids (that would be legacy behaviour)
        legacy_ids = cmd.update.get("approved_tool_call_ids") or []
        assert "tc-ssm-fail" not in legacy_ids, (
            "P1#1 FAIL: approved_tool_call_ids was written — legacy path was taken "
            "despite active PE claim_nonce"
        )

        # commit_resume must NOT have been called
        fake_pe.commit_resume.assert_not_called()

        # pending_ask fields must be cleared (error command clears them)
        assert cmd.update.get("pending_ask_tool_call_id") is None

        # Must route back to tool_node (standard error command destination)
        assert cmd.goto == "tool_node", (
            f"P1#1 FAIL: expected goto='tool_node', got {cmd.goto!r}"
        )

    async def test_p1_1_ssm_failure_without_claim_nonce_falls_back_to_legacy(self):
        """P1#1 (Codex round-10): SSM failure without claim_nonce → legacy fallback allowed.

        Without an active PE claim, legacy fallback is safe: no nonce/mode
        violation and no orphaned Redis claim.
        """
        from langgraph.types import Command

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # SSM that always fails
        fake_ssm = AsyncMock()
        fake_ssm.get_mode_with_revision = AsyncMock(
            side_effect=RuntimeError("DB unavailable")
        )

        fake_pe = AsyncMock()
        state = _make_state_with_pending("tc-ssm-legacy")
        config = _make_config(fake_pe, fake_ssm)

        # No claim_nonce → legacy path is safe to fall back to
        resume_payload = {
            "tool_call_id": "tc-ssm-legacy",
            "action": "approve",
            "scope": "once",
            # claim_nonce intentionally absent
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Legacy path: approved_tool_call_ids written (no claim_nonce → falls back)
        # Note: the check here is that the test doesn't raise — legacy path
        # accepts the resume and writes approved_tool_call_ids.
        assert isinstance(cmd, Command), (
            "P1#1 FAIL: without claim_nonce, interrupt_helper should return a Command"
        )
        # The legacy path writes approved_tool_call_ids (not pe_resume_outcomes)
        approved = cmd.update.get("approved_tool_call_ids") or []
        assert "tc-ssm-legacy" in approved, (
            "P1#1 FAIL: legacy fallback without claim_nonce should write approved_tool_call_ids"
        )


# ---------------------------------------------------------------------------
# P2#1 (Round 13): generic Exception fallback in interrupt_helper
# ---------------------------------------------------------------------------


class TestInterruptHelperGenericException:
    async def test_interrupt_helper_handles_generic_pe_commit_exception(self):
        """P2#1 (round-13): Infrastructure errors from pe.commit_resume must be
        caught and converted to an error Command rather than bubbling up to the
        caller (which would leave the confirmation hanging in 'processing').

        When pe.commit_resume raises a generic Exception (e.g. Redis connection
        failure, DB timeout), interrupt_helper must:
        - NOT raise
        - Return a Command with goto='tool_node' (error ToolMessage)
        - Clear pending_ask_* fields
        - NOT write pe_resume_outcomes
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # PE that raises a generic RuntimeError from commit_resume
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            side_effect=RuntimeError("Redis connection pool exhausted")
        )
        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-infra-fail")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-infra-fail",
            "action": "approve",
            "scope": "session",
            "claim_nonce": "d" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Must NOT bubble — must return a Command
        assert isinstance(cmd, Command), (
            "P2#1 FAIL: generic RuntimeError from pe.commit_resume must not bubble up"
        )

        # Must route back to tool_node
        assert cmd.goto == "tool_node", (
            f"P2#1 FAIL: expected goto='tool_node', got {cmd.goto!r}"
        )

        # pending_ask fields must be cleared
        assert cmd.update.get("pending_ask_outcome") is None
        assert cmd.update.get("pending_ask_tool_call_id") is None
        assert cmd.update.get("pending_ask_artifact") is None
        assert cmd.update.get("pending_ask_tool_args") is None

        # pe_resume_outcomes must NOT contain the tool_call_id (commit failed)
        pe_outcomes = cmd.update.get("pe_resume_outcomes") or {}
        assert "tc-infra-fail" not in pe_outcomes, (
            "P2#1 FAIL: pe_resume_outcomes must not be written when commit_resume raises"
        )

        # There must be an error ToolMessage in messages
        messages = cmd.update.get("messages") or []
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_msgs, "P2#1 FAIL: error ToolMessage must be emitted"
        assert tool_msgs[0].status == "error", (
            "P2#1 FAIL: ToolMessage status must be 'error'"
        )

    async def test_interrupt_helper_handles_connection_error(self):
        """P2#1 follow-up: ConnectionError (another common infra error) is also caught."""
        from langgraph.types import Command

        interrupt_helper_fn = _build_interrupt_helper_fn()

        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            side_effect=ConnectionError("DB connection refused")
        )
        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-conn-err")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-conn-err",
            "action": "approve",
            "scope": "once",
            "claim_nonce": "e" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        assert isinstance(cmd, Command)
        assert cmd.goto == "tool_node"
        assert cmd.update.get("pending_ask_tool_call_id") is None


# ---------------------------------------------------------------------------
# P2 (Round-18): interrupt_helper must cleanup PE queue on error paths to
# prevent stale 'processing' entries that block future /resume attempts.
# ---------------------------------------------------------------------------


class TestInterruptHelperQueueCleanupOnError:
    async def test_interrupt_helper_cleans_queue_on_pe_commit_exception(self):
        """P2 (round-18): When pe.commit_resume raises a generic Exception,
        interrupt_helper must call pe.cleanup_pending_confirmation so the
        queue entry is not left stuck in 'processing'.

        A stale 'processing' entry prevents future /resume because the sweeper
        only rescues entries after a long timeout, and by then the graph
        checkpoint is gone so the user is permanently blocked.
        """
        from langgraph.types import Command
        from langchain_core.messages import ToolMessage

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # PE that raises a generic RuntimeError from commit_resume
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            side_effect=RuntimeError("Redis connection pool exhausted")
        )
        fake_pe.cleanup_pending_confirmation = AsyncMock()
        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-cleanup-commit-err")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-cleanup-commit-err",
            "action": "approve",
            "scope": "session",
            "claim_nonce": "f" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # cleanup_pending_confirmation must have been called to release the
        # 'processing' queue entry
        fake_pe.cleanup_pending_confirmation.assert_awaited_once()
        cleanup_args = fake_pe.cleanup_pending_confirmation.call_args
        assert cleanup_args is not None
        # session_id and tool_call_id must be passed correctly
        pos_args = cleanup_args[0]
        assert pos_args[0] == "s", (
            f"round-18 FAIL: session_id wrong, got {pos_args[0]!r}"
        )
        assert pos_args[1] == "tc-cleanup-commit-err", (
            f"round-18 FAIL: tool_call_id wrong, got {pos_args[1]!r}"
        )

        # Standard error command assertions
        assert isinstance(cmd, Command)
        assert cmd.goto == "tool_node"
        assert cmd.update.get("pending_ask_tool_call_id") is None

    async def test_interrupt_helper_cleans_queue_on_ssm_failure_with_nonce(self):
        """P2 (round-18): When SSM.get_mode_with_revision fails and claim_nonce is
        present, interrupt_helper must call pe.cleanup_pending_confirmation before
        returning the error Command.

        In this scenario PE preflight already claimed the queue entry (marked it
        'processing').  SSM failure means commit_resume will never run, so cleanup
        must happen here or the entry remains stuck.
        """
        from langgraph.types import Command

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # SSM that raises on get_mode_with_revision
        fake_ssm = AsyncMock()
        fake_ssm.get_mode_with_revision = AsyncMock(
            side_effect=RuntimeError("DB connection pool exhausted")
        )

        # PE with cleanup_pending_confirmation that we can assert on
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock()  # must NOT be called
        fake_pe.cleanup_pending_confirmation = AsyncMock()

        state = _make_state_with_pending("tc-cleanup-ssm-fail")
        config = _make_config(fake_pe, fake_ssm)

        # claim_nonce is present → PE claim is active (preflight succeeded)
        resume_payload = {
            "tool_call_id": "tc-cleanup-ssm-fail",
            "action": "approve",
            "scope": "once",
            "claim_nonce": "g" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # cleanup_pending_confirmation must have been called
        fake_pe.cleanup_pending_confirmation.assert_awaited_once()
        cleanup_args = fake_pe.cleanup_pending_confirmation.call_args
        assert cleanup_args is not None
        pos_args = cleanup_args[0]
        assert pos_args[0] == "s", (
            f"round-18 FAIL: session_id wrong, got {pos_args[0]!r}"
        )
        assert pos_args[1] == "tc-cleanup-ssm-fail", (
            f"round-18 FAIL: tool_call_id wrong, got {pos_args[1]!r}"
        )

        # commit_resume must NOT have been called (SSM failed before it)
        fake_pe.commit_resume.assert_not_called()

        # Standard error command assertions
        assert isinstance(cmd, Command)
        assert cmd.goto == "tool_node"
        assert cmd.update.get("pending_ask_tool_call_id") is None


# ---------------------------------------------------------------------------
# P1#1 (Round-25): nonce mismatch must NOT delete the new owner's queue entry
# ---------------------------------------------------------------------------


class TestInterruptHelperNonceMismatchNoCleanup:
    """P1#1 (round-25): claim_nonce_mismatch PolicyConflict must NOT cleanup.

    Scenario:
    - An old/timed-out resume carries a stale claim_nonce.
    - The same confirmation was already swept/reopened and re-claimed by a new
      owner (new nonce written).
    - pe.commit_resume raises PolicyConflict("claim_nonce_mismatch") because
      the nonces don't match.
    - The old caller must NOT call cleanup_pending_confirmation — doing so would
      delete the new owner's in-flight confirmation state, turning their next
      /resume into "no_pending_confirmation" or permanently blocking them.

    Expected: cleanup_pending_confirmation is NOT called; error Command is
    returned so the old caller sees a transient failure.
    """

    async def test_interrupt_helper_does_not_cleanup_on_nonce_mismatch(self):
        """PolicyConflict('claim_nonce_mismatch') → cleanup_pending_confirmation NOT called."""
        from langgraph.types import Command
        from app.domain.services.permission.errors import PolicyConflict

        interrupt_helper_fn = _build_interrupt_helper_fn()

        # PE that raises nonce mismatch from commit_resume
        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            side_effect=PolicyConflict("claim_nonce_mismatch: stored=new_nonce provided=old_nonce")
        )
        fake_pe.cleanup_pending_confirmation = AsyncMock()
        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-nonce-mismatch")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-nonce-mismatch",
            "action": "approve",
            "scope": "once",
            "claim_nonce": "old_stale_nonce" + "x" * 16,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # cleanup_pending_confirmation must NOT have been called — doing so would
        # delete the new owner's queue entry and corrupt their in-flight confirmation.
        fake_pe.cleanup_pending_confirmation.assert_not_awaited()

        # Must return an error Command (not raise, not succeed)
        assert isinstance(cmd, Command), (
            "P1#1 round-25 FAIL: nonce mismatch must return an error Command"
        )
        assert cmd.goto == "tool_node", (
            f"P1#1 round-25 FAIL: expected goto='tool_node', got {cmd.goto!r}"
        )
        # pending_ask fields must be cleared
        assert cmd.update.get("pending_ask_tool_call_id") is None

    async def test_interrupt_helper_cleans_up_on_other_policy_conflict(self):
        """Non-nonce-mismatch PolicyConflict still triggers cleanup (regression guard).

        Only 'claim_nonce_mismatch' PolicyConflict must skip cleanup. Other
        PolicyConflict variants (e.g. 'arg_digest_mismatch') still own the queue
        entry and must clean it up.
        """
        from langgraph.types import Command
        from app.domain.services.permission.errors import PolicyConflict

        interrupt_helper_fn = _build_interrupt_helper_fn()

        fake_pe = AsyncMock()
        fake_pe.commit_resume = AsyncMock(
            side_effect=PolicyConflict("arg_digest_mismatch: stored=abc provided=xyz")
        )
        fake_pe.cleanup_pending_confirmation = AsyncMock()
        fake_ssm = _make_fake_ssm()

        state = _make_state_with_pending("tc-arg-digest-mismatch")
        config = _make_config(fake_pe, fake_ssm)

        resume_payload = {
            "tool_call_id": "tc-arg-digest-mismatch",
            "action": "approve",
            "scope": "once",
            "claim_nonce": "a" * 32,
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.domain.services.graphs.react_graph.interrupt",
                lambda payload: resume_payload,
            )
            cmd = await interrupt_helper_fn(state, config)

        # Non-nonce-mismatch PolicyConflict must still cleanup
        fake_pe.cleanup_pending_confirmation.assert_awaited_once()

        assert isinstance(cmd, Command)
        assert cmd.goto == "tool_node"
