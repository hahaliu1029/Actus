"""PE-0 Phase 7 — DI factory smoke tests for PermissionEngine / SSM wiring.

Three tests (plan lines 4537-4587):
  1. get_confirmation_queue returns ConfirmationQueue(redis_client.client)
  2. build_permission_engine returns a DefaultPermissionEngine
  3. build_session_state_machine returns a DefaultSessionStateMachine
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: get_confirmation_queue unwraps .client
# ---------------------------------------------------------------------------

def test_get_confirmation_queue_unwraps_redis_client():
    """get_confirmation_queue(redis_client) must pass redis_client.client (raw
    redis.asyncio.Redis) to ConfirmationQueue, NOT the RedisClient wrapper.
    C-R2-P0-3 correction."""
    from app.interfaces.service_dependencies import get_confirmation_queue
    from app.domain.services.permission.confirmation_queue import ConfirmationQueue

    raw_redis = MagicMock()
    wrapper = MagicMock()
    wrapper.client = raw_redis

    queue = get_confirmation_queue(redis_client=wrapper)

    assert isinstance(queue, ConfirmationQueue)
    # The queue's internal _redis must be the raw client, not the wrapper.
    assert queue._redis is raw_redis


# ---------------------------------------------------------------------------
# Test 2: build_permission_engine returns DefaultPermissionEngine
# ---------------------------------------------------------------------------

def test_build_permission_engine_returns_default_engine():
    """build_permission_engine(...) must return a DefaultPermissionEngine
    with an escalation_registry that contains 'smart_approve' when
    summary_llm is not None."""
    from app.application.composition.graph_assembly import build_permission_engine
    from app.domain.services.permission.default_engine import DefaultPermissionEngine

    uow_factory = MagicMock()
    writer = MagicMock()
    queue = MagicMock()
    session_machine = MagicMock()
    reader = MagicMock()

    # Patch SmartApprove so no real LLM is needed
    with patch("app.domain.services.smart_approve.SmartApprove") as mock_sa_cls:
        mock_sa_cls.return_value = MagicMock()
        summary_llm = MagicMock()

        engine = build_permission_engine(
            uow_factory=uow_factory,
            writer=writer,
            queue=queue,
            session_machine=session_machine,
            reader=reader,
            summary_llm=summary_llm,
        )

    assert isinstance(engine, DefaultPermissionEngine)
    assert "smart_approve" in engine._escalation_registry


def test_build_permission_engine_no_summary_llm_skips_smart_approve():
    """When summary_llm is None, escalation_registry must be empty (no smart_approve)."""
    from app.application.composition.graph_assembly import build_permission_engine
    from app.domain.services.permission.default_engine import DefaultPermissionEngine

    engine = build_permission_engine(
        uow_factory=MagicMock(),
        writer=MagicMock(),
        queue=MagicMock(),
        session_machine=MagicMock(),
        reader=MagicMock(),
        summary_llm=None,
    )

    assert isinstance(engine, DefaultPermissionEngine)
    assert len(engine._escalation_registry) == 0


# ---------------------------------------------------------------------------
# Test 3: build_session_state_machine returns DefaultSessionStateMachine
# ---------------------------------------------------------------------------

def test_build_session_state_machine_returns_default_ssm():
    """build_session_state_machine(...) must return a DefaultSessionStateMachine
    with the supplied uow_factory stored on the instance."""
    from app.application.composition.graph_assembly import build_session_state_machine
    from app.domain.services.session.default_state_machine import DefaultSessionStateMachine

    uow_factory = MagicMock()
    raw_redis = MagicMock()

    ssm = build_session_state_machine(
        uow_factory=uow_factory,
        redis=raw_redis,
        event_publisher=None,
    )

    assert isinstance(ssm, DefaultSessionStateMachine)
    assert ssm._uow_factory is uow_factory
    assert ssm._redis is raw_redis


# ---------------------------------------------------------------------------
# P3#1: build_permission_engine forwards decision_recorder to DefaultPermissionEngine
# ---------------------------------------------------------------------------


def test_build_permission_engine_passes_decision_recorder():
    """P3#1: build_permission_engine must forward the decision_recorder kwarg to
    DefaultPermissionEngine so that OTel canonical attributes (decision_stage etc.)
    are emitted on every stage transition.
    """
    from app.application.composition.graph_assembly import build_permission_engine
    from app.domain.services.permission.default_engine import DefaultPermissionEngine

    recorder = MagicMock()

    with patch("app.domain.services.smart_approve.SmartApprove"):
        engine = build_permission_engine(
            uow_factory=MagicMock(),
            writer=MagicMock(),
            queue=MagicMock(),
            session_machine=MagicMock(),
            reader=MagicMock(),
            summary_llm=MagicMock(),
            decision_recorder=recorder,
        )

    assert isinstance(engine, DefaultPermissionEngine)
    # The PE must store the exact recorder callable we passed in.
    # DefaultPermissionEngine wraps None with a no-op lambda, but when a real
    # callable is supplied it should be stored directly as _decision_recorder.
    assert engine._decision_recorder is recorder, (
        "DefaultPermissionEngine._decision_recorder must be the recorder passed to build_permission_engine"
    )


def test_build_permission_engine_default_recorder_is_noop():
    """When decision_recorder is omitted (default None), DefaultPermissionEngine
    receives None and uses its internal no-op lambda — no AttributeError."""
    from app.application.composition.graph_assembly import build_permission_engine
    from app.domain.services.permission.default_engine import DefaultPermissionEngine

    engine = build_permission_engine(
        uow_factory=MagicMock(),
        writer=MagicMock(),
        queue=MagicMock(),
        session_machine=MagicMock(),
        reader=MagicMock(),
        summary_llm=None,
    )

    assert isinstance(engine, DefaultPermissionEngine)
    # Should be callable (the no-op lambda or a real recorder) — not None.
    assert callable(engine._decision_recorder)
