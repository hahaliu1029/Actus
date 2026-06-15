"""C2 coordinator-cancel — repo surface (presence/signature) unit slice.

The real SQL is exercised in tests/integration/test_coordinator_cancel_queries.py
(CI only). These assertions catch accidental removal/rename of the two new
queries on both the Protocol and the concrete DBSessionRepository.
"""
from __future__ import annotations

import inspect

from app.domain.repositories.session_repository import SessionRepository
from app.infrastructure.repositories.db_session_repository import DBSessionRepository


def test_fanout_query_present_and_async():
    assert hasattr(SessionRepository, "find_running_mailbox_children_for_parent")
    assert hasattr(DBSessionRepository, "find_running_mailbox_children_for_parent")
    assert inspect.iscoroutinefunction(
        DBSessionRepository.find_running_mailbox_children_for_parent
    )
    sig = inspect.signature(
        DBSessionRepository.find_running_mailbox_children_for_parent
    )
    assert list(sig.parameters)[1] == "parent_session_id"


def test_reaper_query_present_and_async():
    assert hasattr(
        SessionRepository, "find_terminal_coordinator_children_with_active_sandbox"
    )
    assert hasattr(
        DBSessionRepository, "find_terminal_coordinator_children_with_active_sandbox"
    )
    assert inspect.iscoroutinefunction(
        DBSessionRepository.find_terminal_coordinator_children_with_active_sandbox
    )
