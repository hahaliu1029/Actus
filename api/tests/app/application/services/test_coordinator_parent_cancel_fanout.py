"""Unit tests for CoordinatorParentCancelFanout (spec §3.2 / §5 tests 1–9).

Pure / fake-driven — no live redis. Uses the REAL pure CoordinatorEnvelopeFactory
(envelope shape is real) + a capturing publisher + fake repo/starter.
"""
from __future__ import annotations

import pytest

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.application.services.coordinator_parent_cancel_fanout import (
    CancelFanoutResult,
    CoordinatorParentCancelFanout,
)
from app.domain.models.mailbox_envelope import MailboxEnvelopeType
from app.domain.repositories.session_repository import ChildLineageRow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRepo:
    def __init__(self, children=None, raises=False):
        self._children = children or []
        self._raises = raises
        self.calls: list[str] = []

    async def find_running_mailbox_children_for_parent(self, parent_session_id):
        self.calls.append(parent_session_id)
        if self._raises:
            raise RuntimeError("enumeration down")
        return list(self._children)


class _CapturingPublisher:
    def __init__(self, fail_on=None):
        self.published = []
        self._fail_on = set(fail_on or [])

    async def publish(self, envelope):
        if envelope.child_session_id in self._fail_on:
            raise RuntimeError("publish boom")
        self.published.append(envelope)


class _FakeStarter:
    def __init__(self, raises=False):
        self.calls: list[list[str]] = []
        self._raises = raises

    def request_stop_started(self, child_session_ids, reason=None):
        self.calls.append(list(child_session_ids))
        if self._raises:
            raise RuntimeError("fast-path boom")


def _children(n, run_id="run-1"):
    return [ChildLineageRow(f"c{i}", run_id, f"wu-{i}") for i in range(n)]


def _make(repo=None, publisher=None, starter=None):
    return CoordinatorParentCancelFanout(
        session_repository=repo,
        envelope_factory=CoordinatorEnvelopeFactory(),
        mailbox_publisher=publisher,
        child_runner_starter=starter,
    )


# 1 — happy path
async def test_happy_path_three_children():
    repo = _FakeRepo(_children(3))
    pub = _CapturingPublisher()
    starter = _FakeStarter()
    fanout = _make(repo, pub, starter)
    res = await fanout.cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(3, 3, 0)
    assert len(pub.published) == 3
    assert all(e.type == MailboxEnvelopeType.CANCEL_REQUEST for e in pub.published)
    assert all(e.parent_session_id == "p1" for e in pub.published)
    assert {e.child_session_id for e in pub.published} == {"c0", "c1", "c2"}
    assert all(e.payload["reason"] == "parent_cancel" for e in pub.published)
    # correlation_id == the child's non-null coordinator_run_id ("run-1"),
    # NOT the parent ("p1") — kills a "always use parent_session_id" mutation
    # the None-fallback test (test 9) can't catch (R3 P3#1).
    assert all(e.correlation_id == "run-1" for e in pub.published)
    assert starter.calls == [["c0", "c1", "c2"]]


# 2 — empty
async def test_empty_no_publish_no_fastpath():
    repo = _FakeRepo([])
    pub = _CapturingPublisher()
    starter = _FakeStarter()
    res = await _make(repo, pub, starter).cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(0, 0, 0)
    assert pub.published == []
    assert starter.calls == []


# 3 — partial publish failure
async def test_partial_publish_failure():
    repo = _FakeRepo(_children(3))
    pub = _CapturingPublisher(fail_on=["c1"])
    res = await _make(repo, pub, _FakeStarter()).cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(3, 2, 1)
    assert {e.child_session_id for e in pub.published} == {"c0", "c2"}


# 4 — total publish failure (INV-C2)
async def test_total_publish_failure_never_raises():
    repo = _FakeRepo(_children(3))
    pub = _CapturingPublisher(fail_on=["c0", "c1", "c2"])
    res = await _make(repo, pub, _FakeStarter()).cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(3, 0, 3)


# 5 — enumeration failure (INV-C2; mutation: drop the enumeration try/except -> raises)
async def test_enumeration_failure_swallowed():
    repo = _FakeRepo(raises=True)
    pub = _CapturingPublisher()
    starter = _FakeStarter()
    res = await _make(repo, pub, starter).cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(0, 0, 0)
    assert pub.published == []
    assert starter.calls == []


# 6 — null deps (INV-C4)
async def test_null_deps_noop():
    fanout = CoordinatorParentCancelFanout()  # all deps default None
    res = await fanout.cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(0, 0, 0)


# 7 — no starter -> fast-path skipped, envelopes still published
async def test_no_starter_publishes():
    repo = _FakeRepo(_children(2))
    pub = _CapturingPublisher()
    fanout = CoordinatorParentCancelFanout(
        session_repository=repo,
        envelope_factory=CoordinatorEnvelopeFactory(),
        mailbox_publisher=pub,
    )  # child_runner_starter defaults None
    res = await fanout.cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(2, 2, 0)
    assert len(pub.published) == 2


# 8 — fast-path raises -> swallowed, envelopes published (mutation: drop its try/except -> raises)
async def test_fastpath_failure_swallowed():
    repo = _FakeRepo(_children(2))
    pub = _CapturingPublisher()
    starter = _FakeStarter(raises=True)
    res = await _make(repo, pub, starter).cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(2, 2, 0)
    assert len(pub.published) == 2  # publish proceeds despite fast-path error


# 9 — correlation_id falls back to parent when run_id is None
async def test_correlation_id_falls_back_to_parent():
    repo = _FakeRepo([ChildLineageRow("c0", None, None)])
    pub = _CapturingPublisher()
    res = await _make(repo, pub, _FakeStarter()).cancel_children(parent_session_id="p1")
    assert res == CancelFanoutResult(1, 1, 0)
    assert pub.published[0].correlation_id == "p1"
