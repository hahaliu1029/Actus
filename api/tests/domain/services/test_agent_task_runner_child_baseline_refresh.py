"""C2b §4.2 — _refresh_child_permission_baseline: best-effort + live-mode-gated."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.permission.child_permission_context import (
    ChildBudget, ChildPermissionContext, SpawnManifest,
)

pytestmark = pytest.mark.anyio


def _cpc(*, rev: int) -> ChildPermissionContext:
    return ChildPermissionContext(
        parent_session_id="p1", child_session_id="c1", coordinator_run_id="r1",
        work_unit_id="wu1",
        spawn_manifest=SpawnManifest(
            allowed_tools=frozenset({"file_read"}), path_leases=(),
            runtime_caps=frozenset(),
        ),
        session_mode_revision=rev,
        budget=ChildBudget(max_tool_calls=100, max_token_cost_usd=1.0,
                           max_wallclock_seconds=600),
    )


def _runner(*, cpc, ssm_result=None, ssm_raises=None):
    r = object.__new__(AgentTaskRunner)
    r._session_id = "c1"
    r._coordinator_child_permission_context = cpc
    ssm = MagicMock()
    if ssm_raises is not None:
        ssm.get_mode_with_revision = AsyncMock(side_effect=ssm_raises)
    else:
        ssm.get_mode_with_revision = AsyncMock(return_value=ssm_result)
    r._session_state_machine = ssm
    r._flow = MagicMock()
    return r


async def test_refresh_replaces_baseline_on_running_plus_one():
    r = _runner(cpc=_cpc(rev=5), ssm_result=(SessionStatus.RUNNING, 6))
    await r._refresh_child_permission_baseline()
    assert r._coordinator_child_permission_context.session_mode_revision == 6
    # flow received the refreshed cpc
    forwarded = r._flow.set_child_permission_context.call_args.args[0]
    assert forwarded.session_mode_revision == 6


async def test_refresh_no_replace_when_not_running():
    r = _runner(cpc=_cpc(rev=5), ssm_result=(SessionStatus.TAKEOVER_PENDING, 6))
    await r._refresh_child_permission_baseline()
    assert r._coordinator_child_permission_context.session_mode_revision == 5
    r._flow.set_child_permission_context.assert_not_called()


async def test_refresh_no_replace_when_rev_jumps_more_than_one():
    """RUNNING(8) after a RUNNING→TAKEOVER→RUNNING round-trip is != baseline+1."""
    r = _runner(cpc=_cpc(rev=5), ssm_result=(SessionStatus.RUNNING, 8))
    await r._refresh_child_permission_baseline()
    assert r._coordinator_child_permission_context.session_mode_revision == 5
    r._flow.set_child_permission_context.assert_not_called()


async def test_refresh_no_replace_on_read_failure():
    r = _runner(cpc=_cpc(rev=5), ssm_raises=RuntimeError("db blip"))
    await r._refresh_child_permission_baseline()  # must not raise
    assert r._coordinator_child_permission_context.session_mode_revision == 5
    r._flow.set_child_permission_context.assert_not_called()


async def test_refresh_noop_when_no_cpc():
    r = object.__new__(AgentTaskRunner)
    r._session_id = "root1"
    # no _coordinator_child_permission_context attribute at all (root runner)
    await r._refresh_child_permission_baseline()  # getattr-defensive, no raise
