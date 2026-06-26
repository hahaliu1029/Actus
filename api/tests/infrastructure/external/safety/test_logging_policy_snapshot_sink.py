from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from app.domain.external.policy_snapshot_sink import NoopPolicySnapshotSink
from app.domain.models.sandbox_policy import (
    SandboxSettingsView,
    ToolCallInput,
    ValidationResultView,
    sha256_hexdigest,
)
from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler
from app.infrastructure.external.safety.logging_policy_snapshot_sink import (
    LoggingPolicySnapshotSink,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _view():
    return SandboxSettingsView(
        external_address=False, image="img", network=None, mem_limit="4g",
        default_cwd="/root", has_https_proxy=False, has_http_proxy=False,
        has_no_proxy=True, no_proxy_digest=sha256_hexdigest("SECRET.internal"),
        memory_mount_target="/workspace/.memory", memory_mount_enabled=True,
    )


def _tc_snapshot():
    return SandboxPolicyCompiler().compile_tool_call(ToolCallInput(
        session_id="s1", sandbox_id=None, sandbox_generation=0, worker_type="unknown",
        depth=0, tool_call_id="tc-1", tool_name="shell_execute", tool_source="native",
        command="cat /etc/SECRET_CMD",
        validation=ValidationResultView(
            allowed=True, code="ok", effective_cwd="/home/ubuntu/SECRET_DIR"),
        is_default_cwd=False, settings=_view(),
    ))


async def test_log_contains_digests_not_raw(caplog):
    snap = _tc_snapshot()
    with caplog.at_level("INFO", logger="sandbox.policy"):
        await LoggingPolicySnapshotSink().record(snap)
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "tc-1" in line                                    # tool_call_id permitted
    assert sha256_hexdigest("cat /etc/SECRET_CMD") in line   # tool_call_digest present
    assert "SECRET_CMD" not in line                          # raw command absent
    assert "SECRET_DIR" not in line                          # raw effective_cwd absent
    assert "SECRET.internal" not in line                     # raw no_proxy absent


async def test_sink_swallows_render_error():
    class _Boom:
        surface = "tool_call"
        def model_dump(self, *a, **k):
            raise RuntimeError("boom")
    await LoggingPolicySnapshotSink().record(_Boom())  # type: ignore[arg-type]  # must not raise


async def test_noop_records_nothing(caplog):
    snap = _tc_snapshot()
    with caplog.at_level("INFO", logger="sandbox.policy"):
        await NoopPolicySnapshotSink().record(snap)
    assert not [r for r in caplog.records if r.name == "sandbox.policy"]


# ---- §8.9 non-suspending guard (protects INV-0) --------------------------- #
_FORBIDDEN_CALLS = {"create_task", "ensure_future", "run_in_executor", "to_thread"}


@pytest.mark.parametrize("cls", [LoggingPolicySnapshotSink, NoopPolicySnapshotSink])
def test_record_body_is_non_suspending(cls):
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.record)))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Await)], (
        f"{cls.__name__}.record must contain no await (INV-0 non-suspending)"
    )
    bad = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Attribute) and n.func.attr in _FORBIDDEN_CALLS)
            or (isinstance(n.func, ast.Name) and n.func.id in _FORBIDDEN_CALLS)  # bare create_task/to_thread
        )
    ]
    assert not bad, f"{cls.__name__}.record must not schedule/offload work (INV-0)"
