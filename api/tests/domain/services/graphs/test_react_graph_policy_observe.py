from __future__ import annotations

import pytest
from langgraph.types import Command

from core.config import Settings
from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler

# Canonical node-driver helpers (see tests/domain/services/graphs/test_react_graph_pe_dispatch.py):
#   _build_tool_node_fn() -> tool_node afunc (registers a real `shell_execute` tool)
#   _make_state(tool_name, tool_args, call_id="call-1") -> ReactGraphState dict
#   _make_config(fake_pe, fake_ssm, *, user_id="u1", session_id="s1", extra=None) -> {"configurable": {...}}
#   FakeRecordingPE() with .calls / .next_outcome ; _make_fake_ssm(mode="auto", revision=1)
from tests.domain.services.graphs.test_react_graph_pe_dispatch import (  # noqa: E501
    FakeRecordingPE,
    _build_tool_node_fn,
    _make_config,
    _make_fake_ssm,
    _make_state,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _SpySink:
    def __init__(self) -> None:
        self.calls: list = []

    async def record(self, snapshot) -> None:
        self.calls.append(snapshot)


def _flag(monkeypatch, value: bool):
    monkeypatch.setattr(
        "app.domain.services.graphs.react_graph.get_settings",
        lambda: Settings(env="test", sandbox_policy_compiler_enabled=value),
    )


async def test_seam_b_records_once_when_on_allowed(monkeypatch):
    _flag(monkeypatch, True)
    spy = _SpySink()
    fn = _build_tool_node_fn()
    pe = FakeRecordingPE()
    pe.next_outcome = "ask"  # deterministic routing; emission is BEFORE pe.evaluate
    result = await fn(
        _make_state("shell_execute", {"command": "ls"}),
        _make_config(pe, _make_fake_ssm(), extra={"policy_snapshot_sink": spy}),
    )
    assert isinstance(result, Command)
    assert len(spy.calls) == 1
    snap = spy.calls[0]
    assert snap.surface == "tool_call"
    assert snap.decision.verdict == "ok"
    assert snap.subject.tool_name == "shell_execute"
    # best-effort subject defaults at Seam B (§8.4b — full set, codex planR2 P3)
    assert snap.subject.sandbox_id is None
    assert snap.subject.sandbox_generation == 0
    assert snap.subject.worker_type == "unknown"
    assert snap.subject.depth == 0


async def test_seam_b_records_zero_when_off(monkeypatch):
    _flag(monkeypatch, False)
    spy = _SpySink()
    fn = _build_tool_node_fn()
    pe = FakeRecordingPE()
    pe.next_outcome = "ask"
    await fn(
        _make_state("shell_execute", {"command": "ls"}),
        _make_config(pe, _make_fake_ssm(), extra={"policy_snapshot_sink": spy}),
    )
    assert spy.calls == []


async def test_seam_b_inv0_denied_routing_identical(monkeypatch):
    # rm -rf / is AST-denied in BOTH ON and OFF; the observe emission must not
    # change the denial routing (INV-0). The snapshot records the denial when ON.
    fn = _build_tool_node_fn()
    _flag(monkeypatch, True)
    spy = _SpySink()
    res_on = await fn(
        _make_state("shell_execute", {"command": "rm -rf /"}),
        _make_config(FakeRecordingPE(), _make_fake_ssm(), extra={"policy_snapshot_sink": spy}),
    )
    assert len(spy.calls) == 1 and spy.calls[0].decision.verdict == "denied"
    # `rm -rf /` resolves against the default cwd (/root) and trips the AST
    # validator's cwd-boundary rule first (code="cwd_boundary"), which the
    # compiler maps verbatim. (Plan's literal "fs_destructive" predated the
    # real Task 1/2 validator classification — corrected to the actual code.)
    assert spy.calls[0].decision.reason_code == "cwd_boundary"

    _flag(monkeypatch, False)
    res_off = await fn(
        _make_state("shell_execute", {"command": "rm -rf /"}),
        _make_config(FakeRecordingPE(), _make_fake_ssm(), extra={"policy_snapshot_sink": _SpySink()}),
    )
    assert isinstance(res_on, Command) and isinstance(res_off, Command)
    assert getattr(res_on, "goto", None) == getattr(res_off, "goto", None)


async def test_seam_b_child_flow_no_sink_records_nothing(monkeypatch):
    # A coordinator-child runner's _build_config emits policy_snapshot_sink as
    # PRESENT-BUT-None (the key sits in the unconditional base dict, but no child
    # sink is injected). The gate must skip for BOTH the real child shape
    # ("key present = None") and a defensive "key absent" — proven by the
    # compiler never being invoked (§8.8). [codex R1#2]
    _flag(monkeypatch, True)
    called: list = []
    monkeypatch.setattr(
        SandboxPolicyCompiler, "compile_tool_call",
        lambda self, inp: called.append(inp),
    )
    fn = _build_tool_node_fn()
    for child_extra in ({"policy_snapshot_sink": None}, None):  # present-but-None, then absent
        pe = FakeRecordingPE()
        pe.next_outcome = "ask"
        await fn(
            _make_state("shell_execute", {"command": "ls"}),
            _make_config(pe, _make_fake_ssm(), extra=child_extra),
        )
    assert called == []


async def test_seam_b_sink_raise_does_not_propagate(monkeypatch):
    # §8.7 best-effort at Seam B: a sink whose record() raises must NOT propagate
    # out of the node or change routing. [codex planR2 P2]
    _flag(monkeypatch, True)

    class _RaisingSink:
        async def record(self, snapshot):
            raise RuntimeError("boom")

    fn = _build_tool_node_fn()
    pe = FakeRecordingPE()
    pe.next_outcome = "ask"
    result = await fn(  # must NOT raise
        _make_state("shell_execute", {"command": "ls"}),
        _make_config(pe, _make_fake_ssm(), extra={"policy_snapshot_sink": _RaisingSink()}),
    )
    assert isinstance(result, Command)


async def test_seam_b_redacts_raw_exec_dir(monkeypatch):
    # §8.4 builder/wiring redaction: a raw attacker-controlled exec_dir must be
    # DIGESTED, never reaching the snapshot/log. [codex planR2 P3]
    import json
    _flag(monkeypatch, True)
    spy = _SpySink()
    fn = _build_tool_node_fn()
    pe = FakeRecordingPE()
    pe.next_outcome = "ask"
    await fn(
        _make_state("shell_execute", {"command": "ls", "exec_dir": "/home/ubuntu/SECRET_DIR"}),
        _make_config(pe, _make_fake_ssm(), extra={"policy_snapshot_sink": spy}),
    )
    assert len(spy.calls) == 1
    snap = spy.calls[0]
    assert len(snap.command.effective_cwd_digest) == 64        # digested, not raw
    assert snap.command.is_default_cwd is False                # exec_dir was provided
    assert "SECRET_DIR" not in json.dumps(snap.model_dump(mode="json"))


async def test_seam_b_legacy_gate_records_once(monkeypatch):
    # With NO permission_engine/session_state_machine in configurable, the node
    # FALLS THROUGH to the LEGACY tool_node gate (react_graph: `if _pe is not None
    # and _ssm is not None` at :2156 is False → legacy path). This is the only
    # positive test of the legacy emission block, which references `ast_result`
    # (no leading underscore) — a typo there would otherwise go uncaught. [codex planR5 P2]
    _flag(monkeypatch, True)
    spy = _SpySink()
    fn = _build_tool_node_fn()
    result = await fn(
        _make_state("shell_execute", {"command": "ls"}),
        _make_config(None, None, extra={"policy_snapshot_sink": spy}),  # no PE/SSM → legacy gate
    )
    assert isinstance(result, Command)
    assert len(spy.calls) == 1
    assert spy.calls[0].surface == "tool_call"
    assert spy.calls[0].decision.verdict == "ok"
