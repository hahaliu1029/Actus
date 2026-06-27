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
    assert snap.enforcement_mode == "enforce"  # C5b §6: tool_call snapshots now enforce
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


async def test_pe_gate_routing_identity_allow_vs_deny(monkeypatch):
    # INV-0 wiring (§8.4 PE): an allowed command reaches pe.evaluate; an AST-denied
    # command is gated BEFORE pe.evaluate (the gate `continue`s). No flag toggle —
    # asserted against the known-correct routing.
    _flag(monkeypatch, True)
    fn = _build_tool_node_fn()

    pe_allow = FakeRecordingPE()
    pe_allow.next_outcome = "allow"
    res_allow = await fn(
        _make_state("shell_execute", {"command": "ls"}),
        _make_config(pe_allow, _make_fake_ssm(), extra={"policy_snapshot_sink": _SpySink()}),
    )
    assert isinstance(res_allow, Command)
    assert len(pe_allow.calls) == 1  # passed AST gate → reached PE

    pe_deny = FakeRecordingPE()
    pe_deny.next_outcome = "allow"  # would allow IF reached — must NOT be reached
    res_deny = await fn(
        _make_state("shell_execute", {"command": "rm -rf /"}),
        _make_config(pe_deny, _make_fake_ssm(), extra={"policy_snapshot_sink": _SpySink()}),
    )
    assert isinstance(res_deny, Command)
    assert pe_deny.calls == []  # AST-denied → gated before PE.evaluate


# NOTE: the gate-level surrogate-exec_dir test lives in Task 4 (LEGACY path), NOT here.
# PE runs RiskAssessor for native tools (`_risk_assessor.assess` at react_graph.py:1450) BEFORE
# the AST gate, and `_compute_arg_digest` strict-utf-8-encodes exec_dir (risk_assessor.py:269) —
# a lone-surrogate exec_dir raises UnicodeEncodeError there pre-C5b, never reaching
# build_command_policy. The legacy gate runs validate() before RiskAssessor (N1 invariant
# `validate < assess`), so a DENIED command's surrogate exec_dir reaches C5b's build_command_policy
# there without hitting the strict digest. [codex planR3 P1]


def _command_text(cmd) -> str:
    """Concatenate the text of any messages a tool_node Command carries back.
    Defined here in Task 3; reused by Task 4's legacy diagnosis test (do not redefine)."""
    update = getattr(cmd, "update", None)
    if not isinstance(update, dict):
        return ""
    parts: list[str] = []
    for m in update.get("messages", []) or []:
        content = getattr(m, "content", "")
        parts.append(content if isinstance(content, str) else str(content))
    return " ".join(parts)


async def test_pe_gate_denial_diagnosis_byte_identical(monkeypatch):
    # INV-6 (§8.10) PE path: the PE gate's denial content equals the validator-owned
    # helper output — C5b changed only the decision predicate, not the diagnosis. PE has
    # its own finalizer (_finalize_pe_outcome), so this is coverage distinct from legacy.
    # [codex planR2 P2]
    from app.domain.services.safety.shell_ast_validator import format_denied_content, validate

    _flag(monkeypatch, True)
    default_cwd = Settings(env="test").sandbox_default_cwd
    expected = format_denied_content(
        validate("rm -rf /", effective_cwd=default_cwd), original_command="rm -rf /"
    )
    fn = _build_tool_node_fn()
    pe = FakeRecordingPE()
    pe.next_outcome = "allow"  # would allow IF reached — proves the AST gate denied first
    res = await fn(
        _make_state("shell_execute", {"command": "rm -rf /"}),
        _make_config(pe, _make_fake_ssm(), extra={"policy_snapshot_sink": _SpySink()}),
    )
    assert expected and _command_text(res) == expected  # single denial msg → exact equality [codex planR3 P3]
    assert pe.calls == []  # AST-denied before PE.evaluate


# _command_text was defined in Task 3 — reuse it here; do NOT redefine it.
async def test_legacy_gate_routing_identity_allow_vs_deny(monkeypatch):
    # INV-0 wiring (§8.4 legacy): with NO PE/SSM the node uses the LEGACY tool_node
    # gate. An allowed command runs the tool (no denial); a denied command surfaces
    # the validator's [AST 拦截] denial. The two route to distinct results.
    _flag(monkeypatch, True)
    fn = _build_tool_node_fn()

    res_allow = await fn(
        _make_state("shell_execute", {"command": "ls"}),
        _make_config(None, None, extra={"policy_snapshot_sink": _SpySink()}),
    )
    res_deny = await fn(
        _make_state("shell_execute", {"command": "rm -rf /"}),
        _make_config(None, None, extra={"policy_snapshot_sink": _SpySink()}),
    )
    assert isinstance(res_allow, Command) and isinstance(res_deny, Command)
    allow_text, deny_text = _command_text(res_allow), _command_text(res_deny)
    assert "[AST 拦截]" not in allow_text   # allowed → not denied
    assert "[AST 拦截]" in deny_text        # AST-denied → denial surfaced
    assert allow_text != deny_text          # the decision discriminates


async def test_legacy_gate_denial_diagnosis_byte_identical(monkeypatch):
    # INV-6 (§8.10): the gate's denial content equals the validator-owned helper
    # output — C5b changed only the decision predicate, not the diagnosis.
    from app.domain.services.safety.shell_ast_validator import format_denied_content, validate

    _flag(monkeypatch, True)
    default_cwd = Settings(env="test").sandbox_default_cwd
    expected = format_denied_content(
        validate("rm -rf /", effective_cwd=default_cwd), original_command="rm -rf /"
    )
    fn = _build_tool_node_fn()
    res = await fn(
        _make_state("shell_execute", {"command": "rm -rf /"}),
        _make_config(None, None, extra={"policy_snapshot_sink": _SpySink()}),
    )
    assert expected and _command_text(res) == expected  # single denial msg → exact equality [codex planR3 P3]


async def test_legacy_gate_surrogate_exec_dir_does_not_crash(monkeypatch):
    # §8.2 wiring: a lone-surrogate exec_dir flows into C5b's build_command_policy on the
    # always-on decision path. Use the LEGACY gate + a DENIED command: the legacy gate runs
    # validate() before RiskAssessor (N1 invariant `validate < assess`), and a denial
    # `continue`s before RiskAssessor's strict utf-8 digest — so the surrogate reaches
    # build_command_policy (surrogatepass keeps it total) and the tool is never invoked with
    # an unexpected exec_dir kwarg. (PE path can't host this: it strict-hashes exec_dir in
    # RiskAssessor BEFORE the AST gate — react_graph.py:1450 / risk_assessor.py:269.) [codex planR3 P1]
    # Asserting the EXACT denial content proves the surrogate produced a NORMAL policy denial,
    # not a swallowed crash fallback — so build_command_policy provably handled it. [codex planR4 P3]
    from app.domain.services.safety.shell_ast_validator import format_denied_content, validate

    _flag(monkeypatch, True)
    expected = format_denied_content(
        validate("rm -rf /", effective_cwd="/tmp/\ud800x"), original_command="rm -rf /"
    )
    fn = _build_tool_node_fn()
    result = await fn(
        _make_state("shell_execute", {"command": "rm -rf /", "exec_dir": "/tmp/\ud800x"}),
        _make_config(None, None),  # legacy gate, no sink → only the C5b decision path runs
    )
    assert isinstance(result, Command)
    assert _command_text(result) == expected  # exact normal denial → builder handled the surrogate
