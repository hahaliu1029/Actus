from __future__ import annotations

from app.domain.models.sandbox_policy import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    ContainerCreateInput,
    ContainerRuntimePolicy,
    SandboxSettingsView,
    ToolCallInput,
    ValidationResultView,
    compute_policy_hash,
    compute_settings_hash,
    sha256_hexdigest,
)
from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler
from app.domain.services.safety.shell_ast_validator import DENY_VALIDATION_CODES


def _view(**over) -> SandboxSettingsView:
    base = dict(
        external_address=False, image="actus/sandbox:latest", network=None,
        mem_limit="4g", default_cwd="/root", has_https_proxy=False,
        has_http_proxy=False, has_no_proxy=False, no_proxy_digest=None,
        memory_mount_target="/workspace/.memory", memory_mount_enabled=True,
    )
    base.update(over)
    return SandboxSettingsView(**base)


def _cc_input(**over) -> ContainerCreateInput:
    base = dict(
        session_id="s1", user_id="u1", sandbox_id="sbx-1", sandbox_generation=1,
        worker_type="root", depth=0, settings=_view(),
    )
    base.update(over)
    return ContainerCreateInput(**base)


def _tc_input(**over) -> ToolCallInput:
    base = dict(
        session_id="s1", sandbox_id=None, sandbox_generation=0,
        worker_type="unknown", depth=0, tool_call_id="tc-1",
        tool_name="shell_execute", tool_source="native", command="ls -la",
        validation=ValidationResultView(allowed=True, code="ok", effective_cwd="/root"),
        is_default_cwd=True, settings=_view(),
    )
    base.update(over)
    return ToolCallInput(**base)


C = SandboxPolicyCompiler()


# ---- §8.1 FULL-snapshot goldens (lock EVERY §4 field, codex planR2 P2) ----- #
_FS_EXPECTED = {
    "default_cwd": "/root",
    "relative_path_anchor": "/home/ubuntu",
    "service_install_dir": "/sandbox",
    "absolute_path_mode": "pass_through",
    "configured_read_only_mount_targets": ["/workspace/.memory"],
    "protected_write_roots": ["/sandbox"],
}
_NET_EXPECTED = {
    "docker_network": None,
    "egress_mode": "unrestricted",
    "has_https_proxy": False,
    "has_http_proxy": False,
    "has_no_proxy": False,
    "no_proxy_digest": None,
}


def _settings_hash() -> str:
    return compute_settings_hash(_view())


def _assert_full_golden(snap, expected_without_hash):
    dumped = snap.model_dump(mode="json")
    assert dumped.pop("policy_hash") == compute_policy_hash(snap)  # derived — recompute, don't hardcode
    assert dumped == expected_without_hash


def test_container_create_full_golden():
    snap = C.compile_container_create(_cc_input())
    _assert_full_golden(snap, {
        "schema_version": SCHEMA_VERSION,
        "enforcement_mode": "observe_only",
        "surface": "container_create",
        "subject": {
            "session_id": "s1", "sandbox_id": "sbx-1", "sandbox_generation": 1,
            "worker_type": "root", "depth": 0,
            "tool_call_id": None, "tool_name": None, "tool_source": None,
        },
        "provenance": {
            "compiler_version": COMPILER_VERSION, "input_sources": ["settings"],
            "settings_hash": _settings_hash(), "tool_call_digest": None,
        },
        "decision": None,
        "filesystem": _FS_EXPECTED,
        "command": None,
        "network": _NET_EXPECTED,
        "container": {
            "capture_kind": "configured", "creation_mode": "docker_run",
            "image": "actus/sandbox:latest", "mem_limit": "4g", "run_as_user": None,
            "read_only_rootfs": False, "cap_drop": [], "cap_add": [], "security_opt": [],
            "pids_limit": None,
            "mounts": [{"target": "/workspace/.memory", "source_kind": "memory_bind", "read_only": True}],
        },
    })


def _expected_tool_call(*, command, verdict, reason_code):
    return {
        "schema_version": SCHEMA_VERSION,
        "enforcement_mode": "enforce",   # was "observe_only" (C5b §6 step 3)
        "surface": "tool_call",
        "subject": {
            "session_id": "s1", "sandbox_id": None, "sandbox_generation": 0,
            "worker_type": "unknown", "depth": 0,
            "tool_call_id": "tc-1", "tool_name": "shell_execute", "tool_source": "native",
        },
        "provenance": {
            "compiler_version": COMPILER_VERSION,
            "input_sources": ["settings", "tool_source", "ast_validation_result"],
            "settings_hash": _settings_hash(),
            "tool_call_digest": sha256_hexdigest(command),
        },
        "decision": {"decision_source": "shell_ast_validator", "verdict": verdict, "reason_code": reason_code},
        "filesystem": _FS_EXPECTED,
        "command": {
            "validator": "shell_ast_validator", "max_command_bytes": 8192,
            "blocked_validation_codes": list(DENY_VALIDATION_CODES),
            "effective_cwd_digest": sha256_hexdigest("/root"), "is_default_cwd": True,
        },
        "network": _NET_EXPECTED,
        "container": None,
    }


def test_tool_call_ok_full_golden():
    snap = C.compile_tool_call(_tc_input())
    _assert_full_golden(snap, _expected_tool_call(command="ls -la", verdict="ok", reason_code="ok"))


def test_tool_call_denied_full_golden():
    snap = C.compile_tool_call(_tc_input(
        command="rm -rf /",
        validation=ValidationResultView(allowed=False, code="fs_destructive", effective_cwd="/root"),
    ))
    _assert_full_golden(snap, _expected_tool_call(
        command="rm -rf /", verdict="denied", reason_code="fs_destructive"))


def test_external_address_mode_nulls_container_fields():
    snap = C.compile_container_create(_cc_input(settings=_view(external_address=True)))
    assert snap.container.creation_mode == "external_address"
    assert snap.container.image is None and snap.container.mem_limit is None
    assert snap.container.cap_drop == () and snap.container.mounts == ()


def test_docker_run_no_memory_mount_when_disabled():
    snap = C.compile_container_create(_cc_input(settings=_view(memory_mount_enabled=False)))
    assert snap.container.creation_mode == "docker_run"
    assert snap.container.mounts == ()
    assert snap.filesystem.configured_read_only_mount_targets == ()


def test_egress_disabled_when_network_none_string():
    snap = C.compile_container_create(_cc_input(settings=_view(network="none")))
    assert snap.network.egress_mode == "disabled"


def test_egress_proxy_env_when_proxy_present():
    snap = C.compile_container_create(_cc_input(settings=_view(has_https_proxy=True)))
    assert snap.network.egress_mode == "proxy_env"


# ---- §8.6 C5b compiler honesty (shared builder + verdict + enforce flip) ---- #
def test_tool_call_command_uses_shared_builder_surrogate():
    # A surrogate-containing effective_cwd diverges iff the compiler kept the
    # strict inline sha256_hexdigest (which raises on a lone surrogate). Equality
    # locks SEMANTIC equivalence with the shared builder. [§8.6, R2#P3-1]
    from app.domain.services.safety.command_policy_evaluator import build_command_policy

    inp = _tc_input(
        validation=ValidationResultView(allowed=True, code="ok", effective_cwd="/tmp/\ud800x"),
    )
    snap = C.compile_tool_call(inp)
    assert snap.command == build_command_policy(
        effective_cwd=inp.validation.effective_cwd,
        is_default_cwd=inp.is_default_cwd,
    )


def test_tool_call_verdict_derived_from_evaluator():
    # verdict comes from evaluate_command(code, policy), not from v.allowed. [§8.6, codex Q6]
    from app.domain.services.safety.command_policy_evaluator import (
        build_command_policy,
        evaluate_command,
    )

    for code, expected in (("ok", "ok"), ("fs_destructive", "denied")):
        inp = _tc_input(
            command="x",
            validation=ValidationResultView(
                allowed=(code == "ok"), code=code, effective_cwd="/root"
            ),
        )
        snap = C.compile_tool_call(inp)
        pol = build_command_policy(effective_cwd="/root", is_default_cwd=inp.is_default_cwd)
        derived = "ok" if evaluate_command(validation_code=code, policy=pol).allowed else "denied"
        assert snap.decision.verdict == derived == expected


def test_tool_call_verdict_off_contract_follows_evaluator():
    # OFF-CONTRACT discriminator (codex planR1 P2): allowed=True but code is a deny code.
    # Old `"ok" if v.allowed else "denied"` logic would say "ok"; the evaluator (code-driven)
    # says "denied". Asserting "denied" PROVES the verdict is derived from evaluate_command,
    # not from v.allowed. (Tests may build off-contract ValidationResultViews; spec §0.2.)
    inp = _tc_input(
        command="x",
        validation=ValidationResultView(allowed=True, code="fs_destructive", effective_cwd="/root"),
    )
    snap = C.compile_tool_call(inp)
    assert snap.decision.verdict == "denied"


def test_tool_call_verdict_reads_evaluator_output(monkeypatch):
    # Verdict must read the EVALUATOR's output, not re-derive from v.code/v.allowed. Force the
    # compiler-module evaluate_command to DENY an `ok` input; the verdict must then be "denied".
    # A `v.code == "ok"` (or v.allowed) derivation would yield "ok" → fail. [codex planR7 P2]
    import app.domain.services.safety.sandbox_policy_compiler as compiler_mod
    from app.domain.services.safety.command_policy_evaluator import CommandGateDecision

    monkeypatch.setattr(
        compiler_mod,
        "evaluate_command",
        lambda *, validation_code, policy: CommandGateDecision(
            allowed=False, code=validation_code, blocked_by_policy=True
        ),
        raising=False,
    )
    snap = C.compile_tool_call(_tc_input(
        validation=ValidationResultView(allowed=True, code="ok", effective_cwd="/root"),
    ))
    assert snap.decision.verdict == "denied"  # from the (patched) evaluator, not v.code=="ok"


def test_enforce_flip_is_in_fingerprint():
    # compute_policy_hash includes enforcement_mode (sandbox_policy.py:231), so the
    # enforce snapshot's fingerprint must differ from its observe_only twin. [§8.6]
    snap = C.compile_tool_call(_tc_input())
    assert snap.enforcement_mode == "enforce"
    observe_twin = snap.model_copy(update={"enforcement_mode": "observe_only"})
    assert compute_policy_hash(observe_twin) != compute_policy_hash(snap)


# ---- §4.3 C5c: hardened runtime policy + applied enforce snapshot ---------- #
def test_runtime_policy_unhardened_by_default():
    pol = C.compile_container_runtime_policy(_view())  # runtime_hardening_enabled defaults False
    assert pol.capture_kind == "configured"
    assert pol.creation_mode == "docker_run"
    assert pol.cap_drop == () and pol.security_opt == () and pol.pids_limit is None


def test_runtime_policy_hardened_when_flag_set():
    pol = C.compile_container_runtime_policy(_view(runtime_hardening_enabled=True))
    assert pol.cap_drop == ("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE")
    assert pol.pids_limit == 512
    assert pol.security_opt == ()  # no_new_privileges off → empty


def test_runtime_policy_no_new_privileges_opt_in():
    pol = C.compile_container_runtime_policy(
        _view(runtime_hardening_enabled=True, no_new_privileges_enabled=True))
    assert pol.security_opt == ("no-new-privileges:true",)


def test_runtime_policy_external_address_is_unhardened():
    pol = C.compile_container_runtime_policy(
        _view(external_address=True, runtime_hardening_enabled=True))
    assert pol.creation_mode == "external_address"
    assert pol.cap_drop == () and pol.pids_limit is None and pol.security_opt == ()


def test_compile_container_create_reflects_hardening_in_container():
    snap = C.compile_container_create(_cc_input(settings=_view(runtime_hardening_enabled=True)))
    assert snap.enforcement_mode == "observe_only"  # compile_container_create stays observe
    assert snap.container.cap_drop == ("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE")
    assert snap.container.pids_limit == 512


def test_compile_applied_container_create_is_enforce_and_recomputes_hash():
    applied = ContainerRuntimePolicy(
        capture_kind="applied", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, cap_drop=("NET_RAW",), cap_add=(), security_opt=(),
        pids_limit=512, mounts=(),
    )
    snap = C.compile_applied_container_create(applied, _cc_input())
    assert snap.surface == "container_create"
    assert snap.enforcement_mode == "enforce"
    assert snap.container is applied
    assert snap.container.capture_kind == "applied"
    assert snap.command is None and snap.decision is None  # partition still satisfied
    assert snap.policy_hash == compute_policy_hash(snap)  # recomputed (not "")


# ---- §4 C5d-2: strict cap_drop=ALL hardening tier --------------------------- #
_EXPECTED_STRICT_CAP_ADD = (
    "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",
    "SETUID", "SETGID", "SETPCAP", "SETFCAP", "KILL",
)


def test_runtime_policy_strict_when_both_flags_set():
    pol = C.compile_container_runtime_policy(
        _view(runtime_hardening_enabled=True, strict_caps_enabled=True))
    assert pol.cap_drop == ("ALL",)
    assert pol.cap_add == _EXPECTED_STRICT_CAP_ADD
    assert pol.pids_limit == 512


def test_runtime_policy_strict_off_is_conservative_unchanged():
    # INV-0 tier-2: hardening ON + strict OFF == C5c conservative, byte-for-byte.
    pol = C.compile_container_runtime_policy(
        _view(runtime_hardening_enabled=True, strict_caps_enabled=False))
    assert pol.cap_drop == ("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE")
    assert pol.cap_add == ()
    assert pol.pids_limit == 512


def test_runtime_policy_strict_ignored_without_hardening():
    # The view bit alone (no hardening) → unhardened. The Settings guard normally
    # prevents this combo, but the compiler must be safe in isolation.
    pol = C.compile_container_runtime_policy(
        _view(runtime_hardening_enabled=False, strict_caps_enabled=True))
    assert pol.cap_drop == () and pol.cap_add == () and pol.pids_limit is None


def test_runtime_policy_strict_with_nnp_adds_security_opt():
    pol = C.compile_container_runtime_policy(_view(
        runtime_hardening_enabled=True, strict_caps_enabled=True,
        no_new_privileges_enabled=True))
    assert pol.cap_drop == ("ALL",)
    assert pol.cap_add == _EXPECTED_STRICT_CAP_ADD
    assert pol.security_opt == ("no-new-privileges:true",)


def test_runtime_policy_strict_external_address_still_unhardened():
    pol = C.compile_container_runtime_policy(_view(
        external_address=True, runtime_hardening_enabled=True, strict_caps_enabled=True))
    assert pol.creation_mode == "external_address"
    assert pol.cap_drop == () and pol.cap_add == ()
