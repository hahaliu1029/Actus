from __future__ import annotations

from app.domain.models.sandbox_policy import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    ContainerCreateInput,
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
            "read_only_rootfs": False, "cap_drop": [], "security_opt": [],
            "pids_limit": None,
            "mounts": [{"target": "/workspace/.memory", "source_kind": "memory_bind", "read_only": True}],
        },
    })


def _expected_tool_call(*, command, verdict, reason_code):
    return {
        "schema_version": SCHEMA_VERSION,
        "enforcement_mode": "observe_only",
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
