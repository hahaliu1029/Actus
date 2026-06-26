from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.domain.models.sandbox_policy import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    CommandPolicy,
    ContainerRuntimePolicy,
    DecisionSummary,
    FilesystemPolicy,
    MountView,
    NetworkPolicy,
    PolicyProvenance,
    PolicySubject,
    SandboxPolicySnapshot,
    SandboxSettingsView,
    build_settings_view,
    compute_policy_hash,
    compute_settings_hash,
    sha256_hexdigest,
)
from app.domain.services.safety.shell_ast_validator import DENY_VALIDATION_CODES


# ---- builders -------------------------------------------------------------- #
def _fs() -> FilesystemPolicy:
    return FilesystemPolicy(
        default_cwd="/root",
        relative_path_anchor="/home/ubuntu",
        service_install_dir="/sandbox",
        absolute_path_mode="pass_through",
        configured_read_only_mount_targets=["/workspace/.memory"],
        protected_write_roots=["/sandbox"],
    )


def _net(no_proxy_digest: str | None = None) -> NetworkPolicy:
    return NetworkPolicy(
        docker_network=None,
        egress_mode="unrestricted",
        has_https_proxy=False,
        has_http_proxy=False,
        has_no_proxy=no_proxy_digest is not None,
        no_proxy_digest=no_proxy_digest,
    )


def _container() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured",
        creation_mode="docker_run",
        image="actus/sandbox:latest",
        mem_limit="4g",
        run_as_user=None,
        read_only_rootfs=False,
        cap_drop=[],
        security_opt=[],
        pids_limit=None,
        mounts=[MountView(target="/workspace/.memory", source_kind="memory_bind", read_only=True)],
    )


def _command(blocked: list[str] | None = None) -> CommandPolicy:
    return CommandPolicy(
        validator="shell_ast_validator",
        max_command_bytes=8192,
        blocked_validation_codes=list(DENY_VALIDATION_CODES) if blocked is None else blocked,
        effective_cwd_digest=sha256_hexdigest("/root"),
        is_default_cwd=True,
    )


def _container_snapshot(**over) -> SandboxPolicySnapshot:
    base = dict(
        schema_version=SCHEMA_VERSION,
        policy_hash="placeholder",
        enforcement_mode="observe_only",
        surface="container_create",
        subject=PolicySubject(
            session_id="s1", sandbox_id="sbx-1", sandbox_generation=1,
            worker_type="root", depth=0,
        ),
        provenance=PolicyProvenance(
            compiler_version=COMPILER_VERSION, input_sources=["settings"],
            settings_hash="h", tool_call_digest=None,
        ),
        decision=None, filesystem=_fs(), command=None, network=_net(), container=_container(),
    )
    base.update(over)
    return SandboxPolicySnapshot(**base)


def _tool_call_snapshot(**over) -> SandboxPolicySnapshot:
    base = dict(
        schema_version=SCHEMA_VERSION,
        policy_hash="placeholder",
        enforcement_mode="observe_only",
        surface="tool_call",
        subject=PolicySubject(
            session_id="s1", sandbox_id=None, sandbox_generation=0,
            worker_type="unknown", depth=0,
            tool_call_id="tc-1", tool_name="shell_execute", tool_source="native",
        ),
        provenance=PolicyProvenance(
            compiler_version=COMPILER_VERSION,
            input_sources=["settings", "tool_source", "ast_validation_result"],
            settings_hash="h", tool_call_digest=sha256_hexdigest("ls"),
        ),
        decision=DecisionSummary(decision_source="shell_ast_validator", verdict="ok", reason_code="ok"),
        filesystem=_fs(), command=_command(), network=_net(), container=None,
    )
    base.update(over)
    return SandboxPolicySnapshot(**base)


# ---- INV-4 surface partition ---------------------------------------------- #
def test_container_create_snapshot_valid():
    snap = _container_snapshot()
    assert snap.surface == "container_create"
    assert snap.command is None and snap.decision is None and snap.container is not None


def test_tool_call_snapshot_valid():
    snap = _tool_call_snapshot()
    assert snap.surface == "tool_call"
    assert snap.command is not None and snap.decision is not None and snap.container is None


@pytest.mark.parametrize("missing", ["command", "decision"])
def test_tool_call_missing_required_raises(missing):
    with pytest.raises(ValidationError):
        _tool_call_snapshot(**{missing: None})


def test_tool_call_with_container_raises():
    with pytest.raises(ValidationError):
        _tool_call_snapshot(container=_container())


@pytest.mark.parametrize("field", ["command", "decision"])
def test_container_create_with_toolcall_field_raises(field):
    payload = {"command": _command(), "decision": DecisionSummary(
        decision_source="shell_ast_validator", verdict="ok", reason_code="ok")}
    with pytest.raises(ValidationError):
        _container_snapshot(**{field: payload[field]})


def test_container_create_without_container_raises():
    with pytest.raises(ValidationError):
        _container_snapshot(container=None)


def test_container_create_with_toolcall_subject_field_raises():
    bad_subject = PolicySubject(
        session_id="s1", sandbox_id="sbx-1", sandbox_generation=1,
        worker_type="root", depth=0, tool_name="shell_execute",
    )
    with pytest.raises(ValidationError):
        _container_snapshot(subject=bad_subject)


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        PolicySubject(
            session_id="s1", sandbox_id=None, sandbox_generation=0,
            worker_type="root", depth=0, bogus="x",
        )


# ---- INV-4 EXHAUSTIVE partition (§8.3 — both surfaces × every field, codex planR2 P2) ---- #
def _mutate(payload: dict, path: str, value):
    d = payload
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return payload


_TOOLCALL_REQUIRED = [
    "command", "decision", "subject.tool_call_id", "subject.tool_name",
    "subject.tool_source", "provenance.tool_call_digest",
]


@pytest.mark.parametrize("path", _TOOLCALL_REQUIRED)
def test_tool_call_dropping_any_required_field_raises(path):
    payload = _tool_call_snapshot().model_dump()  # nested dicts; Pydantic re-validates on reconstruct
    _mutate(payload, path, None)
    with pytest.raises(ValidationError):
        SandboxPolicySnapshot(**payload)


def test_tool_call_adding_container_raises():
    payload = _tool_call_snapshot().model_dump()
    payload["container"] = _container().model_dump()
    with pytest.raises(ValidationError):
        SandboxPolicySnapshot(**payload)


_CONTAINER_FORBIDDEN = {
    "command": lambda: _command().model_dump(),
    "decision": lambda: {"decision_source": "shell_ast_validator", "verdict": "ok", "reason_code": "ok"},
    "subject.tool_call_id": lambda: "tc-x",
    "subject.tool_name": lambda: "shell_execute",
    "subject.tool_source": lambda: "native",
    "provenance.tool_call_digest": lambda: sha256_hexdigest("x"),
}


@pytest.mark.parametrize("path", list(_CONTAINER_FORBIDDEN))
def test_container_create_adding_any_forbidden_field_raises(path):
    payload = _container_snapshot().model_dump()
    _mutate(payload, path, _CONTAINER_FORBIDDEN[path]())
    with pytest.raises(ValidationError):
        SandboxPolicySnapshot(**payload)


def test_container_create_dropping_container_raises():
    payload = _container_snapshot().model_dump()
    payload["container"] = None
    with pytest.raises(ValidationError):
        SandboxPolicySnapshot(**payload)


# ---- INV-2 hash stability -------------------------------------------------- #
def test_hash_same_ruleset_same_hash_regardless_of_subject_and_cwd():
    a = _tool_call_snapshot()
    b = _tool_call_snapshot(
        subject=PolicySubject(
            session_id="DIFFERENT", sandbox_id="x", sandbox_generation=99,
            worker_type="subagent", depth=3,
            tool_call_id="tc-999", tool_name="shell_execute", tool_source="mcp",
        ),
        command=CommandPolicy(
            validator="shell_ast_validator", max_command_bytes=8192,
            blocked_validation_codes=list(DENY_VALIDATION_CODES),
            effective_cwd_digest=sha256_hexdigest("/some/OTHER/cwd"),  # different cwd
            is_default_cwd=False,                                      # different flag
        ),
        provenance=PolicyProvenance(
            compiler_version=COMPILER_VERSION,
            input_sources=["settings", "tool_source", "ast_validation_result"],
            settings_hash="h", tool_call_digest=sha256_hexdigest("rm -rf /"),
        ),
    )
    assert compute_policy_hash(a) == compute_policy_hash(b)


def test_hash_differs_when_ruleset_field_changes():
    a = _tool_call_snapshot()
    b = _tool_call_snapshot(command=_command(blocked=["fs_destructive"]))  # different deny set
    assert compute_policy_hash(a) != compute_policy_hash(b)


def test_hash_differs_when_no_proxy_digest_changes():
    a = _container_snapshot(network=_net(no_proxy_digest=None))
    b = _container_snapshot(network=_net(no_proxy_digest=sha256_hexdigest("corp.internal")))
    assert compute_policy_hash(a) != compute_policy_hash(b)


def test_hash_differs_across_surfaces():
    assert compute_policy_hash(_container_snapshot()) != compute_policy_hash(_tool_call_snapshot())


def test_hash_is_hex_sha256():
    h = compute_policy_hash(_container_snapshot())
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# ---- DENY_VALIDATION_CODES source of truth (INV-5) ------------------------- #
def test_deny_codes_exclude_ok_and_include_all_denials():
    assert "ok" not in DENY_VALIDATION_CODES
    assert set(DENY_VALIDATION_CODES) == {
        "fs_destructive", "process_control", "network_exfil", "system_admin",
        "cwd_boundary", "parse_failed", "oversized_command",
    }


# ---- build_settings_view --------------------------------------------------- #
class _FakeSettings:
    sandbox_address = None
    sandbox_image = "actus/sandbox:latest"
    sandbox_network = None
    sandbox_mem_limit = "4g"
    sandbox_default_cwd = "/root"
    sandbox_https_proxy = None
    sandbox_http_proxy = None
    sandbox_no_proxy = "corp.internal"
    sandbox_memory_mount_target = "/workspace/.memory"
    sandbox_memory_mount_enabled = True


def test_build_settings_view_digests_no_proxy_and_reads_attrs():
    view = build_settings_view(_FakeSettings())
    assert view.external_address is False
    assert view.image == "actus/sandbox:latest"
    assert view.mem_limit == "4g"
    assert view.has_no_proxy is True
    assert view.no_proxy_digest == sha256_hexdigest("corp.internal")
    assert "corp.internal" not in json.dumps(view.model_dump(mode="json"))
    assert view.memory_mount_enabled is True
    assert view.relative_path_anchor == "/home/ubuntu"
    assert view.service_install_dir == "/sandbox"


def test_settings_hash_is_deterministic_hex():
    v = build_settings_view(_FakeSettings())
    assert compute_settings_hash(v) == compute_settings_hash(v)
    assert len(compute_settings_hash(v)) == 64
