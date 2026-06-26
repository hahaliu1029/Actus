"""C5a Sandbox Policy Compiler — pure two-surface compiler. Observe-only.

PURE DOMAIN: imports only app.domain siblings + stdlib. Never calls
get_settings(); the caller passes a pre-built SandboxSettingsView (§3/§5).
"""
from __future__ import annotations

from app.domain.models.sandbox_policy import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    CommandPolicy,
    ContainerCreateInput,
    ContainerRuntimePolicy,
    DecisionSummary,
    FilesystemPolicy,
    MountView,
    NetworkPolicy,
    PolicyProvenance,
    PolicySubject,
    SandboxPolicySnapshot,
    SandboxSettingsView,
    ToolCallInput,
    compute_policy_hash,
    compute_settings_hash,
    sha256_hexdigest,
)
from app.domain.services.safety.shell_ast_validator import (
    DENY_VALIDATION_CODES,
    MAX_COMMAND_BYTES,
)


def _egress_mode(s: SandboxSettingsView) -> str:
    # R3#3: only the exact string "none" disables egress; None/unset omits the
    # kwarg → Docker default bridge ≠ disabled.
    if s.network == "none":
        return "disabled"
    if s.has_https_proxy or s.has_http_proxy:
        return "proxy_env"
    return "unrestricted"


def _filesystem(s: SandboxSettingsView) -> FilesystemPolicy:
    return FilesystemPolicy(
        default_cwd=s.default_cwd,
        relative_path_anchor=s.relative_path_anchor,
        service_install_dir=s.service_install_dir,
        absolute_path_mode="pass_through",
        configured_read_only_mount_targets=(
            [s.memory_mount_target] if s.memory_mount_enabled else []
        ),
        protected_write_roots=[s.service_install_dir],
    )


def _network(s: SandboxSettingsView) -> NetworkPolicy:
    return NetworkPolicy(
        docker_network=s.network,
        egress_mode=_egress_mode(s),
        has_https_proxy=s.has_https_proxy,
        has_http_proxy=s.has_http_proxy,
        has_no_proxy=s.has_no_proxy,
        no_proxy_digest=s.no_proxy_digest,
    )


class SandboxPolicyCompiler:
    """Pure, stateless. Renders the CURRENT effective sandbox posture into a
    versioned SandboxPolicySnapshot. observe-only: never mutates anything."""

    def compile_container_create(self, inp: ContainerCreateInput) -> SandboxPolicySnapshot:
        s = inp.settings
        if s.external_address:
            # R4#1: external/pre-existing sandbox — orchestrator never built a
            # container_config, so image/mem/cap/mounts are UNKNOWN → None/[].
            container = ContainerRuntimePolicy(
                capture_kind="configured", creation_mode="external_address",
                image=None, mem_limit=None, run_as_user=None,
                read_only_rootfs=False, cap_drop=[], security_opt=[],
                pids_limit=None, mounts=[],
            )
        else:
            mounts = (
                [MountView(target=s.memory_mount_target, source_kind="memory_bind",
                           read_only=True)]
                if s.memory_mount_enabled else []
            )
            container = ContainerRuntimePolicy(
                capture_kind="configured", creation_mode="docker_run",
                image=s.image, mem_limit=s.mem_limit, run_as_user=None,
                read_only_rootfs=False, cap_drop=[], security_opt=[],
                pids_limit=None, mounts=mounts,
            )
        subject = PolicySubject(
            session_id=inp.session_id, sandbox_id=inp.sandbox_id,
            sandbox_generation=inp.sandbox_generation,
            worker_type=inp.worker_type, depth=inp.depth,
        )
        provenance = PolicyProvenance(
            compiler_version=COMPILER_VERSION, input_sources=["settings"],
            settings_hash=compute_settings_hash(s), tool_call_digest=None,
        )
        snap = SandboxPolicySnapshot(
            schema_version=SCHEMA_VERSION, policy_hash="",
            enforcement_mode="observe_only", surface="container_create",
            subject=subject, provenance=provenance, decision=None,
            filesystem=_filesystem(s), command=None, network=_network(s),
            container=container,
        )
        return snap.model_copy(update={"policy_hash": compute_policy_hash(snap)})

    def compile_tool_call(self, inp: ToolCallInput) -> SandboxPolicySnapshot:
        s = inp.settings
        v = inp.validation
        command = CommandPolicy(
            validator="shell_ast_validator",
            max_command_bytes=MAX_COMMAND_BYTES,
            blocked_validation_codes=list(DENY_VALIDATION_CODES),
            effective_cwd_digest=sha256_hexdigest(v.effective_cwd),
            is_default_cwd=inp.is_default_cwd,
        )
        decision = DecisionSummary(
            decision_source="shell_ast_validator",
            verdict="ok" if v.allowed else "denied",
            reason_code=v.code,
        )
        subject = PolicySubject(
            session_id=inp.session_id, sandbox_id=inp.sandbox_id,
            sandbox_generation=inp.sandbox_generation,
            worker_type=inp.worker_type, depth=inp.depth,
            tool_call_id=inp.tool_call_id, tool_name=inp.tool_name,
            tool_source=inp.tool_source,
        )
        provenance = PolicyProvenance(
            compiler_version=COMPILER_VERSION,
            input_sources=["settings", "tool_source", "ast_validation_result"],
            settings_hash=compute_settings_hash(s),
            tool_call_digest=sha256_hexdigest(inp.command),
        )
        snap = SandboxPolicySnapshot(
            schema_version=SCHEMA_VERSION, policy_hash="",
            enforcement_mode="observe_only", surface="tool_call",
            subject=subject, provenance=provenance, decision=decision,
            filesystem=_filesystem(s), command=command, network=_network(s),
            container=None,
        )
        return snap.model_copy(update={"policy_hash": compute_policy_hash(snap)})
