"""C5a/C5b Sandbox Policy Compiler — pure two-surface compiler. tool_call snapshots
report enforcement_mode="enforce" (C5b); container_create stays observe_only.

PURE DOMAIN: imports only app.domain siblings + stdlib. Never calls
get_settings(); the caller passes a pre-built SandboxSettingsView (§3/§5).
"""
from __future__ import annotations

from app.domain.models.sandbox_policy import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
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
from app.domain.services.safety.command_policy_evaluator import (
    build_command_policy,
    evaluate_command,
)

# ── C5c conservative hardening baseline (spec §4.2; each cap justified §0.6) ── #
# cap_drop: raw sockets / device-node creation / audit writes / privileged-port
# bind — none used by the workload (Chrome runs --no-sandbox; ports all > 1024).
# NOT cap_drop=["ALL"] (that needs add-back → C5d). pids_limit: fork/thread guard
# above steady-state (Chrome + supervisord minprocs=200). no-new-privileges is a
# SEPARATE opt-in (breaks sudo) → only added when its own flag is on.
_BASELINE_CAP_DROP: tuple[str, ...] = ("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE")
_BASELINE_PIDS_LIMIT: int = 512
_NO_NEW_PRIVILEGES_OPT: str = "no-new-privileges:true"


# ── C5d-2 strict hardening tier (cap_drop=ALL + vetted minimal add-back) ───── #
# Each cap is justified by the all-root + runtime-apt/pip/sudo workload (spec
# §0.5/§0.6); SYS_CHROOT is the one extra drop vs the C5c conservative profile.
# SETFCAP + KILL are conservative-rationale-NOT-smoke-validated (safe over-grant
# of a non-escape cap; prime tightening candidates — spec §9/§12). Emitted only
# when BOTH runtime_hardening_enabled AND strict_caps_enabled are on.
_STRICT_CAP_DROP: tuple[str, ...] = ("ALL",)
_STRICT_CAP_ADD: tuple[str, ...] = (
    "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",   # apt/pip ownership / mode / setuid-bit
    "SETUID", "SETGID", "SETPCAP",                  # sudo NOPASSWD privilege transition
    "SETFCAP",                                      # apt file-caps (conservative, not smoke-validated)
    "KILL",                                         # cross-uid signal (conservative, not smoke-validated)
)  # 9 caps. CI-gated; parity-10 fallback = append "SYS_CHROOT" (spec §9 / §4.2).


# ── C5d-3 non-root tier (runtime --user) ──────────────────────────────────── #
# The ONE vetted non-root identity. uid:gid 1000:1000 = the `ubuntu` user the image
# pins (Dockerfile useradd -u 1000 -g 1000). Numeric (not "ubuntu") so the kernel
# enforces it without a passwd lookup; gid pinned (not bare "1000") so the process is in
# a non-root primary group. Emitted only when hardening AND run_as_user_enabled.
_RUN_AS_USER: str = "1000:1000"


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
    versioned SandboxPolicySnapshot. The compiler never mutates anything; the
    snapshot's enforcement_mode is "enforce" for tool_call (C5b), "observe_only"
    for container_create."""

    def compile_container_runtime_policy(
        self, s: SandboxSettingsView
    ) -> ContainerRuntimePolicy:
        """C5c: the INTENDED ContainerRuntimePolicy (capture_kind='configured').
        Pure. docker_run is hardened iff s.runtime_hardening_enabled;
        external_address is always unhardened (no container to harden)."""
        if s.external_address:
            return ContainerRuntimePolicy(
                capture_kind="configured", creation_mode="external_address",
                image=None, mem_limit=None, run_as_user=None,
                read_only_rootfs=False, cap_drop=(), cap_add=(), security_opt=(),
                pids_limit=None, mounts=(),
            )
        mounts = (
            (MountView(target=s.memory_mount_target, source_kind="memory_bind",
                       read_only=True),)
            if s.memory_mount_enabled else ()
        )
        if s.runtime_hardening_enabled:
            if s.strict_caps_enabled:
                cap_drop = _STRICT_CAP_DROP
                cap_add = _STRICT_CAP_ADD
            else:
                cap_drop = _BASELINE_CAP_DROP
                cap_add = ()
            security_opt = (
                (_NO_NEW_PRIVILEGES_OPT,) if s.no_new_privileges_enabled else ()
            )
            pids_limit = _BASELINE_PIDS_LIMIT
            run_as_user = _RUN_AS_USER if s.run_as_user_enabled else None
            read_only_rootfs = True if s.read_only_rootfs_enabled else False
        else:
            cap_drop = ()
            cap_add = ()
            security_opt = ()
            pids_limit = None
            run_as_user = None
            read_only_rootfs = False
        return ContainerRuntimePolicy(
            capture_kind="configured", creation_mode="docker_run",
            image=s.image, mem_limit=s.mem_limit, run_as_user=run_as_user,
            read_only_rootfs=read_only_rootfs, cap_drop=cap_drop, cap_add=cap_add,
            security_opt=security_opt, pids_limit=pids_limit, mounts=mounts,
        )

    def _container_create_snapshot(
        self, *, inp: ContainerCreateInput, container: ContainerRuntimePolicy,
        enforcement_mode: str,
    ) -> SandboxPolicySnapshot:
        """Shared assembly for the configured (observe_only) and applied (enforce)
        container_create snapshots — identical subject/provenance/filesystem/network,
        differing only in container + enforcement_mode. Recomputes policy_hash."""
        s = inp.settings
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
            enforcement_mode=enforcement_mode, surface="container_create",
            subject=subject, provenance=provenance, decision=None,
            filesystem=_filesystem(s), command=None, network=_network(s),
            container=container,
        )
        return snap.model_copy(update={"policy_hash": compute_policy_hash(snap)})

    def compile_container_create(self, inp: ContainerCreateInput) -> SandboxPolicySnapshot:
        return self._container_create_snapshot(
            inp=inp,
            container=self.compile_container_runtime_policy(inp.settings),
            enforcement_mode="observe_only",
        )

    def compile_applied_container_create(
        self, applied: ContainerRuntimePolicy, inp: ContainerCreateInput
    ) -> SandboxPolicySnapshot:
        """C5c: assemble the enforce snapshot from the APPLIED policy (real kwargs,
        capture_kind='applied' set by _create_task). Reuses subject/provenance/
        filesystem/network from inp; container = the applied policy verbatim."""
        return self._container_create_snapshot(
            inp=inp, container=applied, enforcement_mode="enforce",
        )

    def compile_tool_call(self, inp: ToolCallInput) -> SandboxPolicySnapshot:
        s = inp.settings
        v = inp.validation
        command = build_command_policy(
            effective_cwd=v.effective_cwd,
            is_default_cwd=inp.is_default_cwd,
        )
        _decision = evaluate_command(validation_code=v.code, policy=command)
        decision = DecisionSummary(
            decision_source="shell_ast_validator",
            verdict="ok" if _decision.allowed else "denied",
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
            enforcement_mode="enforce", surface="tool_call",
            subject=subject, provenance=provenance, decision=decision,
            filesystem=_filesystem(s), command=command, network=_network(s),
            container=None,
        )
        return snap.model_copy(update={"policy_hash": compute_policy_hash(snap)})
