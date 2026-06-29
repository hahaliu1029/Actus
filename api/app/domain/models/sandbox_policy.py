"""C5a Sandbox Policy Compiler — frozen policy-snapshot model + fingerprint helpers.

C5b: tool_call snapshots report enforcement_mode="enforce", container_create stays
observe_only (the C5a flag now gates only snapshot EMISSION). PURE DOMAIN: imports no FastAPI/SQLAlchemy/
infrastructure and never calls get_settings(). See
docs/superpowers/specs/2026-06-26-c5a-sandbox-policy-compiler-design.md (§4).
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = "c5.sandbox_policy.v1"
COMPILER_VERSION = "c5d2.1"  # bump on any mapping change (C5d-2: cap_add in container hash)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def sha256_hexdigest(value: str) -> str:
    """Stable sha256 hex of a UTF-8 string. Used for every digest field."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ── identity / provenance / decision ─────────────────────────────────────── #
class PolicySubject(BaseModel):
    model_config = _FROZEN
    session_id: str
    sandbox_id: str | None
    sandbox_generation: int
    worker_type: Literal["root", "subagent", "unknown"]
    depth: int
    tool_call_id: str | None = None  # tool_call only (INV-4)
    tool_name: str | None = None     # tool_call only (INV-4)
    tool_source: str | None = None   # tool_call only (INV-4)


class PolicyProvenance(BaseModel):
    model_config = _FROZEN
    compiler_version: str
    # tuples (not lists): the frozen+hashable snapshot must be DEEPLY immutable so policy_hash can't desync from in-place mutation [codex final-audit P2-1]
    input_sources: tuple[str, ...]
    settings_hash: str
    tool_call_digest: str | None = None  # sha256(command); None for container_create


class DecisionSummary(BaseModel):
    model_config = _FROZEN
    decision_source: Literal["shell_ast_validator"]
    verdict: Literal["ok", "denied"]
    reason_code: str  # ValidationCode value — NO raw command / reason text


# ── policy sub-models ─────────────────────────────────────────────────────── #
class FilesystemPolicy(BaseModel):
    model_config = _FROZEN
    default_cwd: str
    relative_path_anchor: str
    service_install_dir: str
    absolute_path_mode: Literal["pass_through"]
    configured_read_only_mount_targets: tuple[str, ...]
    protected_write_roots: tuple[str, ...]


class CommandPolicy(BaseModel):  # tool_call only
    model_config = _FROZEN
    validator: Literal["shell_ast_validator"]
    max_command_bytes: int
    blocked_validation_codes: tuple[str, ...]
    effective_cwd_digest: str  # sha256(effective_cwd) — NEVER raw (INV-1)
    is_default_cwd: bool        # exec_dir empty → fell back to default cwd


class NetworkPolicy(BaseModel):
    model_config = _FROZEN
    docker_network: str | None
    egress_mode: Literal["unrestricted", "proxy_env", "disabled"]
    has_https_proxy: bool
    has_http_proxy: bool
    has_no_proxy: bool
    no_proxy_digest: str | None = None  # sha256(no_proxy) or None — NO raw value


class MountView(BaseModel):  # sanitized mount descriptor (no host source path)
    model_config = _FROZEN
    target: str
    source_kind: Literal["memory_bind"]
    read_only: bool


class ContainerRuntimePolicy(BaseModel):  # container_create only
    model_config = _FROZEN
    capture_kind: Literal["configured", "applied"]  # "applied" = real _create_task kwargs (C5c)
    creation_mode: Literal["docker_run", "external_address"]
    image: str | None
    mem_limit: str | None
    run_as_user: str | None
    read_only_rootfs: bool
    cap_drop: tuple[str, ...]
    cap_add: tuple[str, ...]  # C5d-2: strict drop-ALL add-back allowlist; () for conservative/unhardened/external
    security_opt: tuple[str, ...]
    pids_limit: int | None
    mounts: tuple[MountView, ...]


# ── caller-built views (keep the compiler pure) ──────────────────────────── #
class SandboxSettingsView(BaseModel):
    model_config = _FROZEN
    external_address: bool
    image: str | None
    network: str | None
    mem_limit: str
    default_cwd: str
    has_https_proxy: bool
    has_http_proxy: bool
    has_no_proxy: bool
    no_proxy_digest: str | None = None
    memory_mount_target: str
    memory_mount_enabled: bool
    relative_path_anchor: str = "/home/ubuntu"
    service_install_dir: str = "/sandbox"
    # C5c: hardening posture bits (default False → INV-0 unhardened). build_settings_view
    # fills these from the two new Settings flags; the compiler hardens the docker_run
    # container iff runtime_hardening_enabled is set.
    runtime_hardening_enabled: bool = False
    no_new_privileges_enabled: bool = False
    # C5d-2: strict-caps tier bit (default False → conservative/unhardened). Layered
    # under runtime_hardening_enabled; the compiler consults it only on the hardened
    # docker_run branch.
    strict_caps_enabled: bool = False


class ValidationResultView(BaseModel):  # 3-field projection of the real ValidationResult
    model_config = _FROZEN
    allowed: bool
    code: str
    effective_cwd: str  # raw on INPUT only; compiler digests it, never stores raw


# ── per-surface compile inputs ────────────────────────────────────────────── #
class ContainerCreateInput(BaseModel):
    model_config = _FROZEN
    session_id: str
    user_id: str | None  # NEVER copied into the snapshot (INV-1)
    sandbox_id: str | None
    sandbox_generation: int
    worker_type: Literal["root", "subagent", "unknown"]
    depth: int
    settings: SandboxSettingsView


class ToolCallInput(BaseModel):
    model_config = _FROZEN
    session_id: str
    sandbox_id: str | None
    sandbox_generation: int
    worker_type: Literal["root", "subagent", "unknown"]
    depth: int
    tool_call_id: str
    tool_name: str
    tool_source: str
    command: str  # raw — hashed by the compiler, NEVER stored verbatim
    validation: ValidationResultView
    is_default_cwd: bool
    settings: SandboxSettingsView


# ── top-level snapshot ────────────────────────────────────────────────────── #
class SandboxPolicySnapshot(BaseModel):
    model_config = _FROZEN
    schema_version: Literal["c5.sandbox_policy.v1"]
    policy_hash: str
    enforcement_mode: Literal["observe_only", "enforce"]
    surface: Literal["container_create", "tool_call"]
    subject: PolicySubject
    provenance: PolicyProvenance
    decision: DecisionSummary | None = None
    filesystem: FilesystemPolicy
    command: CommandPolicy | None = None
    network: NetworkPolicy
    container: ContainerRuntimePolicy | None = None

    @model_validator(mode="after")
    def _surface_partition(self) -> "SandboxPolicySnapshot":
        """INV-4: full per-surface presence partition."""
        if self.surface == "tool_call":
            missing: list[str] = []
            if self.command is None:
                missing.append("command")
            if self.decision is None:
                missing.append("decision")
            if self.subject.tool_call_id is None:
                missing.append("subject.tool_call_id")
            if self.subject.tool_name is None:
                missing.append("subject.tool_name")
            if self.subject.tool_source is None:
                missing.append("subject.tool_source")
            if self.provenance.tool_call_digest is None:
                missing.append("provenance.tool_call_digest")
            if missing:
                raise ValueError(f"surface=tool_call requires set: {missing}")
            if self.container is not None:
                raise ValueError("surface=tool_call forbids container")
        else:  # container_create
            forbidden: list[str] = []
            if self.command is not None:
                forbidden.append("command")
            if self.decision is not None:
                forbidden.append("decision")
            if self.subject.tool_call_id is not None:
                forbidden.append("subject.tool_call_id")
            if self.subject.tool_name is not None:
                forbidden.append("subject.tool_name")
            if self.subject.tool_source is not None:
                forbidden.append("subject.tool_source")
            if self.provenance.tool_call_digest is not None:
                forbidden.append("provenance.tool_call_digest")
            if forbidden:
                raise ValueError(f"surface=container_create forbids: {forbidden}")
            if self.container is None:
                raise ValueError("surface=container_create requires container")
        return self


# ── fingerprint + view-builder helpers ───────────────────────────────────── #
def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_policy_hash(snapshot: SandboxPolicySnapshot) -> str:
    """Stable fingerprint over the STATIC rule-set only (§4.5).

    Excludes subject/provenance AND the per-call-volatile
    CommandPolicy.effective_cwd_digest + is_default_cwd, so identical static
    rule-set ⇒ identical hash regardless of cwd or call identity (INV-2).
    """
    payload = snapshot.model_dump(
        mode="json",
        include={
            "enforcement_mode": True,
            "surface": True,
            "filesystem": True,
            "network": True,
            # C5c: nested include EXCLUDING mounts (per-bind-volatile → INV-2). A
            # tool_call snapshot's container is None → still serializes as null, so
            # tool_call fingerprints are unchanged.
            "container": {
                "capture_kind",
                "creation_mode",
                "image",
                "mem_limit",
                "run_as_user",
                "read_only_rootfs",
                "cap_drop",
                "cap_add",
                "security_opt",
                "pids_limit",
            },
            "command": {"validator", "max_command_bytes", "blocked_validation_codes"},
        },
    )
    return sha256_hexdigest(_canonical_json(payload))


def compute_settings_hash(view: SandboxSettingsView) -> str:
    """sha256 of the canonical SandboxSettingsView JSON (provenance.settings_hash)."""
    return sha256_hexdigest(_canonical_json(view.model_dump(mode="json")))


def build_settings_view(settings) -> SandboxSettingsView:
    """Project a Settings-like object into the frozen view the compiler consumes.

    Duck-typed (reads attributes only) so this module imports nothing from
    core/infra and stays INV-3 pure. Callers (Seam A/B) pass the live Settings.
    """
    no_proxy = getattr(settings, "sandbox_no_proxy", None)
    return SandboxSettingsView(
        external_address=bool(getattr(settings, "sandbox_address", None)),
        image=settings.sandbox_image,
        network=settings.sandbox_network,
        mem_limit=settings.sandbox_mem_limit,
        default_cwd=settings.sandbox_default_cwd,
        has_https_proxy=bool(settings.sandbox_https_proxy),
        has_http_proxy=bool(settings.sandbox_http_proxy),
        has_no_proxy=bool(no_proxy),
        no_proxy_digest=sha256_hexdigest(no_proxy) if no_proxy else None,
        memory_mount_target=settings.sandbox_memory_mount_target,
        memory_mount_enabled=settings.sandbox_memory_mount_enabled,
        # C5c: defensive getattr (mirrors sandbox_no_proxy above) so legacy/test
        # Settings namespaces lacking the flags fall back to the unhardened default.
        runtime_hardening_enabled=getattr(settings, "sandbox_runtime_hardening_enabled", False),
        no_new_privileges_enabled=getattr(settings, "sandbox_no_new_privileges_enabled", False),
        strict_caps_enabled=getattr(settings, "sandbox_strict_caps_enabled", False),
    )
