"""C5c shared helper: compile the hardened ContainerRuntimePolicy from Settings.

Both production create() paths (bind_new + SkillCreatorService) call this so the
hardening flag's contract — "ON ⇒ fresh sandboxes are hardened" — holds uniformly
(partial coverage would be a footgun). Returns None when the flag is OFF →
create() stays byte-identical (INV-0). The defensive getattr (mirrors the C5a
build_settings_view pattern) tolerates legacy/test Settings namespaces.
"""
from __future__ import annotations

from app.domain.models.sandbox_policy import (
    ContainerRuntimePolicy,
    build_settings_view,
)
from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler


def compile_runtime_policy(
    settings, *, worker_type: str = "root"
) -> ContainerRuntimePolicy | None:
    # C5d-6: ``worker_type`` (keyword-only) drives per-child egress selection in the
    # compiler. Default "root" keeps every existing caller (e.g. SkillCreatorService,
    # F6 root-only) at the unchanged root baseline.
    if not getattr(settings, "sandbox_runtime_hardening_enabled", False):
        return None
    view = build_settings_view(settings)
    return SandboxPolicyCompiler().compile_container_runtime_policy(
        view, worker_type=worker_type
    )
