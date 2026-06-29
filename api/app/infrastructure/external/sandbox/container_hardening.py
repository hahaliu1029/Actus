"""C5c pure-infra translator: ContainerRuntimePolicy → docker run kwargs +
applied-policy projection.

INV-0 heart — an unhardened policy yields {} so the ON-unhardened and OFF paths
produce a byte-identical container_config. Lives in infrastructure (it knows
docker-py kwarg names); the domain compiler stays pure (INV-5).
"""
from __future__ import annotations

from app.domain.models.sandbox_policy import ContainerRuntimePolicy, MountView


def container_hardening_kwargs(policy: ContainerRuntimePolicy) -> dict:
    """Render ONLY the hardening kwargs docker-py consumes (cap_drop/security_opt
    as list[str], pids_limit as int). Empty/None fields are OMITTED → an unhardened
    policy → {} (INV-0). run_as_user/read_only_rootfs are intentionally NOT emitted
    (C5c hardcodes them None/False → C5d)."""
    kwargs: dict = {}
    if policy.cap_drop:
        kwargs["cap_drop"] = list(policy.cap_drop)
    if policy.security_opt:
        kwargs["security_opt"] = list(policy.security_opt)
    if policy.pids_limit is not None:
        kwargs["pids_limit"] = policy.pids_limit
    return kwargs


def build_applied_runtime_policy(
    *, container_config: dict, memory_mount, memory_mount_target: str,
) -> ContainerRuntimePolicy:
    """Build the honest applied ContainerRuntimePolicy from the REAL, fully-assembled
    container_config (read back the merged kwargs) + the real per-bind mount decision.
    capture_kind='applied'. mounts reflect reality: a MountView iff a Mount was created
    (option (b) — never echoes an intended-but-skipped mount)."""
    mounts: tuple[MountView, ...] = ()
    if memory_mount is not None:
        mounts = (
            MountView(target=memory_mount_target, source_kind="memory_bind",
                      read_only=True),
        )
    return ContainerRuntimePolicy(
        capture_kind="applied",
        creation_mode="docker_run",
        image=container_config.get("image"),
        mem_limit=container_config.get("mem_limit"),
        run_as_user=container_config.get("user"),                # None in C5c
        read_only_rootfs=bool(container_config.get("read_only", False)),  # False in C5c
        cap_drop=tuple(container_config.get("cap_drop", ())),
        security_opt=tuple(container_config.get("security_opt", ())),
        pids_limit=container_config.get("pids_limit"),
        mounts=mounts,
    )
