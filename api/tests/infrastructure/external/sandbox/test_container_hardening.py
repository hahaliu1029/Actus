from __future__ import annotations

from app.domain.models.sandbox_policy import ContainerRuntimePolicy, MountView
from app.infrastructure.external.sandbox.container_hardening import (
    build_applied_runtime_policy,
    container_hardening_kwargs,
)


def _policy(**over) -> ContainerRuntimePolicy:
    base = dict(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, cap_drop=(), security_opt=(), pids_limit=None,
        mounts=(),
    )
    base.update(over)
    return ContainerRuntimePolicy(**base)


def test_unhardened_policy_yields_empty_kwargs():
    # INV-0: an unhardened policy must add NOTHING to container_config.
    assert container_hardening_kwargs(_policy()) == {}


def test_hardened_policy_renders_list_kwargs():
    kw = container_hardening_kwargs(_policy(
        cap_drop=("NET_RAW", "MKNOD"),
        security_opt=("no-new-privileges:true",), pids_limit=512))
    assert kw == {
        "cap_drop": ["NET_RAW", "MKNOD"],            # tuple → list (SDK-native)
        "security_opt": ["no-new-privileges:true"],
        "pids_limit": 512,
    }


def test_empty_fields_are_omitted_not_emitted_empty():
    kw = container_hardening_kwargs(_policy(pids_limit=512))
    assert kw == {"pids_limit": 512}
    assert "cap_drop" not in kw and "security_opt" not in kw


def test_applied_policy_reflects_real_kwargs_and_mount():
    cfg = {
        "image": "actus/sandbox:latest", "mem_limit": "4g",
        "cap_drop": ["NET_RAW"], "security_opt": [], "pids_limit": 512,
    }
    applied = build_applied_runtime_policy(
        container_config=cfg, memory_mount=object(),  # non-None Mount stand-in
        memory_mount_target="/workspace/.memory")
    assert applied.capture_kind == "applied" and applied.creation_mode == "docker_run"
    assert applied.cap_drop == ("NET_RAW",)
    assert applied.pids_limit == 512
    assert applied.security_opt == ()
    assert applied.run_as_user is None and applied.read_only_rootfs is False
    assert applied.mounts == (
        MountView(target="/workspace/.memory", source_kind="memory_bind", read_only=True),
    )


def test_applied_policy_no_mount_when_skipped():
    # Honesty contract: mount skipped → applied.mounts == () (the lie option (a) tells).
    applied = build_applied_runtime_policy(
        container_config={"image": "actus/sandbox:latest", "mem_limit": "4g"},
        memory_mount=None, memory_mount_target="/workspace/.memory")
    assert applied.mounts == ()
