from __future__ import annotations

import docker
import pytest

from app.domain.models.sandbox_policy import ContainerRuntimePolicy, MountView
from app.infrastructure.external.sandbox.container_hardening import (
    SandboxHardeningConfigError,
    build_applied_runtime_policy,
    container_hardening_kwargs,
    validate_hardening_config,
    verify_egress_network_internal,
)


def _policy(**over) -> ContainerRuntimePolicy:
    base = dict(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, egress_network=None,
        cap_drop=(), cap_add=(), security_opt=(), pids_limit=None,
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


def test_strict_policy_emits_cap_add_and_drop_all():
    kw = container_hardening_kwargs(_policy(
        cap_drop=("ALL",), cap_add=("CHOWN", "SETUID"), pids_limit=512))
    assert kw == {
        "cap_drop": ["ALL"],
        "cap_add": ["CHOWN", "SETUID"],   # tuple → list (SDK-native)
        "pids_limit": 512,
    }


def test_empty_cap_add_is_omitted():
    kw = container_hardening_kwargs(_policy(cap_drop=("NET_RAW",), pids_limit=512))
    assert "cap_add" not in kw


# ---- C5d-3: run_as_user emission + applied read-back ------------------------ #
def test_run_as_user_emitted_when_set():
    kw = container_hardening_kwargs(_policy(
        run_as_user="1000:1000", cap_drop=("NET_RAW",), pids_limit=512))
    assert kw["user"] == "1000:1000"


def test_run_as_user_omitted_when_none():
    # INV-0: a root (run_as_user=None) policy adds NO `user` key.
    kw = container_hardening_kwargs(_policy(
        run_as_user=None, cap_drop=("NET_RAW",), pids_limit=512))
    assert "user" not in kw


def test_run_as_user_empty_string_is_emitted_not_omitted():
    # codex R1 P2: `is not None` (not truthy) — "" must REACH the validator (which rejects
    # it), never silently downgrade to root by omission.
    kw = container_hardening_kwargs(_policy(run_as_user="", cap_drop=("NET_RAW",)))
    assert kw["user"] == ""


def test_applied_reads_user_back():
    cfg = {"image": "actus/sandbox:latest", "mem_limit": "4g", "user": "1000:1000",
           "cap_drop": ["NET_RAW"], "pids_limit": 512}
    applied = build_applied_runtime_policy(
        container_config=cfg, memory_mount=None, memory_mount_target="/workspace/.memory")
    assert applied.run_as_user == "1000:1000"


# ---- C5d-4: read_only + tmpfs emission + applied read-back ------------------ #
def test_read_only_rootfs_emitted_when_set():
    kw = container_hardening_kwargs(_policy(
        read_only_rootfs=True, cap_drop=("NET_RAW",), pids_limit=512))
    assert kw["read_only"] is True
    assert kw["tmpfs"] == {"/tmp": "rw,exec,nosuid,nodev,size=512m"}


def test_read_only_rootfs_omitted_when_false():
    # INV-0: a writable-rootfs (read_only_rootfs=False) policy adds NO read_only/tmpfs key.
    kw = container_hardening_kwargs(_policy(
        read_only_rootfs=False, cap_drop=("NET_RAW",), pids_limit=512))
    assert "read_only" not in kw
    assert "tmpfs" not in kw


def test_read_only_rootfs_tmpfs_is_a_fresh_copy():
    # The emitted tmpfs must be a fresh dict, never the shared module constant (a caller mutation
    # of container_config["tmpfs"] must not corrupt _READONLY_TMPFS for the next container).
    from app.infrastructure.external.sandbox.container_hardening import _READONLY_TMPFS
    kw = container_hardening_kwargs(_policy(read_only_rootfs=True))
    assert kw["tmpfs"] == _READONLY_TMPFS
    assert kw["tmpfs"] is not _READONLY_TMPFS


def test_applied_reads_read_only_back():
    cfg = {"image": "actus/sandbox:latest", "mem_limit": "4g", "read_only": True,
           "tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=512m"},
           "cap_drop": ["NET_RAW"], "pids_limit": 512}
    applied = build_applied_runtime_policy(
        container_config=cfg, memory_mount=None, memory_mount_target="/workspace/.memory")
    assert applied.read_only_rootfs is True


def test_read_only_round_trip_translator_to_applied():
    # End-to-end: read-only policy → translator emits read_only+tmpfs into container_config →
    # build_applied carries read_only_rootfs=True back (the honest applied snapshot).
    policy = _policy(read_only_rootfs=True, cap_drop=("NET_RAW",), pids_limit=512)
    cfg = {"image": "actus/sandbox:latest", "mem_limit": "4g"}
    cfg.update(container_hardening_kwargs(policy))
    assert cfg["read_only"] is True
    assert cfg["tmpfs"] == {"/tmp": "rw,exec,nosuid,nodev,size=512m"}
    applied = build_applied_runtime_policy(
        container_config=cfg, memory_mount=None, memory_mount_target="/workspace/.memory")
    assert applied.read_only_rootfs is True


# ---- C5d-5/6: validator network rejections (always-on, hardened path) ------ #
@pytest.mark.parametrize("bad", [
    {"network_mode": "host"},
    {"network_mode": "none"},
    {"network_mode": "container:abc"},
    {"network_disabled": True},
    {"networking_config": {"EndpointsConfig": {}}},
    {"ports": {"8080/tcp": 8080}},
    {"publish_all_ports": True},
])
def test_validator_rejects_host_exposure_and_network_mode_kwargs(bad):
    # §7.1: Actus never sets these on a hardened sandbox → reject deny-by-default. A conservative
    # config (cap_drop+pids) plus the bad kwarg must fail closed.
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512, **bad}
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config(cfg)


def test_validator_conservative_config_without_network_still_passes():
    # No false positive: a vetted conservative config with NO network kwarg passes (expected None).
    validate_hardening_config({"cap_drop": ["NET_RAW"], "pids_limit": 512})


# ---- §7.2: expected_egress_network name confinement ------------------------ #
def test_validator_accepts_matching_egress_network():
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512, "network": "actus-sandbox-internal"}
    validate_hardening_config(cfg, expected_egress_network="actus-sandbox-internal")  # no raise


def test_validator_rejects_egress_network_mismatch():
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512, "network": "actus-net"}
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config(cfg, expected_egress_network="actus-sandbox-internal")


def test_validator_rejects_missing_network_when_egress_expected():
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512}  # no network at all
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config(cfg, expected_egress_network="actus-sandbox-internal")


def test_validator_rejects_empty_expected_egress_network():
    # R5 P2: a forged/buggy empty expected network must fail closed BEFORE the `==` check
    # (else "" == "" would accept an empty network).
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512, "network": ""}
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config(cfg, expected_egress_network="")


def test_validator_expected_none_skips_network_name_check():
    # expected_egress_network=None (egress OFF) → the name-confinement branch is skipped even if a
    # `network` kwarg is present (the base line-231 network on the OFF path is legitimate).
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512, "network": "actus-net"}
    validate_hardening_config(cfg, expected_egress_network=None)  # no raise


# ---- §8 preflight: verify_egress_network_internal -------------------------- #
class _FakeNet:
    def __init__(self, attrs):
        self.attrs = attrs


class _FakeNetworks:
    def __init__(self, *, net=None, raise_not_found=False):
        self._net = net
        self._raise = raise_not_found

    def get(self, name):
        if self._raise:
            raise docker.errors.NotFound(f"no such network {name}")
        return self._net


class _FakeDockerClient:
    def __init__(self, *, net=None, raise_not_found=False):
        self.networks = _FakeNetworks(net=net, raise_not_found=raise_not_found)


def test_preflight_accepts_internal_true_network():
    client = _FakeDockerClient(net=_FakeNet({"Internal": True}))
    verify_egress_network_internal(client, "actus-sandbox-internal")  # no raise


@pytest.mark.parametrize("attrs", [
    {"Internal": False},
    {"Internal": None},
    {},               # missing key
    {"Internal": "true"},  # truthy non-bool → strict `is not True` rejects
])
def test_preflight_rejects_non_internal_network(attrs):
    client = _FakeDockerClient(net=_FakeNet(attrs))
    with pytest.raises(SandboxHardeningConfigError):
        verify_egress_network_internal(client, "actus-sandbox-internal")


def test_preflight_rejects_missing_network():
    client = _FakeDockerClient(raise_not_found=True)
    with pytest.raises(SandboxHardeningConfigError):
        verify_egress_network_internal(client, "actus-sandbox-internal")


# ---- C5d-5/6: translator emits network + applied echoes egress_network ----- #
def test_translator_emits_network_when_egress_network_set():
    kw = container_hardening_kwargs(_policy(
        egress_network="actus-sandbox-internal", cap_drop=("NET_RAW",), pids_limit=512))
    assert kw["network"] == "actus-sandbox-internal"


def test_translator_omits_network_when_egress_none():
    # INV-0: egress_network=None → NO `network` key (the base line-231 network stands).
    kw = container_hardening_kwargs(_policy(
        egress_network=None, cap_drop=("NET_RAW",), pids_limit=512))
    assert "network" not in kw


def test_translator_emits_empty_network_not_silently_dropped():
    # `is not None` (not truthy): an empty "" must REACH the validator (which rejects it), never
    # silently leave the container on the routable base network.
    kw = container_hardening_kwargs(_policy(egress_network="", cap_drop=("NET_RAW",)))
    assert kw["network"] == ""


def test_applied_echoes_egress_network():
    applied = build_applied_runtime_policy(
        container_config={"image": "actus/sandbox:latest", "mem_limit": "4g",
                          "network": "actus-sandbox-internal"},
        memory_mount=None, memory_mount_target="/workspace/.memory",
        egress_network="actus-sandbox-internal")
    assert applied.egress_network == "actus-sandbox-internal"


def test_applied_egress_network_none_when_off():
    applied = build_applied_runtime_policy(
        container_config={"image": "actus/sandbox:latest", "mem_limit": "4g"},
        memory_mount=None, memory_mount_target="/workspace/.memory", egress_network=None)
    assert applied.egress_network is None
