from __future__ import annotations

import pytest

from app.domain.models.sandbox_policy import ContainerRuntimePolicy, MountView
from app.infrastructure.external.sandbox import docker_sandbox as ds_mod
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox


class _FakeContainer:
    attrs = {"NetworkSettings": {"IPAddress": "1.2.3.4"}}

    def reload(self):  # _wait_for_container_ip calls this; harmless here
        ...


class _FakeContainers:
    def __init__(self, sink: dict):
        self._sink = sink

    def run(self, **kwargs):
        self._sink.clear()
        self._sink.update(kwargs)
        return _FakeContainer()


class _FakeClient:
    def __init__(self, sink: dict):
        self.containers = _FakeContainers(sink)

    def close(self):
        ...


@pytest.fixture
def captured_kwargs(monkeypatch):
    sink: dict = {}
    monkeypatch.setattr(DockerSandbox, "_create_docker_client",
                        classmethod(lambda cls: _FakeClient(sink)))
    monkeypatch.setattr(DockerSandbox, "_wait_for_container_ip",
                        classmethod(lambda cls, container, **kw: "1.2.3.4"))
    return sink


def _patch_settings(monkeypatch, **over):
    class _S:
        sandbox_image = "actus/sandbox:latest"
        sandbox_name_prefix = "actus-sandbox"
        sandbox_ttl_minutes = 60
        sandbox_chrome_args = ""
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        container_timezone = "UTC"
        skill_sandbox_bundle_root = "/home/ubuntu/workspace/.skills"
        sandbox_mem_limit = "4g"
        sandbox_network = None
        sandbox_memory_mount_enabled = False  # no mount → deterministic kwargs
        sandbox_memory_mount_target = "/workspace/.memory"
        memory_root_container = "/app/data/memory"
        memory_root_host = "/srv/actus-memory"
    s = _S()
    for k, v in over.items():
        setattr(s, k, v)
    monkeypatch.setattr(ds_mod, "get_settings", lambda: s)
    return s


def _unhardened() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, cap_drop=(), cap_add=(), security_opt=(), pids_limit=None,
        mounts=(),
    )


def _hardened() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False,
        cap_drop=("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"),
        cap_add=(),
        security_opt=(), pids_limit=512, mounts=(),
    )


def test_inv0_off_path_has_no_hardening_keys(captured_kwargs, monkeypatch):
    _patch_settings(monkeypatch)
    DockerSandbox._create_task(user_id=None, runtime_policy=None)
    assert "cap_drop" not in captured_kwargs
    assert "security_opt" not in captured_kwargs
    assert "pids_limit" not in captured_kwargs
    assert captured_kwargs["image"] == "actus/sandbox:latest"
    assert captured_kwargs["mem_limit"] == "4g"
    assert captured_kwargs["detach"] is True and captured_kwargs["remove"] is True


def test_inv0_unhardened_policy_equals_off_path(captured_kwargs, monkeypatch):
    # Proof #2: ON-unhardened kwargs == OFF kwargs (delta is ∅), excluding the uuid name.
    _patch_settings(monkeypatch)
    DockerSandbox._create_task(user_id=None, runtime_policy=None)
    off = {k: v for k, v in captured_kwargs.items() if k != "name"}
    DockerSandbox._create_task(user_id=None, runtime_policy=_unhardened())
    on_unhardened = {k: v for k, v in captured_kwargs.items() if k != "name"}
    assert on_unhardened == off


def test_hardened_policy_merges_kwargs_and_builds_applied(captured_kwargs, monkeypatch):
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=_hardened())
    assert captured_kwargs["cap_drop"] == ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"]
    assert captured_kwargs["pids_limit"] == 512
    assert "security_opt" not in captured_kwargs  # empty → omitted
    applied = sandbox.applied_runtime_policy
    assert applied is not None and applied.capture_kind == "applied"
    assert applied.cap_drop == ("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE")
    assert applied.pids_limit == 512
    assert applied.mounts == ()  # mount disabled → honest empty


def test_off_path_applied_is_none(captured_kwargs, monkeypatch):
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=None)
    assert sandbox.applied_runtime_policy is None


def test_inv0_off_path_matches_captured_baseline(captured_kwargs, monkeypatch):
    # Spec §9.1 proof (a): the OFF-path container_config equals a CAPTURED pre-C5c
    # baseline EXACTLY (excluding the uuid name) — catches an accidental OFF-path
    # change to environment / mem_limit / network / mounts, not just hardening keys.
    _patch_settings(monkeypatch)
    DockerSandbox._create_task(user_id=None, runtime_policy=None)
    captured = {k: v for k, v in captured_kwargs.items() if k != "name"}
    assert captured == {
        "image": "actus/sandbox:latest",
        "detach": True,
        "remove": True,
        "environment": {
            "SERVICE_TIMEOUT_MINUTES": 60,
            "CHROME_ARGS": "",
            "HTTPS_PROXY": None,
            "HTTP_PROXY": None,
            "NO_PROXY": None,
            "TZ": "UTC",
            "SKILL_SANDBOX_BUNDLE_ROOT": "/home/ubuntu/workspace/.skills",
        },
        "mem_limit": "4g",
    }


def test_applied_reflects_real_mount_when_mountable(captured_kwargs, monkeypatch, tmp_path):
    # Spec §9.3: _create_task with a mountable user_id → the REAL _build_memory_mount
    # Mount flows end-to-end into applied.mounts (option (b)), not just the translator stub.
    host_root = tmp_path / "host"
    cont_root = tmp_path / "cont"
    host_root.mkdir()
    cont_root.mkdir()
    _patch_settings(
        monkeypatch, sandbox_memory_mount_enabled=True,
        memory_root_host=str(host_root), memory_root_container=str(cont_root),
    )
    sandbox = DockerSandbox._create_task(user_id="u1", runtime_policy=_hardened())
    assert "mounts" in captured_kwargs  # a real Mount was added to container_config
    assert sandbox.applied_runtime_policy.mounts == (
        MountView(target="/workspace/.memory", source_kind="memory_bind", read_only=True),
    )


def test_applied_no_mount_when_user_id_non_whitelist(captured_kwargs, monkeypatch, tmp_path):
    # Spec §9.3: a non-whitelist user_id → _build_memory_mount returns None → the honest
    # applied.mounts == () (the lie option (a) would have told).
    host_root = tmp_path / "host"
    cont_root = tmp_path / "cont"
    host_root.mkdir()
    cont_root.mkdir()
    _patch_settings(
        monkeypatch, sandbox_memory_mount_enabled=True,
        memory_root_host=str(host_root), memory_root_container=str(cont_root),
    )
    sandbox = DockerSandbox._create_task(user_id="bad id!", runtime_policy=_hardened())
    assert "mounts" not in captured_kwargs
    assert sandbox.applied_runtime_policy.mounts == ()


def test_inv0_tier2_conservative_create_matches_c5c_container_config(captured_kwargs, monkeypatch):
    # INV-0 tier-2: hardening ON + strict OFF → the FULL container_config equals the OFF-path
    # baseline PLUS EXACTLY the C5c conservative hardening kwargs (cap_drop=[4] + pids_limit=512),
    # and NOTHING else — no cap_add, no security_opt, no other new kwarg. FULL-dict equality (not a
    # field subset, R3 P1) so a future conservative-path emission of any extra kwarg (user /
    # read_only / tmpfs / …) breaks it. Built through the REAL compile chain (settings →
    # compile_runtime_policy → _create_task), mirroring the tier-1 test_inv0_unhardened_policy_equals_off_path.
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _Hardened:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/root"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = False   # match _patch_settings base → deterministic, mount-free
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = False
        sandbox_strict_caps_enabled = False

    _patch_settings(monkeypatch)  # base config (image/env/mem); mount disabled → deterministic kwargs
    # 1) capture the OFF-path baseline (runtime_policy=None) — the pre-C5c container_config.
    DockerSandbox._create_task(user_id=None, runtime_policy=None)
    off = {k: v for k, v in captured_kwargs.items() if k != "name"}
    # 2) run the conservative (strict-OFF) compiled policy through the SAME path.
    DockerSandbox._create_task(user_id=None, runtime_policy=compile_runtime_policy(_Hardened()))
    conservative = {k: v for k, v in captured_kwargs.items() if k != "name"}
    # 3) byte-identity: conservative == OFF baseline + EXACTLY the C5c hardening kwargs, nothing more.
    assert conservative == off | {
        "cap_drop": ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"],
        "pids_limit": 512,
    }


def _strict() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, cap_drop=("ALL",),
        cap_add=("CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",
                 "SETUID", "SETGID", "SETPCAP", "SETFCAP", "KILL"),
        security_opt=(), pids_limit=512, mounts=(),
    )


def test_strict_policy_threads_cap_add_into_kwargs_and_applied(captured_kwargs, monkeypatch):
    # End-to-end: strict policy → translator emits cap_add=[9] + cap_drop=["ALL"] →
    # _create_task merges → build_applied carries cap_add into the applied snapshot.
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=_strict())
    assert captured_kwargs["cap_drop"] == ["ALL"]
    assert captured_kwargs["cap_add"] == [
        "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",
        "SETUID", "SETGID", "SETPCAP", "SETFCAP", "KILL",
    ]
    applied = sandbox.applied_runtime_policy
    assert applied is not None and applied.capture_kind == "applied"
    assert applied.cap_drop == ("ALL",)
    assert applied.cap_add == (
        "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",
        "SETUID", "SETGID", "SETPCAP", "SETFCAP", "KILL",
    )


def test_create_task_rejects_out_of_ceiling_cap_add_with_typed_error(captured_kwargs, monkeypatch):
    # The validator runs after the kwargs merge, before containers.run; the typed
    # SandboxHardeningConfigError must propagate INTACT (not re-wrapped by the broad
    # `except Exception`). The fake container.run is never reached (fail-closed).
    from app.infrastructure.external.sandbox.container_hardening import (
        SandboxHardeningConfigError,
    )
    _patch_settings(monkeypatch)
    bad = ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, cap_drop=("ALL",), cap_add=("SYS_ADMIN",),
        security_opt=(), pids_limit=512, mounts=(),
    )
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=bad)


def test_create_task_strict_config_passes_validator(captured_kwargs, monkeypatch):
    # The real strict config is ACCEPTED → containers.run proceeds, applied carries it.
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=_strict())
    assert sandbox.applied_runtime_policy.cap_drop == ("ALL",)
    assert "SYS_ADMIN" not in captured_kwargs.get("cap_add", [])
