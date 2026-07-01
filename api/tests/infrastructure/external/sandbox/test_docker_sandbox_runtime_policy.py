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


class _FakeNetwork:
    def __init__(self, internal):
        self.attrs = {"Internal": internal}


class _FakeNetworks:
    def __init__(self, *, internal=True, raise_not_found=False):
        self._internal = internal
        self._raise = raise_not_found

    def get(self, name):
        import docker
        if self._raise:
            raise docker.errors.NotFound(f"no such network {name}")
        return _FakeNetwork(self._internal)


class _FakeClient:
    def __init__(self, sink: dict, *, internal=True, raise_not_found=False):
        self.containers = _FakeContainers(sink)
        self.networks = _FakeNetworks(internal=internal, raise_not_found=raise_not_found)

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


def _patch_client(monkeypatch, sink, *, internal=True, raise_not_found=False):
    # Egress create-path tests need a client whose network reports a chosen Internal value.
    monkeypatch.setattr(
        DockerSandbox, "_create_docker_client",
        classmethod(lambda cls: _FakeClient(sink, internal=internal, raise_not_found=raise_not_found)),
    )
    monkeypatch.setattr(DockerSandbox, "_wait_for_container_ip",
                        classmethod(lambda cls, container, **kw: "1.2.3.4"))


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
        read_only_rootfs=False, egress_network=None,
        cap_drop=(), cap_add=(), security_opt=(), pids_limit=None,
        mounts=(),
    )


def _hardened() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, egress_network=None,
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
        read_only_rootfs=False, egress_network=None, cap_drop=("ALL",),
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
        read_only_rootfs=False, egress_network=None, cap_drop=("ALL",), cap_add=("SYS_ADMIN",),
        security_opt=(), pids_limit=512, mounts=(),
    )
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=bad)
    # codex final-audit P3: pin fail-closed-BEFORE-run — sink stays empty because the fake
    # containers.run is never reached (the out-of-ceiling cap_add is rejected first).
    assert captured_kwargs == {}, "validation must reject BEFORE containers.run (fail-closed)"


def test_create_task_strict_config_passes_validator(captured_kwargs, monkeypatch):
    # The real strict config is ACCEPTED → containers.run proceeds, applied carries it.
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=_strict())
    assert sandbox.applied_runtime_policy.cap_drop == ("ALL",)
    assert "SYS_ADMIN" not in captured_kwargs.get("cap_add", [])


# ---- C5d-3: non-root run_as_user through _create_task ----------------------- #
def _nonroot_conservative() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user="1000:1000",
        read_only_rootfs=False, egress_network=None,
        cap_drop=("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"),
        cap_add=(), security_opt=(), pids_limit=512, mounts=(),
    )


def test_nonroot_policy_threads_user_into_kwargs_and_applied(captured_kwargs, monkeypatch):
    # End-to-end: non-root policy → translator emits user="1000:1000" → _create_task merges
    # → build_applied carries run_as_user into the applied snapshot.
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=_nonroot_conservative())
    assert captured_kwargs["user"] == "1000:1000"
    applied = sandbox.applied_runtime_policy
    assert applied is not None and applied.capture_kind == "applied"
    assert applied.run_as_user == "1000:1000"


def test_create_task_rejects_forged_root_user_with_typed_error(captured_kwargs, monkeypatch):
    # A policy carrying a forged ROOT user → the validator rejects it with the typed
    # SandboxHardeningConfigError BEFORE containers.run (not re-wrapped by the broad except).
    from app.infrastructure.external.sandbox.container_hardening import (
        SandboxHardeningConfigError,
    )
    _patch_settings(monkeypatch)
    bad = ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user="0:0",
        read_only_rootfs=False, egress_network=None, cap_drop=("NET_RAW",), cap_add=(), security_opt=(),
        pids_limit=512, mounts=(),
    )
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=bad)
    # codex final-audit P3: pin fail-closed-BEFORE-run — the fake containers.run never
    # populated the sink, so validation rejected the forged root user before any container was
    # created (the typed raise alone would still pass if validate moved after containers.run).
    assert captured_kwargs == {}, "validation must reject BEFORE containers.run (fail-closed)"


def test_inv0_run_as_user_off_create_has_no_user_key(captured_kwargs, monkeypatch):
    # INV-0 (Surface ③): hardening ON + run_as_user OFF → NO `user` kwarg; the FULL
    # container_config == the OFF baseline + EXACTLY the C5c conservative kwargs, nothing
    # more. A stray unconditional `user` emission breaks it. Mirrors the C5d-2 tier-2 golden.
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _ConservativeNoNonroot:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/root"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = False
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = False
        sandbox_strict_caps_enabled = False
        sandbox_run_as_user_enabled = False   # the bit under test

    _patch_settings(monkeypatch)
    DockerSandbox._create_task(user_id=None, runtime_policy=None)
    off = {k: v for k, v in captured_kwargs.items() if k != "name"}
    DockerSandbox._create_task(
        user_id=None, runtime_policy=compile_runtime_policy(_ConservativeNoNonroot()))
    conservative = {k: v for k, v in captured_kwargs.items() if k != "name"}
    assert "user" not in conservative
    assert conservative == off | {
        "cap_drop": ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"],
        "pids_limit": 512,
    }


def test_run_as_user_on_create_emits_user_via_compile_chain(captured_kwargs, monkeypatch):
    # Full compile chain ON: settings(run_as_user on) → compile_runtime_policy → _create_task
    # → user="1000:1000" kwarg + the applied snapshot carries it. Exercises the SHARED compile
    # chain (compile_runtime_policy → _create_task) that BOTH production create() paths route
    # through (sandbox_lifecycle_service.py:327 + skill_creator_service.py:514); it calls
    # _create_task directly, so it proves the shared chain, not the outer wrappers (R5 P3 — §8
    # "no new call sites" rests on both wrappers already calling compile_runtime_policy).
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _NonRoot:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/home/ubuntu"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = False
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = False
        sandbox_strict_caps_enabled = False
        sandbox_run_as_user_enabled = True

    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(
        user_id=None, runtime_policy=compile_runtime_policy(_NonRoot()))
    assert captured_kwargs["user"] == "1000:1000"
    assert sandbox.applied_runtime_policy.run_as_user == "1000:1000"


def test_strict_plus_run_as_user_through_create_path(captured_kwargs, monkeypatch):
    # spec §3 (R5 P3): all cap×nonroot hardened combos are valid — exercise strict + non-root
    # TOGETHER through _create_task so cap_drop=ALL + the 9-cap cap_add + user="1000:1000" all
    # reach containers.run AND pass the validator (vetted user + vetted caps) in one compose.
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _StrictNonRoot:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/home/ubuntu"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = False
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = False
        sandbox_strict_caps_enabled = True
        sandbox_run_as_user_enabled = True

    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(
        user_id=None, runtime_policy=compile_runtime_policy(_StrictNonRoot()))
    assert captured_kwargs["user"] == "1000:1000"
    assert captured_kwargs["cap_drop"] == ["ALL"]
    assert captured_kwargs["cap_add"] == [
        "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",
        "SETUID", "SETGID", "SETPCAP", "SETFCAP", "KILL",
    ]
    assert sandbox.applied_runtime_policy.run_as_user == "1000:1000"
    assert sandbox.applied_runtime_policy.cap_drop == ("ALL",)


# ---- C5d-4: read-only rootfs → /home/ubuntu anon-volume append -------------- #
def _readonly_conservative() -> ContainerRuntimePolicy:
    # read_only tier ON, run_as_user OFF (isolates the volume-append concern; the full
    # read_only × non-root compose is the §9① smoke). user=None → validator's root-default boot.
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=True, egress_network=None,
        cap_drop=("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"),
        cap_add=(), security_opt=(), pids_limit=512, mounts=(),
    )


def test_read_only_policy_appends_anon_volume_memory_absent(captured_kwargs, monkeypatch):
    # skill_creator path (no user_id → no memory mount): mounts == [the /home/ubuntu anon volume].
    from app.infrastructure.external.sandbox.container_hardening import _READONLY_WORKSPACE_TARGET
    _patch_settings(monkeypatch)  # memory_mount_enabled=False → deterministic
    DockerSandbox._create_task(user_id=None, runtime_policy=_readonly_conservative())
    assert captured_kwargs["read_only"] is True
    assert captured_kwargs["tmpfs"] == {"/tmp": "rw,exec,nosuid,nodev,size=512m"}
    mounts = captured_kwargs["mounts"]
    assert len(mounts) == 1
    vol = mounts[0]  # docker-py Mount IS a dict
    assert vol["Target"] == _READONLY_WORKSPACE_TARGET == "/home/ubuntu"
    assert vol["Type"] == "volume"
    assert vol["Source"] is None
    assert vol["ReadOnly"] is False


def test_read_only_policy_appends_anon_volume_after_memory_mount(captured_kwargs, monkeypatch, tmp_path):
    # agent path (mountable user_id → memory :ro bind present): mounts == [memory_bind,
    # /home/ubuntu volume] — setdefault APPENDS, never clobbers the memory mount.
    host_root = tmp_path / "host"
    cont_root = tmp_path / "cont"
    host_root.mkdir()
    cont_root.mkdir()
    _patch_settings(
        monkeypatch, sandbox_memory_mount_enabled=True,
        memory_root_host=str(host_root), memory_root_container=str(cont_root),
    )
    DockerSandbox._create_task(user_id="u1", runtime_policy=_readonly_conservative())
    mounts = captured_kwargs["mounts"]
    assert [m["Target"] for m in mounts] == ["/workspace/.memory", "/home/ubuntu"]
    mem, vol = mounts
    assert mem["Type"] == "bind" and mem["ReadOnly"] is True
    assert vol["Type"] == "volume" and vol["Source"] is None and vol["ReadOnly"] is False


def test_read_only_off_appends_no_volume_and_no_read_only_keys(captured_kwargs, monkeypatch):
    # INV-0: a conservative (read_only OFF) policy adds NO read_only / tmpfs / volume.
    _patch_settings(monkeypatch)
    DockerSandbox._create_task(user_id=None, runtime_policy=_hardened())
    assert "read_only" not in captured_kwargs
    assert "tmpfs" not in captured_kwargs
    assert "mounts" not in captured_kwargs


def test_create_task_rejects_forged_system_tmpfs_with_typed_error(captured_kwargs, monkeypatch):
    # A forged config carrying a system-path tmpfs → the validator rejects it with the typed
    # SandboxHardeningConfigError BEFORE containers.run (not re-wrapped by the broad except). The
    # model field read_only_rootfs is a bool (cannot encode a bad tmpfs), so forge by monkeypatching
    # the translator to emit a /usr tmpfs — this proves INV-8 fail-closes the create path.
    from app.infrastructure.external.sandbox.container_hardening import (
        SandboxHardeningConfigError,
    )
    from app.infrastructure.external.sandbox import docker_sandbox as ds_mod
    _patch_settings(monkeypatch)
    monkeypatch.setattr(
        ds_mod, "container_hardening_kwargs",
        lambda policy: {"read_only": True, "tmpfs": {"/usr": "rw"}},
    )
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=_readonly_conservative())


def test_read_only_rootfs_off_helper_emits_no_read_only_or_tmpfs():
    # Surface ②(a) / INV-0: hardening ON + read_only OFF → the production helper emits the C5d-3
    # conservative kwargs and NOTHING read-only (pins INV-0; a stray unconditional emission breaks it).
    from tests.sandbox._docker_helpers import hardening_kwargs
    kw = hardening_kwargs(hardening=True, read_only_rootfs=False)
    assert "read_only" not in kw and "tmpfs" not in kw
    assert set(kw) == {"cap_drop", "pids_limit"}


def test_read_only_rootfs_on_helper_emits_read_only_and_tmpfs():
    # Surface ②(b): hardening ON + read_only ON → read_only=True + the vetted /tmp tmpfs, with the
    # conservative caps unchanged. This pins the EXACT kwargs the §9① smoke boots the real image with.
    from tests.sandbox._docker_helpers import hardening_kwargs
    kw = hardening_kwargs(hardening=True, read_only_rootfs=True)
    assert kw["read_only"] is True
    assert kw["tmpfs"] == {"/tmp": "rw,exec,nosuid,nodev,size=512m"}
    assert kw["cap_drop"] == ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"]
    assert kw["pids_limit"] == 512
    # and the validator accepts that vetted tmpfs but rejects a forged system-path one.
    from app.infrastructure.external.sandbox.container_hardening import (
        SandboxHardeningConfigError, validate_hardening_config,
    )
    validate_hardening_config({**kw})  # vetted → no raise
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"read_only": True, "tmpfs": {"/usr": "rw"}})


# ---- C5d-5/6: egress through _create_task ---------------------------------- #
def _egress_policy() -> ContainerRuntimePolicy:
    return ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, egress_network="actus-sandbox-internal",
        cap_drop=("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"),
        cap_add=(), security_opt=(), pids_limit=512, mounts=(),
    )


def test_egress_on_emits_network_and_builds_applied(captured_kwargs, monkeypatch):
    sink: dict = {}
    _patch_client(monkeypatch, sink, internal=True)
    _patch_settings(monkeypatch)
    sandbox = DockerSandbox._create_task(user_id=None, runtime_policy=_egress_policy())
    assert sink["network"] == "actus-sandbox-internal"  # translator override of the base network
    applied = sandbox.applied_runtime_policy
    assert applied is not None and applied.egress_network == "actus-sandbox-internal"


def test_egress_off_create_has_no_network_from_hardening(captured_kwargs, monkeypatch):
    # INV-0: a conservative (egress_network=None) policy adds NO `network` key from hardening; the
    # base line-231 network (None in _patch_settings) stands → no `network` in the final config.
    _patch_settings(monkeypatch)
    DockerSandbox._create_task(user_id=None, runtime_policy=_hardened())
    assert "network" not in captured_kwargs


def test_egress_on_non_internal_network_rejected_before_run(monkeypatch):
    # Fail-closed: the preflight sees Internal=False → SandboxHardeningConfigError BEFORE run.
    from app.infrastructure.external.sandbox.container_hardening import SandboxHardeningConfigError
    sink: dict = {}
    _patch_client(monkeypatch, sink, internal=False)
    _patch_settings(monkeypatch)
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=_egress_policy())
    assert sink == {}, "preflight must reject BEFORE containers.run (fail-closed)"


def test_egress_on_missing_network_rejected_before_run(monkeypatch):
    from app.infrastructure.external.sandbox.container_hardening import SandboxHardeningConfigError
    sink: dict = {}
    _patch_client(monkeypatch, sink, raise_not_found=True)
    _patch_settings(monkeypatch)
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=_egress_policy())
    assert sink == {}


def test_egress_on_name_mismatch_rejected_before_run(monkeypatch):
    # The validator name-confinement fires when the emitted network != expected. Forge by
    # monkeypatching the translator to emit a DIFFERENT network than runtime_policy.egress_network.
    from app.infrastructure.external.sandbox.container_hardening import SandboxHardeningConfigError
    from app.infrastructure.external.sandbox import docker_sandbox as ds_mod
    sink: dict = {}
    _patch_client(monkeypatch, sink, internal=True)
    _patch_settings(monkeypatch)
    monkeypatch.setattr(
        ds_mod, "container_hardening_kwargs",
        lambda policy: {"cap_drop": ["NET_RAW"], "pids_limit": 512, "network": "actus-net"},
    )
    with pytest.raises(SandboxHardeningConfigError):
        DockerSandbox._create_task(user_id=None, runtime_policy=_egress_policy())
    assert sink == {}


# ---- C5d-5/6: compile_runtime_policy threads worker_type ------------------- #
def test_compile_runtime_policy_threads_worker_type_for_child_egress():
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _ChildEgress:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/home/ubuntu"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = False
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = False
        sandbox_strict_caps_enabled = False
        sandbox_run_as_user_enabled = False
        sandbox_read_only_rootfs_enabled = False
        sandbox_egress_isolation_enabled = False
        sandbox_child_egress_isolation_enabled = True
        sandbox_egress_internal_network = "actus-sandbox-internal"

    # subagent → isolated; root → not (child flag only)
    sub = compile_runtime_policy(_ChildEgress(), worker_type="subagent")
    root = compile_runtime_policy(_ChildEgress(), worker_type="root")
    assert sub.egress_network == "actus-sandbox-internal"
    assert root.egress_network is None


def test_compile_runtime_policy_default_worker_type_is_root():
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _ChildEgress:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/home/ubuntu"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = False
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = False
        sandbox_strict_caps_enabled = False
        sandbox_run_as_user_enabled = False
        sandbox_read_only_rootfs_enabled = False
        sandbox_egress_isolation_enabled = False
        sandbox_child_egress_isolation_enabled = True
        sandbox_egress_internal_network = "actus-sandbox-internal"

    # No worker_type arg → defaults to "root" → not isolated under the child flag.
    assert compile_runtime_policy(_ChildEgress()).egress_network is None


def test_compile_runtime_policy_off_returns_none_regardless_of_worker_type():
    # INV-0: hardening OFF → None whatever the worker_type.
    from app.application.services.sandbox_runtime_policy import compile_runtime_policy

    class _Off:
        sandbox_runtime_hardening_enabled = False

    assert compile_runtime_policy(_Off(), worker_type="subagent") is None
