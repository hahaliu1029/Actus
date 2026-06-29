"""Shared Docker test helpers for the C5d-1 sandbox runtime-profile proof.

A NORMAL importable module (NOT conftest.py — pytest's conftest is not a normal
import target here; codex R2 import-probe confirmed `from conftest import ...`
fails). Tests do `from tests.sandbox._docker_helpers import ...`.

Split from the old single `_docker_or_skip()` (which bundled an alpine:3.20
requirement) into a client helper + a per-image requirement so each test
requires only the image it needs (codex R2#3).

INV-2: `hardening_kwargs()` returns the LITERAL production-emitted kwargs by
calling the shipped C5c wiring — never a hand-written cap list.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.application.services.sandbox_runtime_policy import compile_runtime_policy
from app.infrastructure.external.sandbox.container_hardening import (
    container_hardening_kwargs,
)

# ── capability bits (linux/capability.h; verified codex R1) ──────────────── #
CAP_NET_BIND_SERVICE = 10
CAP_NET_RAW = 13
CAP_MKNOD = 27
CAP_AUDIT_WRITE = 29
DROPPED_CAP_BITS: frozenset[int] = frozenset(
    {CAP_NET_BIND_SERVICE, CAP_NET_RAW, CAP_MKNOD, CAP_AUDIT_WRITE}
)
EPERM: int = 1

# Images used by the proof.
ALPINE_IMAGE = "alpine:3.20"                 # existing memory-mount test
PYTHON_ALPINE_IMAGE = "python:3.12-alpine"   # negative cap-proof
SANDBOX_IMAGE = "actus-sandbox:latest"       # positive smoke


def _require_env() -> bool:
    return os.environ.get("ACTUS_REQUIRE_SANDBOX_TESTS") == "1"


def _miss(msg: str) -> None:
    """fail under REQUIRE=1 (no silent-skip safety regression), else skip."""
    if _require_env():
        pytest.fail(f"ACTUS_REQUIRE_SANDBOX_TESTS=1 but {msg}")
    pytest.skip(msg)


def _docker_client_or_skip():
    """Return a docker client, trying the macOS/Linux/CI socket candidates.

    Decoupled from any image requirement (codex R2#3): callers pair this with
    `_require_image(client, <image>)`.
    """
    try:
        import docker
    except ImportError:
        _miss("docker SDK unavailable")

    candidates: list[str] = []
    env_host = os.environ.get("DOCKER_HOST")
    if env_host:
        candidates.append(env_host)
    home_sock = Path.home() / ".docker" / "run" / "docker.sock"
    if home_sock.exists():
        candidates.append(f"unix://{home_sock}")
    if Path("/var/run/docker.sock").exists():
        candidates.append("unix:///var/run/docker.sock")

    client = None
    last_err: Exception | None = None
    for url in candidates or [None]:
        try:
            client = docker.DockerClient(base_url=url) if url else docker.from_env()
            client.ping()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            client = None
    if client is None:
        _miss(f"docker daemon unreachable (tried {candidates or 'env'}): {last_err}")
    return client


def _require_image(client, image: str) -> None:
    """fail-not-skip (REQUIRE=1) / skip if a needed image is absent locally."""
    try:
        client.images.get(image)
    except Exception:  # noqa: BLE001 — docker.errors.ImageNotFound + transport
        _miss(f"image {image} missing locally; CI must pre-pull or build it")


def decode_caps(hex_str: str) -> set[int]:
    """Bit-set of a /proc/<pid>/status Cap* hex value: bit N set iff cap N present."""
    val = int(hex_str, 16)
    return {i for i in range(64) if (val >> i) & 1}


class _StubSettings(SimpleNamespace):
    """Duck-typed Settings for build_settings_view/compile_runtime_policy.

    Provides EVERY attribute build_settings_view reads directly (else
    AttributeError) plus the two hardening flags. Defaults mirror config.yaml.
    """

    def __init__(self, *, hardening: bool, nnp: bool = False) -> None:
        super().__init__(
            sandbox_image=SANDBOX_IMAGE,
            sandbox_network="",
            sandbox_mem_limit="4g",
            sandbox_default_cwd="/home/ubuntu",
            sandbox_https_proxy="",
            sandbox_http_proxy="",
            sandbox_no_proxy="",
            sandbox_memory_mount_target="/workspace/.memory",
            sandbox_memory_mount_enabled=False,
            sandbox_address="",
            sandbox_runtime_hardening_enabled=hardening,
            sandbox_no_new_privileges_enabled=nnp,
        )


def hardening_kwargs(*, hardening: bool, nnp: bool = False) -> dict:
    """The EXACT docker-py kwargs production emits (INV-2), or {} when off.

    Mirrors production: `create(runtime_policy=None)` (flag off) → no hardening
    kwargs merged; `compile_runtime_policy` returns None when the flag is off,
    so we must NOT pass None to container_hardening_kwargs (codex R1#2).
    """
    policy = compile_runtime_policy(_StubSettings(hardening=hardening, nnp=nnp))
    return container_hardening_kwargs(policy) if policy is not None else {}


def run_container_probe(
    client, image: str, command, *, environment=None, timeout: int = 60,
    container_kwargs=None,
) -> tuple[int, bytes, bytes]:
    """Run `image` with `command` to completion; return (exit_code, stdout, stderr).

    `container_kwargs` is forwarded to `containers.run` — THIS is how the hardening
    kwargs (cap_drop/pids_limit/security_opt) reach the container (plan-R1 P1: the
    proof is invalid if they are not threaded through). stdout and stderr are
    SEPARATED (codex R4 / R2 §9 contract): JSON probes read stdout only (a stderr
    warning can never corrupt the parse), and stderr is returned for diagnostics.
    Always removed (force=True).
    """
    container = client.containers.run(
        image,
        command=command,
        environment=environment or {},
        detach=True,
        remove=False,
        **(container_kwargs or {}),
    )
    try:
        result = container.wait(timeout=timeout)
        exit_code = int(result.get("StatusCode", -1))
        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True)
        return exit_code, stdout, stderr
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


# Robust pids.max read for a RUNNING container (real-image assertions, R2 §9
# contract): fast-path `/sys/fs/cgroup/pids.max`, else resolve the process's own
# pids controller via /proc/self/cgroup (host-cgroupns safe). Same algorithm the
# alpine cap-proof inlines (necessarily duplicated: that runs in a separate image).
_PIDS_MAX_PY = (
    "def r():\n"
    " try:\n"
    "  v=open('/sys/fs/cgroup/pids.max').read().strip()\n"
    "  if v=='512': return v\n"
    " except OSError:\n"
    "  v=None\n"
    " try: cg=open('/proc/self/cgroup').read()\n"
    " except OSError: cg=''\n"
    " for ln in cg.splitlines():\n"
    "  p=ln.split(':',2)\n"
    "  if len(p)!=3: continue\n"
    "  if p[0]=='0': c='/sys/fs/cgroup'+(p[2].rstrip('/') or '')+'/pids.max'\n"
    "  elif 'pids' in p[1].split(','): c='/sys/fs/cgroup/pids'+p[2]+'/pids.max'\n"
    "  else: continue\n"
    "  try: return open(c).read().strip()\n"
    "  except OSError: continue\n"
    " try: return open('/sys/fs/cgroup/pids/pids.max').read().strip()\n"
    " except OSError: return v or ''\n"
    "print(r(), end='')"
)


def read_pids_max(container) -> str:
    """Exec the robust pids.max resolution inside a running container (needs python3)."""
    res = container.exec_run(["python3", "-c", _PIDS_MAX_PY], demux=True)
    out = (res.output[0] or b"") if res.output else b""
    return out.decode().strip()
