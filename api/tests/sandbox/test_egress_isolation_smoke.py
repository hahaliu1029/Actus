"""C5d-5/6 internal-network egress proof (CI-only, image-agnostic).

Proves the KERNEL boundary: a container on an `internal=True` Docker network cannot reach a public
IP (direct-IP TCP connect fails within a bound), an intra-network peer IS reachable, and the SAME
probe on a NON-internal network SUCCEEDS (teeth — mirrors C5d-1's ON/OFF pattern). Marked
`pytest.mark.sandbox` (NOT sandbox_real_image): alpine, no actus image → runs in the existing
`sandbox-adversarial` CI job by marker (NO ci.yml change). INV-7: all docker access inside tests.
"""
from __future__ import annotations

import uuid

import pytest

from tests.sandbox._docker_helpers import (
    PYTHON_ALPINE_IMAGE,  # "python:3.12-alpine" — alpine:3.20 lacks python3; this image ships it + busybox nc
    _docker_client_or_skip,
    _require_image,
    run_container_probe,
)

pytestmark = pytest.mark.sandbox

# Bounded direct-IP egress probe (NOT DNS — codex Q6): a raw TCP connect to a public IP. Prints
# "OK" on connect success, "FAIL:<errno-or-timeout>" otherwise. The bound is short so a DROP/no-route
# internal bridge resolves quickly; we assert the connect did NOT succeed within N s (R4 —
# ENETUNREACH/no-route OR a bounded timeout both count, we do NOT require strictly "timeout").
_EGRESS_PROBE = (
    "import socket\n"
    "try:\n"
    "    s=socket.create_connection(('1.1.1.1',443),3)\n"
    "    s.close(); print('OK')\n"
    "except OSError as e:\n"
    "    print('FAIL:'+str(getattr(e,'errno','?')))\n"
)


def _new_network(client, *, internal):
    return client.networks.create(
        f"actus-egress-test-{uuid.uuid4().hex[:8]}", driver="bridge", internal=internal
    )


def test_internal_network_blocks_direct_ip_egress():
    client = _docker_client_or_skip()
    _require_image(client, PYTHON_ALPINE_IMAGE)
    net = _new_network(client, internal=True)
    try:
        # internal=True → external egress blocked (bounded-fail, NOT "OK"). PYTHON_ALPINE_IMAGE
        # ("python:3.12-alpine") ships python3 so the `python3 -c` probe runs.
        code, out, _ = run_container_probe(
            client, PYTHON_ALPINE_IMAGE, ["python3", "-c", _EGRESS_PROBE],
            container_kwargs={"network": net.name}, timeout=30,
        )
        assert b"OK" not in out, (
            f"internal network unexpectedly allowed direct-IP egress: {out!r}")
    finally:
        net.remove()


def test_non_internal_network_allows_direct_ip_egress_teeth():
    # NEGATIVE CONTROL (teeth): the SAME probe on a NON-internal bridge SUCCEEDS — proves the test
    # measures real egress, not a universally-broken probe (mirrors C5d-1's ON/OFF teeth).
    client = _docker_client_or_skip()
    _require_image(client, PYTHON_ALPINE_IMAGE)
    net = _new_network(client, internal=False)
    try:
        code, out, _ = run_container_probe(
            client, PYTHON_ALPINE_IMAGE, ["python3", "-c", _EGRESS_PROBE],
            container_kwargs={"network": net.name}, timeout=30,
        )
        assert b"OK" in out, (
            f"non-internal network unexpectedly blocked egress (test has no teeth): {out!r}")
    finally:
        net.remove()


def test_internal_network_intra_peer_reachable():
    # api↔sandbox connectivity proxy: a second container on the SAME internal net IS reachable by IP
    # (the boundary blocks EXTERNAL egress, not intra-network L3 — INV-2 of the mechanism). Start a
    # listener, read its intra-net IP, then connect to it from a probe container on the same net.
    client = _docker_client_or_skip()
    _require_image(client, PYTHON_ALPINE_IMAGE)
    net = _new_network(client, internal=True)
    listener = None
    try:
        # busybox `nc -l` listener on :9000 (python:3.12-alpine ships busybox nc too); detached, internal net.
        listener = client.containers.run(
            PYTHON_ALPINE_IMAGE, command=["nc", "-l", "-p", "9000"],
            detach=True, remove=False, network=net.name,
        )
        listener.reload()
        peer_ip = (
            (listener.attrs.get("NetworkSettings", {}) or {})
            .get("Networks", {}).get(net.name, {}).get("IPAddress")
        )
        assert peer_ip, f"listener got no intra-net IP: {listener.attrs.get('NetworkSettings')}"
        # probe: connect to the peer's intra-net IP:9000 → must succeed (intra-net L3 is up).
        probe = (
            "import socket,sys\n"
            f"try:\n"
            f"    socket.create_connection(({peer_ip!r},9000),5).close(); print('PEER_OK')\n"
            f"except OSError as e:\n"
            f"    print('PEER_FAIL:'+str(getattr(e,'errno','?')))\n"
        )
        code, out, _ = run_container_probe(
            client, PYTHON_ALPINE_IMAGE, ["python3", "-c", probe],
            container_kwargs={"network": net.name}, timeout=30,
        )
        assert b"PEER_OK" in out, f"intra-network peer not reachable: {out!r}"
    finally:
        if listener is not None:
            try:
                listener.remove(force=True)
            except Exception:  # noqa: BLE001
                pass
        net.remove()
