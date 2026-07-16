"""SPM PR-2 Task 23 — minimal HONEST on_demand e2e on the REAL actus-sandbox image.

Markers ``sandbox`` + ``sandbox_real_image``: the on_demand supply chain must run
the real ``DockerSandbox.create`` → ``ensure_sandbox`` (supervisord readiness
probe), which needs the real ``actus-sandbox`` image. The ordinary adversarial
``sandbox`` job does NOT build that image; only the heavy ``sandbox and
sandbox_real_image`` CI job does (``ci.yml`` real-image job; ``pytest.ini:32``
marker). The file lives in ``tests/sandbox/`` so both jobs collect from here.

Drive face = **minimal honest e2e**: REAL lifecycle + REAL ``DockerSandbox`` +
REAL provisioner (an in-memory UoW stands in for Postgres — this exercises the
CONTAINER supply chain, not DB persistence):

① first ``provisioner.get()`` builds a container and can ``exec`` a shell command;
② a second ``get()`` (after dropping the ready cache) reuses the SAME container
   via the registry-hit path — zero new containers;
③ cleanup destroys it.

It does NOT drive the full agent graph; agent-level flows are covered by the
service-fake flow group (``test_sandbox_on_demand_flows.py``).

This file will NOT run in the ordinary local/unit run (it is ``sandbox``-marked,
excluded by ``pytest.ini`` addopts, and self-skips without a Docker daemon +
image). It MUST collect cleanly. All Docker access lives inside the test body /
skip guard (INV-7 mirror of the other ``tests/sandbox`` files).
"""
from __future__ import annotations

import uuid

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.application.services.sandbox_provisioner import SandboxProvisioner
from app.domain.models.session import (
    DestroyReason,
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox
from tests.sandbox._docker_helpers import (
    SANDBOX_IMAGE,
    _docker_client_or_skip,
    _require_image,
)

pytestmark = [pytest.mark.sandbox, pytest.mark.sandbox_real_image, pytest.mark.anyio]

_READY_TIMEOUT = 180  # real create + supervisord readiness can be slow in CI


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── minimal in-memory UoW (container supply chain e2e — DB is not under test) ──


class _MemSessionRepo:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions
        self.events: list = []

    async def get_by_id(self, session_id: str):
        return self._sessions.get(session_id)

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session

    async def add_event(self, session_id: str, event) -> None:
        self.events.append(event)


class _MemAuditRepo:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def create(self, **kwargs) -> None:
        self.rows.append(dict(kwargs))


class _MemUoW:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self.session = _MemSessionRepo(sessions)
        self.sandbox_lifecycle_log = _MemAuditRepo()

    async def __aenter__(self) -> "_MemUoW":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


def _labelled_count(client, session_id: str) -> int:
    """Number of live/exited containers this platform stamped for ``session_id``
    (``actus.session_id`` label — the same label ``DockerSandbox.create`` writes)."""
    return len(
        client.containers.list(
            all=True, filters={"label": f"actus.session_id={session_id}"}
        )
    )


def _force_remove_labelled(client, session_id: str) -> None:
    for c in client.containers.list(
        all=True, filters={"label": f"actus.session_id={session_id}"}
    ):
        try:
            c.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


async def test_on_demand_provision_reuse_and_destroy():
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)

    session_id = f"e2e-ondemand-{uuid.uuid4().hex[:8]}"
    sessions = {
        session_id: Session(
            id=session_id,
            user_id="e2e-user",
            status=SessionStatus.PENDING,
            sandbox_binding=SandboxBinding(state=SandboxBindingState.UNBOUND),
        )
    }
    uow = _MemUoW(sessions)
    svc = SandboxLifecycleService(
        sandbox_cls=DockerSandbox, uow_factory=lambda: uow
    )
    prov = SandboxProvisioner(
        session_id=session_id,
        user_id="e2e-user",
        lifecycle=svc,
        timeout_seconds=_READY_TIMEOUT,
        trigger="tool_call",
    )

    try:
        # ① first tool touch: provision builds exactly one container + exec works.
        handle1 = await prov.get()
        assert _labelled_count(client, session_id) == 1

        result = await handle1.exec_command(
            session_id="e2e-shell",
            exec_dir="/home/ubuntu",
            command="echo actus-e2e-ok",
            wait_seconds=60,
        )
        # the exec round-trip succeeding proves the real container's shell service
        # is live (create → supervisord-ready → handle → exec end-to-end).
        assert result.success is True

        # ② reuse: drop the ready cache so get() re-enters lifecycle.acquire() →
        # the real registry hit returns a handle on the SAME container. No bind_new,
        # no new container.
        prov.release_held_handle()
        handle2 = await prov.get()
        assert handle2.id == handle1.id
        assert _labelled_count(client, session_id) == 1
    finally:
        # ③ cleanup: destroy tears the container down.
        try:
            await svc.destroy(session_id, reason=DestroyReason.SESSION_DELETE)
        except Exception:  # noqa: BLE001 — best-effort; hard-remove below backstops
            pass
        _force_remove_labelled(client, session_id)

    assert _labelled_count(client, session_id) == 0
