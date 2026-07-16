"""Task 5 (SPM PR-1a): DockerSandbox provision-mode surface.

Covers the four pieces the lifecycle layer already drives against fakes:
  * container labels (``actus.session_id`` / ``actus.attempt``) stamped in
    ``_create_task`` and left absent when the kwargs are None,
  * the IP-wait *half-success* cleanup (spec §5.2c-3): any failure after
    ``containers.run`` tears the container down on the create path itself,
  * ``get_strict`` — distinguishes daemon-unreachable (raise
    ``SandboxDaemonUnreachable``) from terminal / gone (return None),
  * ``list_managed_containers`` / ``remove_container`` — the label-sweep
    primitives Task 6's reconcile consumes.

Async-runner note (DEVIATION from brief, intentional): this project ships
**pytest-anyio**, NOT pytest-asyncio (see ``tests/conftest.py`` and the sibling
``tests/app/application/services/test_sandbox_provision_flight.py``). Under this
repo a bare ``@pytest.mark.asyncio`` leaves the coroutine *un-awaited* → the
async test silently "passes" without ever calling ``get_strict`` (a false-green
that would break the required RED step). The brief's per-method
``@pytest.mark.asyncio`` markers are therefore replaced by the module-level
``pytestmark = pytest.mark.anyio`` — the authoritative repo convention, which
also mixes sync + async methods under one marker (mirrors
``tests/app/infrastructure/external/test_github_search_client.py``).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docker.errors import APIError, NotFound

from app.domain.errors.sandbox_lifecycle import SandboxDaemonUnreachable
from app.infrastructure.external.sandbox import docker_sandbox as ds_mod
from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.anyio


def _mk_container(name="c1", ip="10.0.0.2", status="running", labels=None, created=None):
    c = MagicMock()
    c.name = name
    c.status = status
    c.attrs = {
        "NetworkSettings": {"IPAddress": ip, "Networks": {}},
        "Created": created or "2026-07-13T00:00:00Z",
        "Config": {"Labels": labels or {}},
    }
    return c


class TestCreateTaskLabelsAndCleanup:
    @patch.object(DockerSandbox, "_create_docker_client")
    def test_labels_injected_when_session_and_attempt_given(self, mk_client):
        client = MagicMock()
        mk_client.return_value = client
        client.containers.run.return_value = _mk_container()
        DockerSandbox._create_task(None, session_id="sess-1", attempt="a" * 32)
        cfg = client.containers.run.call_args.kwargs
        assert cfg["labels"] == {"actus.session_id": "sess-1", "actus.attempt": "a" * 32}

    @patch.object(DockerSandbox, "_create_docker_client")
    def test_no_labels_key_when_not_given(self, mk_client):
        client = MagicMock()
        mk_client.return_value = client
        client.containers.run.return_value = _mk_container()
        DockerSandbox._create_task(None)
        assert "labels" not in client.containers.run.call_args.kwargs

    @patch.object(DockerSandbox, "_wait_for_container_ip", return_value=None)
    @patch.object(DockerSandbox, "_create_docker_client")
    def test_ip_wait_failure_removes_container(self, mk_client, _wait):
        client = MagicMock()
        mk_client.return_value = client
        container = _mk_container(ip=None)
        client.containers.run.return_value = container
        with pytest.raises(Exception):
            DockerSandbox._create_task(None)
        container.remove.assert_called_once_with(force=True)


class TestCreateTaskConstructAndCloseCleanup:
    """FIX-B (P1-3): widen the half-success cleanup so it also covers the
    ``DockerSandbox(...)`` construction, and make the finally's
    ``docker_client.close()`` best-effort so a close failure can neither discard a
    successfully-built sandbox nor mask an in-flight exception."""

    @patch.object(DockerSandbox, "_wait_for_container_ip", return_value="10.0.0.2")
    @patch.object(DockerSandbox, "_create_docker_client")
    def test_construct_failure_after_run_removes_container(self, mk_client, _wait):
        """The ``DockerSandbox(...)`` construction raising after a successful
        ``containers.run`` must tear the container down (``remove(force=True)``)
        and propagate — no ownerless leaked container."""
        client = MagicMock()
        mk_client.return_value = client
        container = _mk_container()
        client.containers.run.return_value = container

        def _raise_init(self, *args, **kwargs):
            raise RuntimeError("ctor boom")

        with patch.object(DockerSandbox, "__init__", _raise_init):
            with pytest.raises(Exception):
                DockerSandbox._create_task(None)
        container.remove.assert_called_once_with(force=True)

    @patch.object(DockerSandbox, "_wait_for_container_ip", return_value="10.0.0.2")
    @patch.object(DockerSandbox, "_create_docker_client")
    def test_close_failure_does_not_discard_sandbox(self, mk_client, _wait):
        """A ``docker_client.close()`` raising in the finally must be swallowed so
        the successfully-built sandbox is still returned (no exception)."""
        client = MagicMock()
        client.close.side_effect = RuntimeError("close boom")
        mk_client.return_value = client
        container = _mk_container()
        client.containers.run.return_value = container
        sandbox = DockerSandbox._create_task(None)
        assert isinstance(sandbox, DockerSandbox)


class TestGetStrict:
    @patch.object(DockerSandbox, "_create_docker_client")
    async def test_not_found_returns_none(self, mk_client):
        client = MagicMock()
        mk_client.return_value = client
        client.containers.get.side_effect = NotFound("gone")
        assert await DockerSandbox.get_strict("x") is None

    @patch.object(DockerSandbox, "_create_docker_client")
    async def test_api_error_raises_daemon_unreachable(self, mk_client):
        client = MagicMock()
        mk_client.return_value = client
        client.containers.get.side_effect = APIError("daemon down")
        with pytest.raises(SandboxDaemonUnreachable):
            await DockerSandbox.get_strict("x")

    @patch.object(
        DockerSandbox, "_create_docker_client", side_effect=RuntimeError("no socket")
    )
    async def test_client_failure_raises_daemon_unreachable(self, mk_client):
        with pytest.raises(SandboxDaemonUnreachable):
            await DockerSandbox.get_strict("x")

    @patch.object(DockerSandbox, "_create_docker_client")
    async def test_exited_container_returns_none(self, mk_client):
        """FIX-K: a container that exists but is ``exited`` is terminal → None
        (must NOT be treated as a live sandbox to rehydrate)."""
        client = MagicMock()
        mk_client.return_value = client
        client.containers.get.return_value = _mk_container(status="exited")
        assert await DockerSandbox.get_strict("x") is None

    @patch.object(DockerSandbox, "_create_docker_client")
    async def test_paused_container_returns_none(self, mk_client):
        """FIX-K: a ``paused`` container is not ``running`` → terminal → None."""
        client = MagicMock()
        mk_client.return_value = client
        client.containers.get.return_value = _mk_container(status="paused")
        assert await DockerSandbox.get_strict("x") is None

    @patch.object(DockerSandbox, "_create_docker_client")
    async def test_running_without_ip_returns_none(self, mk_client):
        """FIX-K: ``running`` but no reachable IP (empty NetworkSettings) → None —
        an addressless container cannot be dialed, so it is treated as gone."""
        client = MagicMock()
        mk_client.return_value = client
        client.containers.get.return_value = _mk_container(status="running", ip=None)
        assert await DockerSandbox.get_strict("x") is None


class TestListAndRemove:
    @patch.object(DockerSandbox, "_create_docker_client")
    def test_list_managed_filters_by_label(self, mk_client):
        client = MagicMock()
        mk_client.return_value = client
        client.containers.list.return_value = [
            _mk_container(
                name="sb-1", labels={"actus.session_id": "s1", "actus.attempt": "a1"}
            ),
        ]
        rows = DockerSandbox.list_managed_containers()
        client.containers.list.assert_called_once_with(
            all=True, filters={"label": "actus.session_id"}
        )
        assert rows[0]["session_id"] == "s1" and rows[0]["name"] == "sb-1"

    @patch.object(DockerSandbox, "_create_docker_client")
    def test_list_managed_raises_daemon_unreachable_on_api_error(self, mk_client):
        """Fail-safe contract Task 6's reconcile depends on: a daemon blip while
        enumerating (``containers.list`` raising ``APIError``) MUST surface as
        ``SandboxDaemonUnreachable`` — never a silent ``[]`` that the label-sweep
        would misread as "no managed containers" and act on. This locks the raise."""
        client = MagicMock()
        mk_client.return_value = client
        client.containers.list.side_effect = APIError("boom")
        with pytest.raises(SandboxDaemonUnreachable):
            DockerSandbox.list_managed_containers()

    @patch.object(DockerSandbox, "_create_docker_client")
    def test_remove_container_swallows_not_found(self, mk_client):
        client = MagicMock()
        mk_client.return_value = client
        client.containers.get.side_effect = NotFound("gone")
        DockerSandbox.remove_container("sb-x")  # 不抛

    def test_list_managed_returns_empty_when_sandbox_address_set(self, monkeypatch):
        # external sandbox_address 模式：没有本地 Docker daemon 可枚举 → []。
        # 照抄本文件所在目录既有的 settings-patch 模式（monkeypatch
        # ``ds_mod.get_settings``；见 test_docker_sandbox.py / runtime_policy 测试）。
        monkeypatch.setattr(
            ds_mod,
            "get_settings",
            lambda: SimpleNamespace(sandbox_address="http://sandbox:8080"),
        )
        assert DockerSandbox.list_managed_containers() == []


class TestDestroyAcloseResilience:
    """FIX-I: an httpx ``client.aclose()`` failure inside ``destroy()`` must NOT
    skip the container removal. Before the fix, ``aclose`` lived inside the main
    ``try`` → any error jumped straight to the ``except`` (return False), leaking
    the container. Now aclose is best-effort; the return contract reflects
    container-removal truth (aclose fail + successful remove → True)."""

    @patch.object(DockerSandbox, "_create_docker_client")
    async def test_aclose_failure_still_removes_container(self, mk_client):
        docker_client = MagicMock()
        mk_client.return_value = docker_client
        container = _mk_container()
        docker_client.containers.get.return_value = container

        sandbox = DockerSandbox(ip="10.0.0.2", container_name="c1")
        # inject a fake httpx client whose aclose raises
        sandbox.client = MagicMock()
        sandbox.client.aclose = AsyncMock(side_effect=RuntimeError("aclose boom"))

        result = await sandbox.destroy()

        assert result is True
        container.remove.assert_called_once_with(force=True)
