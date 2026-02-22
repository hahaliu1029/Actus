from types import SimpleNamespace

from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox


class _FakeContainer:
    def __init__(self) -> None:
        self.attrs = {"NetworkSettings": {"Networks": {"actus-net": {}}}}

    def reload(self) -> None:
        return None


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.run_kwargs: dict = {}

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self._container


class _FakeDockerClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_create_task_sets_tz_for_spawned_sandbox_container(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        sandbox_image="actus-sandbox:latest",
        sandbox_name_prefix="actus-sb",
        sandbox_ttl_minutes=60,
        sandbox_chrome_args="",
        sandbox_https_proxy=None,
        sandbox_http_proxy=None,
        sandbox_no_proxy=None,
        sandbox_network="actus-net",
        container_timezone="Asia/Shanghai",
    )
    fake_container = _FakeContainer()
    fake_docker_client = _FakeDockerClient(fake_container)

    monkeypatch.setattr(
        "app.infrastructure.external.sandbox.docker_sandbox.get_settings",
        lambda: fake_settings,
    )
    monkeypatch.setattr(
        DockerSandbox,
        "_create_docker_client",
        classmethod(lambda cls: fake_docker_client),
    )
    monkeypatch.setattr(
        DockerSandbox,
        "_wait_for_container_ip",
        classmethod(lambda cls, container, retries=20, interval_seconds=0.5: "172.18.0.2"),
    )

    sandbox = DockerSandbox._create_task()

    assert sandbox.id.startswith("actus-sb-")
    assert fake_docker_client.containers.run_kwargs["environment"]["TZ"] == "Asia/Shanghai"
    assert fake_docker_client.containers.run_kwargs["network"] == "actus-net"
    assert fake_docker_client.closed is True
