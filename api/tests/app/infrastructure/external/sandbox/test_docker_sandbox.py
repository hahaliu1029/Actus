import uuid
from types import SimpleNamespace

import pytest
from docker.errors import APIError, NotFound

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
        sandbox_mem_limit="4g",
        sandbox_https_proxy=None,
        sandbox_http_proxy=None,
        sandbox_no_proxy=None,
        sandbox_network="actus-net",
        container_timezone="Asia/Shanghai",
        # NON-default value: proves the env carries the CONFIGURED value (the
        # codex-R4-F2 sync purpose), not a hardcoded default. If the production
        # code emitted a literal default instead of reading settings, this fails.
        skill_sandbox_bundle_root="/opt/custom/skills-bundle",
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
    # R10-2 / codex-R4-F2: a customized api ``skill_sandbox_bundle_root`` must be
    # propagated into the spawned container env so the SANDBOX snapshot walker
    # prunes the SAME ``.skills`` path the api writes — else the bundle-sync diff
    # pollution returns (api writes one path, sandbox excludes its default).
    assert (
        fake_docker_client.containers.run_kwargs["environment"][
            "SKILL_SANDBOX_BUNDLE_ROOT"
        ]
        == "/opt/custom/skills-bundle"
    )
    assert fake_docker_client.containers.run_kwargs["network"] == "actus-net"
    assert fake_docker_client.containers.run_kwargs["mem_limit"] == "4g"
    assert fake_docker_client.closed is True


# ── _build_memory_mount ────────────────────────────────────────────────────
# 覆盖 PR-0 新引入的 bind mount helper 所有分支：feature gate / 非法 user_id /
# target-path 构造 / mkdir 失败降级。这些分支独立于 Docker SDK，可以脱离
# _create_task 单独测，避免引入 docker 依赖。


def _mount_settings(tmp_path, *, enabled: bool = True, target: str = "/workspace/.memory"):
    """构造 _build_memory_mount 所需的最小 settings。"""
    return SimpleNamespace(
        memory_root_host=str(tmp_path / "host"),
        memory_root_container=str(tmp_path / "container"),
        sandbox_memory_mount_target=target,
        sandbox_memory_mount_enabled=enabled,
    )


def test_build_memory_mount_returns_none_when_user_id_missing(tmp_path) -> None:
    """user_id=None → 返回 None（旧 caller 不受影响）。"""
    settings = _mount_settings(tmp_path)
    assert DockerSandbox._build_memory_mount(settings, None) is None
    assert DockerSandbox._build_memory_mount(settings, "") is None


@pytest.mark.parametrize(
    "bad_user_id",
    [
        "../etc",
        "../../root",
        "/absolute/path",
        "with/slash",
        "has space",
        "has\ttab",
        "x" * 200,  # 超长
        "unicode-café",
    ],
)
def test_build_memory_mount_rejects_unsafe_user_id(tmp_path, bad_user_id) -> None:
    """非白名单字符的 user_id 必须被拒绝，防止路径穿越或注入。"""
    settings = _mount_settings(tmp_path)
    assert DockerSandbox._build_memory_mount(settings, bad_user_id) is None


def test_build_memory_mount_returns_none_when_feature_gate_off(tmp_path) -> None:
    """``sandbox_memory_mount_enabled=False`` 时即使 user_id 合法也不挂载。

    PR-0 引入默认 False，M1 PR-6A 起默认翻为 True；本测试显式传 False 验证
    降级路径——部署时 host 端 MEMORY_ROOT_HOST 未就绪时部署者可临时关闭。
    """
    settings = _mount_settings(tmp_path, enabled=False)
    user_id = str(uuid.uuid4())
    assert DockerSandbox._build_memory_mount(settings, user_id) is None


def test_build_memory_mount_rejects_non_absolute_roots(tmp_path) -> None:
    """host/container root 不是绝对路径时直接拒绝。

    回归 codex round-4：``~/.actus/memory`` 在 api 容器里会被误展开成
    ``/root/...``，然后作为宿主机 bind source 传给 Docker daemon。
    """
    settings = SimpleNamespace(
        memory_root_host="~/.actus/memory",
        memory_root_container="/app/data/memory",
        sandbox_memory_mount_target="/workspace/.memory",
        sandbox_memory_mount_enabled=True,
    )
    user_id = str(uuid.uuid4())
    assert DockerSandbox._build_memory_mount(settings, user_id) is None


def test_build_memory_mount_builds_readonly_bind_when_enabled(tmp_path) -> None:
    """feature gate 开 + 合法 user_id → 返回 read-only bind mount。

    - source = ${memory_root_host}/{user_id}（宿主机路径，docker daemon 视角）
    - target = sandbox_memory_mount_target（sandbox 内固定路径，M0 spike 约定）
    - target **不带** user_id 后缀，因为 sandbox 容器本身就是 per-user
    - read_only=True，agent 只读取 memory，写入由 api 容器侧的 FsMemoryWriter 负责
    - api 容器视角的 ${memory_root_container}/{user_id} 应被 mkdir 出来
    """
    settings = _mount_settings(tmp_path, target="/workspace/.memory")
    user_id = str(uuid.uuid4())

    mount = DockerSandbox._build_memory_mount(settings, user_id)

    assert mount is not None
    # docker.types.Mount spec dict 的字段名大写，见 docker-py Mount.__init__
    assert mount["Target"] == "/workspace/.memory"
    assert mount["Source"] == str(tmp_path / "host" / user_id)
    assert mount["Type"] == "bind"
    assert mount["ReadOnly"] is True

    # mkdir 已生效，container 侧的 user 子目录存在
    assert (tmp_path / "container" / user_id).is_dir()


def test_build_memory_mount_returns_none_when_mkdir_fails(
    tmp_path, monkeypatch
) -> None:
    """mkdir 抛 OSError → 降级为不挂载，避免把更难诊断的错误甩给 Docker daemon。"""
    settings = _mount_settings(tmp_path)
    user_id = str(uuid.uuid4())

    def _boom(self, *args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr("pathlib.Path.mkdir", _boom)

    assert DockerSandbox._build_memory_mount(settings, user_id) is None


def test_build_memory_mount_honors_custom_target(tmp_path) -> None:
    """sandbox_memory_mount_target 必须被使用——不能硬编码 /workspace/.memory。"""
    settings = _mount_settings(tmp_path, target="/mnt/memory")
    user_id = str(uuid.uuid4())

    mount = DockerSandbox._build_memory_mount(settings, user_id)

    assert mount is not None
    assert mount["Target"] == "/mnt/memory"


# ── destroy() NotFound contract (C3 PR-1 codex round 10 P2) ───────────────
# Externally removed containers (NotFound) must be treated as terminal
# success, not retryable failure. Without this, the registry would
# translate False → SandboxLifecycleError and loop forever in DESTROYING.


class _NotFoundContainers:
    """Fake docker_client.containers that raises NotFound on get()."""

    def get(self, name):  # noqa: ANN001
        raise NotFound(f"container {name} not found")


class _NotFoundDockerClient:
    def __init__(self) -> None:
        self.containers = _NotFoundContainers()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _APIErrorContainers:
    """Fake docker_client.containers that raises APIError on get()."""

    def get(self, name):  # noqa: ANN001
        raise APIError("docker daemon down")


class _APIErrorDockerClient:
    def __init__(self) -> None:
        self.containers = _APIErrorContainers()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RemoveNotFoundContainer:
    """Container whose remove(force=True) raises NotFound (race condition)."""

    def remove(self, force: bool = False) -> None:  # noqa: ARG002
        raise NotFound("container vanished mid-remove")


class _RemoveNotFoundContainers:
    def __init__(self) -> None:
        self._container = _RemoveNotFoundContainer()

    def get(self, name):  # noqa: ANN001, ARG002
        return self._container


class _RemoveNotFoundDockerClient:
    def __init__(self) -> None:
        self.containers = _RemoveNotFoundContainers()
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_destroy_returns_true_when_container_already_gone(monkeypatch) -> None:
    """C3 PR-1 (codex round 10 P2) — NotFound is terminal success.

    When ``containers.get(name)`` raises ``NotFound`` (externally removed
    container), ``destroy()`` returns ``True`` so the registry treats it
    as idempotent terminal success instead of a retryable failure that
    would loop forever in DESTROYING.
    """
    fake_docker_client = _NotFoundDockerClient()
    monkeypatch.setattr(
        DockerSandbox,
        "_create_docker_client",
        classmethod(lambda cls: fake_docker_client),
    )
    sandbox = DockerSandbox(ip="127.0.0.1", container_name="actus-sb-gone")

    result = await sandbox.destroy()

    assert result is True
    assert fake_docker_client.closed is True


@pytest.mark.anyio
async def test_destroy_returns_true_when_remove_races_with_external_delete(
    monkeypatch,
) -> None:
    """C3 PR-1 (codex round 10 P2) — NotFound on ``container.remove()``
    (race between ``get()`` and ``remove()``) is also terminal success.
    """
    fake_docker_client = _RemoveNotFoundDockerClient()
    monkeypatch.setattr(
        DockerSandbox,
        "_create_docker_client",
        classmethod(lambda cls: fake_docker_client),
    )
    sandbox = DockerSandbox(ip="127.0.0.1", container_name="actus-sb-race")

    result = await sandbox.destroy()

    assert result is True
    assert fake_docker_client.closed is True


@pytest.mark.anyio
async def test_destroy_returns_false_on_genuine_docker_error(monkeypatch) -> None:
    """C3 PR-1 (codex round 10 P2) — only ``NotFound`` is success; genuine
    errors (e.g. APIError when daemon down) must still return ``False`` so
    the registry surfaces ``SandboxLifecycleError`` for retry.
    """
    fake_docker_client = _APIErrorDockerClient()
    monkeypatch.setattr(
        DockerSandbox,
        "_create_docker_client",
        classmethod(lambda cls: fake_docker_client),
    )
    sandbox = DockerSandbox(ip="127.0.0.1", container_name="actus-sb-broken")

    result = await sandbox.destroy()

    assert result is False
    assert fake_docker_client.closed is True


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"
