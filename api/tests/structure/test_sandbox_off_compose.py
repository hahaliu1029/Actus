"""SPM Task 12 — base compose 透传断言。

api service 的 ``environment`` block 是显式 whitelist：缺 passthrough 则 Docker
部署永远无法启用/传入该项。本任务断言 base ``docker-compose.yml`` 已透传
``SANDBOX_PROVISION_MODE`` + 全三个 C2 flag（现只透传 2/3——补齐
``ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED``）。

off 的 ``!reset``/``!override`` 双文件断言（Task 31 追加）必须走 subprocess
``docker compose config`` 解析——纯 ``yaml.safe_load`` 不认 compose 自定义 tag。
本任务先放 base 断言（纯 YAML），并预置 ``_compose_config_json`` / ``requires_docker``
供 Task 31 复用（本任务尚未消费，故不引入 docker 运行期依赖）。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

# api/tests/structure/<this>.py → parents[3] == worktree 仓库根（docker-compose.yml 所在）。
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_compose_api_environment() -> dict:
    doc = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    return doc["services"]["api"]["environment"]


def _compose_config_json(extra_files: list[str]) -> dict:
    """subprocess 跑 ``docker compose config --format json``（Task 31 off 双文件断言用）。

    off override 文件里的 ``!reset`` / ``!override`` 是 compose 自定义 tag，
    ``yaml.safe_load`` 会抛，必须让 compose 自己解析。
    """
    cmd = ["docker", "compose"]
    for f in extra_files:
        cmd += ["-f", f]
    cmd += ["--env-file", ".env.example", "config", "--format", "json"]
    out = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


requires_docker = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker CLI absent"
)


def test_base_compose_passes_mode_and_all_three_c2_flags():
    api_env = _load_compose_api_environment()
    assert "SANDBOX_PROVISION_MODE" in api_env
    # FIX-M1 (flip-readiness): on_demand 供给超时也必须透传，否则 Docker 部署
    # 无法调整 create/ready/hooks 总预算（只能吃 config 默认 90）。
    assert "SANDBOX_PROVISION_TIMEOUT_SECONDS" in api_env
    for flag in (
        "ACTUS_C2_COORDINATOR_ENABLED",
        "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED",
        "ACTUS_C2_AGENT_TEAMS_ENABLED",
    ):
        assert flag in api_env, f"missing {flag} passthrough"


# ---------------------------------------------------------------------------
# Task 31 — off 双文件值级验收（override + !override depends_on/volumes + profile）
# ---------------------------------------------------------------------------
# 所有断言经同一 ``docker compose config --format json`` subprocess 解析：base 与
# off 两侧走同一归一化管线，故期望值零手写（off == base 精确减两删除项），任何
# condition 降级 / 漏删 / 多删都立即变红。需 docker CLI（``@requires_docker``）。

OFF = ["docker-compose.yml", "docker-compose.sandbox-off.yml"]
BASE = ["docker-compose.yml"]

_DOCKER_SOCK = "/var/run/docker.sock"


def _api_service(config: dict) -> dict:
    return config["services"]["api"]


def _volume_is_docker_sock(vol: dict) -> bool:
    """long-form volume 条目是否指向 docker.sock（source 或 target 任一命中）。"""
    return _DOCKER_SOCK in (vol.get("source"), vol.get("target"))


@requires_docker
def test_off_config_pins_mode_off():
    """off override literal-pin：消灭「override 文件在、mode 仍默认 always」漂移。"""
    api_env = _api_service(_compose_config_json(OFF))["environment"]
    assert api_env["SANDBOX_PROVISION_MODE"] == "off"


@requires_docker
def test_off_services_exclude_sandbox_image():
    """sandbox-image 去激活 profile 生效 → off 渲染的 services 不含它。"""
    services = _compose_config_json(OFF)["services"]
    assert "sandbox-image" not in services


@requires_docker
def test_off_api_volumes_have_no_docker_sock():
    """off api volumes 已用 !override 完整替换，移除 docker.sock 挂载。"""
    volumes = _api_service(_compose_config_json(OFF))["volumes"]
    assert not any(_volume_is_docker_sock(v) for v in volumes)


@requires_docker
def test_off_depends_and_volumes_are_base_minus_exactly_the_two():
    """dict/list 级：off == base 精确减 sandbox-image 依赖 + docker.sock 卷。

    base 与 off 同经 ``docker compose config --format json``，期望值全部由 base
    派生（零手写），故同名依赖 condition 被降级 / 少删 / 多删都会立即红。
    """
    base_api = _api_service(_compose_config_json(BASE))
    off_api = _api_service(_compose_config_json(OFF))

    expected_depends = {
        name: cond
        for name, cond in base_api["depends_on"].items()
        if name != "sandbox-image"
    }
    assert off_api["depends_on"] == expected_depends

    expected_volumes = [
        v for v in base_api["volumes"] if not _volume_is_docker_sock(v)
    ]
    assert off_api["volumes"] == expected_volumes


@requires_docker
def test_base_config_still_has_sandbox_image_and_sock():
    """INV-SPM-2 对照：base 单文件仍含 sandbox-image service + docker.sock 卷。"""
    base = _compose_config_json(BASE)
    assert "sandbox-image" in base["services"]
    base_volumes = _api_service(base)["volumes"]
    assert any(_volume_is_docker_sock(v) for v in base_volumes)
