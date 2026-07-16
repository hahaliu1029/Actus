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
