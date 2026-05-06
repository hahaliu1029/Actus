"""BR1 GroundedClick eval — Playwright + 本地静态页面。

需 Playwright 浏览器二进制：`uv run playwright install chromium`。
公网 case (18-20) 用 `@pytest.mark.network`，CI 默认排除。
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def _migrate():
    """Override the parent `tests/integration/conftest.py:_migrate` autouse fixture.

    Browser eval tests don't touch Postgres — they only need Playwright + the
    in-process GroundedClickEngine. Without this override, the parent
    `_run_migrations()` call fails when Postgres isn't reachable from the dev
    environment, blocking the eval harness entirely.
    """
    yield


@pytest.fixture(scope="session")
def eval_root() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "browser_eval"


@pytest.fixture(scope="session")
def cases(eval_root: Path) -> list[dict]:
    out: list[dict] = []
    for case_path in sorted((eval_root / "cases").glob("*.json")):
        data = json.loads(case_path.read_text(encoding="utf-8"))
        # `{eval_root}` placeholder → absolute path
        data["url"] = data["url"].replace("{eval_root}", str(eval_root))
        out.append(data)
    return out
