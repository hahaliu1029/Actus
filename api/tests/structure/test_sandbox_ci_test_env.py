"""Keep standalone sandbox pytest jobs collection-safe in CI."""

from pathlib import Path

import pytest
import yaml


pytestmark = [pytest.mark.structure]

_CI_YML = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"
_SANDBOX_JOBS = ("sandbox-adversarial", "sandbox-hardening-smoke")


@pytest.mark.parametrize("job_name", _SANDBOX_JOBS)
def test_sandbox_pytest_job_exports_test_settings(job_name: str) -> None:
    workflow = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    job = workflow["jobs"][job_name]
    pytest_steps = [
        step
        for step in job["steps"]
        if "pytest tests/sandbox/" in str(step.get("run", ""))
    ]

    assert len(pytest_steps) == 1
    env = pytest_steps[0].get("env", {})
    assert env.get("ENV") == "test"
    jwt_secret = env.get("JWT_SECRET_KEY")
    assert jwt_secret and jwt_secret != "change-me-in-env"


def test_hardening_job_exports_real_sandbox_runtime_settings() -> None:
    workflow = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    job = workflow["jobs"]["sandbox-hardening-smoke"]
    pytest_step = next(
        step
        for step in job["steps"]
        if "pytest tests/sandbox/" in str(step.get("run", ""))
    )
    env = pytest_step.get("env", {})

    assert env.get("SANDBOX_IMAGE") == "actus-sandbox:latest"
    assert env.get("SANDBOX_NAME_PREFIX") == "actus-sb"
    assert env.get("MEMORY_ROOT_HOST") == env.get("MEMORY_ROOT_CONTAINER")
    assert "runner.temp" in env.get("MEMORY_ROOT_HOST", "")
