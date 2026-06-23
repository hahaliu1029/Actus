"""[S2 PR-6] Fast guard: the coordinator-e2e CI job MUST export
ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED. The live shell-mode E2E reads this flag
from os.environ at call time and does NOT monkeypatch it (Task 6.2), so the CI
job env is the SOLE flag-on source. Without this guard a dropped env line would
silently coerce the E2E to typed-only and the live shell proof would vanish with
the suite still green. This unit runs in the default `cd api && uv run pytest`
(no DB/Redis/sandbox) — it only parses YAML text.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.structure]

_FLAG = "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED"
# api/tests/structure/<file> -> .../structure -> .../tests -> api -> repo root
_CI_YML = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"


def test_coordinator_e2e_job_exports_shell_mode_flag() -> None:
    assert _CI_YML.exists(), f"ci.yml missing at {_CI_YML}"
    raw = _CI_YML.read_text(encoding="utf-8")
    try:
        import yaml  # PyYAML (transitive dep); fall back to text scan if absent.
    except Exception:  # noqa: BLE001
        assert f"{_FLAG}:" in raw, (
            f"{_FLAG} not found in ci.yml — the coordinator-e2e job must export "
            "the shell-mode flag (Task 6.3)."
        )
        return

    doc = yaml.safe_load(raw)
    job = (doc.get("jobs") or {}).get("coordinator-e2e")
    assert job, "ci.yml has no `coordinator-e2e` job (renamed? update this guard)."
    steps = job.get("steps") or []
    # The flag must live on the SAME step that RUNS the E2E (`-m
    # coordinator_recovery`). GitHub Actions step env is per-step — the flag on
    # any OTHER step would NOT reach the E2E process, so scanning "any step env"
    # would let a "flag moved to an unrelated step" regression slip (codex PR-6
    # R1 P2). Identify the run step by its selector, then check ITS env.
    e2e_steps = [
        s for s in steps
        if isinstance(s, dict) and "coordinator_recovery" in str(s.get("run", ""))
    ]
    assert e2e_steps, (
        "ci.yml coordinator-e2e job has no step running `-m coordinator_recovery` "
        "(renamed/removed? update this guard)."
    )
    has_flag = any(
        str((s.get("env") or {}).get(_FLAG, "")).strip().lower()
        in {"true", "1", "yes", "on"}
        for s in e2e_steps
    )
    assert has_flag, (
        f"the coordinator-e2e job must export {_FLAG}: \"true\" on the SAME step "
        "that runs `-m coordinator_recovery` — it is the only place the live "
        "shell-mode E2E gets the flag ON (Task 6.2 does not monkeypatch it), and "
        "GitHub Actions step env does not leak across steps."
    )
