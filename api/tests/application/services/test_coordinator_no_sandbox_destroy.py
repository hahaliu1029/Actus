"""[C2 PR-9 §15.2 Gate #4] CoordinatorRunOrchestrator + PatchApplier must NOT
import ``SandboxLifecycleService`` (nor anything from the ``sandbox_lifecycle``
module path).

Rationale: per spec the coordinator MUST NOT touch sandbox lifecycle — sandbox
creation / destruction is the supervisor's job. Letting orchestrator or
applier import the lifecycle service would let a future change accidentally
call ``.destroy()`` on a parent sandbox mid-apply, breaking session affinity
guarantees. This is purely an import-surface gate; it cannot detect indirect
calls via duck-typed adapters, but those are caught by the dedicated unit
tests around each service.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.coordinator_pure


# Resolve repo-relative ``api/app/...`` paths from this test file's location so
# the gate works regardless of pytest's CWD (``api/`` vs repo root).
_API_ROOT = Path(__file__).resolve().parents[3]
_ORCHESTRATOR = _API_ROOT / "app" / "application" / "services" / "coordinator_run_orchestrator.py"
_APPLIER = _API_ROOT / "app" / "application" / "services" / "patch_applier.py"


def _check_no_lifecycle_import(path: Path):
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if "SandboxLifecycleService" in alias.name or "sandbox_lifecycle" in module:
                    return alias.name
    return None


def test_orchestrator_no_lifecycle_import():
    assert _check_no_lifecycle_import(_ORCHESTRATOR) is None


def test_applier_no_lifecycle_import():
    assert _check_no_lifecycle_import(_APPLIER) is None
