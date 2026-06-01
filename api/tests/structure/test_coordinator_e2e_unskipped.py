"""[C2 finish-core INV-F7.1] Inverse of the retired skip-honesty guard: the 3
coordinator E2E MUST NOT be skipped. Any reappearing @pytest.mark.skip = FAIL.
Marked coordinator_recovery so the coordinator-e2e CI job selects it (§5.8)."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest

pytestmark = [pytest.mark.structure, pytest.mark.coordinator_recovery]

ENUMERATED_FILES = (
    "api/tests/integration/test_coordinator_e2e_apply_rollback.py",
    "api/tests/integration/test_coordinator_e2e_3_work_units.py",
    "api/tests/integration/test_coordinator_e2e_sibling_cancel.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _has_skip(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        chain = ast.dump(dec)
        if "skip" in chain and "mark" in chain:
            return True
    return False


@pytest.mark.parametrize("rel", ENUMERATED_FILES)
def test_e2e_files_have_no_skip(rel):
    src = (_repo_root() / rel).read_text()
    tree = ast.parse(src)
    skipped = [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_") and _has_skip(node)
    ]
    assert skipped == [], f"{rel}: these tests are still skipped: {skipped}"
