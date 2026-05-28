"""[C2 PR-9 §15.2 Gate #3] ``PatchReducerService.reduce`` must stay pure —
no publisher / sandbox writer / DB writer / patch applier imports.

The reducer is responsible only for deciding the group-level outcome from
per-worker results plus a drift probe (read-only sandbox access). Side
effects belong downstream: mailbox publishing in the supervisor,
parent-sandbox writes in ``PatchApplier``, audit writes in the orchestrator.
This gate AST-scans ``patch_reducer_service.py`` for imports of the
forbidden surfaces so a reviewer can't accidentally inline a write call.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.coordinator_pure


# Resolve repo-relative ``api/app/...`` paths from this test file's location so
# the gate works regardless of pytest's CWD (``api/`` vs repo root).
_API_ROOT = Path(__file__).resolve().parents[3]
_REDUCER = _API_ROOT / "app" / "application" / "services" / "patch_reducer_service.py"


def test_reducer_no_forbidden_imports():
    src = _REDUCER.read_text()
    tree = ast.parse(src)
    forbidden = {"MailboxPublisher", "ParentSandboxAdapter", "DbCoordinatorApplyAuditRepository",
                  "patch_applier"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in forbidden or node.module and any(f in (node.module or "") for f in forbidden):
                    found.add(alias.name)
    assert not found
