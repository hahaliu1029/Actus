"""I8: Recovery package MUST NOT import graphs/context/application.

All compact-budget decisions live in the Runner callback (PlannerReActFlow).
_actions.py is the highest-risk file: TriggerRecompact could easily get
"convenience" imports of GradualCompactor or resolve_context_window — this
gate catches that at PR review time.

Audit Round 2 P2 #4 expanded coverage: cover every _*.py in the recovery
package so PR-2 additions (_actions, _rules) and PR-3 additions (_event)
don't slip past the guard.
"""
import ast
from pathlib import Path

import pytest


RECOVERY_MODULES = [
    "app/domain/services/recovery/_base.py",
    "app/domain/services/recovery/_registry.py",
    # PR-2 additions — highest risk (TriggerRecompact could import compactor)
    "app/domain/services/recovery/_actions.py",
    "app/domain/services/recovery/_rules.py",
    # PR-3 addition
    "app/domain/services/recovery/_event.py",
]


# Modules skipped until created — later PRs un-skip via pytest.importorskip
_MODULE_SKIP_UNTIL_PR = {
    "app/domain/services/recovery/_actions.py": "PR-2",
    "app/domain/services/recovery/_rules.py": "PR-2",
    "app/domain/services/recovery/_event.py": "PR-3",
}

FORBIDDEN_PREFIXES = (
    "app.domain.services.graphs",
    "app.domain.services.context",
    "app.application",
)


def _imports(relpath: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[5]
    api_root = repo_root / "api"
    src = (api_root / relpath).read_text()
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


@pytest.mark.parametrize("module_path", RECOVERY_MODULES)
def test_recovery_modules_have_no_forbidden_imports(module_path: str):
    repo_root = Path(__file__).resolve().parents[5]
    api_root = repo_root / "api"
    full_path = api_root / module_path
    if not full_path.exists():
        # Created in a later PR (see _MODULE_SKIP_UNTIL_PR). Skip until then.
        expected_pr = _MODULE_SKIP_UNTIL_PR.get(module_path, "later PR")
        pytest.skip(f"{module_path} not yet created — part of {expected_pr}")
    for imp in _imports(module_path):
        for forbidden in FORBIDDEN_PREFIXES:
            assert not imp.startswith(forbidden), (
                f"{module_path} imports {imp!r} which violates I8 "
                f"(must not depend on {forbidden})"
            )
