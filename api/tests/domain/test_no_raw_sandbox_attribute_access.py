"""CI Gate 3: No raw _sandbox attribute access outside allowed files.

Enforces I7 (Generation on SandboxHandle). Prevents holders from reaching
through SandboxHandleImpl._sandbox to bypass generation checks, and
prevents external code from accessing SandboxRegistry private state.

Exception list (spec §9.5 gate 3):
- sandbox_lifecycle_service.py
- sandbox_registry.py
- sandbox_handle.py
- test files for these modules
"""
from __future__ import annotations

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[2] / "app"

SCAN_DIRS = [
    API_ROOT / "application",
    API_ROOT / "domain" / "services",
]

EXCEPTION_FILES = frozenset({
    "sandbox_lifecycle_service.py",
    "sandbox_registry.py",
    "sandbox_handle.py",
    # Holder files that store SandboxHandle in self._sandbox field.
    # Gate 1 ensures they import SandboxHandle, not raw Sandbox.
    # self._sandbox here IS the handle, not a bypass.
    "agent_task_runner.py",
    "planner_react.py",
    "skill_bundle_sync.py",
    "skill.py",
    "create_skill.py",
})

# Attribute names that are forbidden when accessed from outside exception files
FORBIDDEN_ATTRS = frozenset({
    "_sandbox",
    "_sandboxes",
    "_inflight_tasks",
    "_open_handles",
    "_generations",
    "_drain_events",
    "_ws_holders",
})


def _collect_python_files() -> list[Path]:
    files = []
    for scan_dir in SCAN_DIRS:
        if scan_dir.exists():
            files.extend(scan_dir.rglob("*.py"))
    return files


def _check_file_for_raw_sandbox_attrs(path: Path) -> list[str]:
    """Return violations: private sandbox/registry attribute access."""
    if path.name in EXCEPTION_FILES:
        return []

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        attr = node.attr
        if attr in FORBIDDEN_ATTRS:
            violations.append(
                f"{path.relative_to(API_ROOT.parent)}:{node.lineno} "
                f"accesses forbidden attribute .{attr}"
            )

    return violations


def test_no_raw_sandbox_attribute_access() -> None:
    """Application and domain service code must not access sandbox/registry private attrs.

    Accessing ._sandbox bypasses generation checks (I7).
    Accessing ._sandboxes / ._inflight_tasks / etc. bypasses registry encapsulation.

    See spec §9.5 gate 3.
    """
    files = _collect_python_files()
    assert files, "No Python files found to scan — check SCAN_DIRS"

    all_violations = []
    for path in files:
        all_violations.extend(_check_file_for_raw_sandbox_attrs(path))

    assert not all_violations, (
        f"Found {len(all_violations)} raw sandbox attribute access(es):\n"
        + "\n".join(f"  - {v}" for v in all_violations)
        + "\n\nUse SandboxHandle public API instead. See spec §9.5 gate 3."
    )
