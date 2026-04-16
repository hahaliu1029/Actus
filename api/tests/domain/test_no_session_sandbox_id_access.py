"""CI Gate 2: No direct session.sandbox_id access in domain/application layers.

Enforces I8 (SandboxBinding as Session aggregate sub-object). All sandbox
state access must go through session.sandbox_binding.*, never through the
raw session.sandbox_id field.

Exception (spec §9.5 gate 2):
- db_session_repository.py — ORM mapping layer
- infrastructure/models/session.py — ORM column definition
"""
from __future__ import annotations

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[2] / "app"

SCAN_DIRS = [
    API_ROOT / "domain",
    API_ROOT / "application",
]

EXCEPTION_FILES = frozenset({
    "db_session_repository.py",
})

EXCEPTION_PATHS = frozenset({
    "infrastructure/models/session.py",
    "infrastructure/models/__pycache__",
})


def _collect_python_files() -> list[Path]:
    files = []
    for scan_dir in SCAN_DIRS:
        if scan_dir.exists():
            files.extend(scan_dir.rglob("*.py"))
    return files


def _is_exception(path: Path) -> bool:
    if path.name in EXCEPTION_FILES:
        return True
    rel = str(path.relative_to(API_ROOT))
    return any(exc in rel for exc in EXCEPTION_PATHS)


def _check_file_for_sandbox_id_access(path: Path) -> list[str]:
    """Return violations: attribute access to .sandbox_id on likely session objects."""
    if _is_exception(path):
        return []

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr != "sandbox_id":
            continue
        # Check if accessing on a variable likely named 'session'
        if isinstance(node.value, ast.Name) and node.value.id == "session":
            violations.append(
                f"{path.relative_to(API_ROOT.parent)}:{node.lineno} "
                f"accesses session.sandbox_id directly"
            )
        elif isinstance(node.value, ast.Attribute) and node.value.attr == "session":
            violations.append(
                f"{path.relative_to(API_ROOT.parent)}:{node.lineno} "
                f"accesses *.session.sandbox_id directly"
            )

    return violations


def test_no_session_sandbox_id_access() -> None:
    """Domain and application code must not access session.sandbox_id directly.

    Use session.sandbox_binding.id (or .state, .generation, etc.) instead.
    See spec §9.5 gate 2 and I8 invariant.
    """
    files = _collect_python_files()
    assert files, "No Python files found to scan — check SCAN_DIRS"

    all_violations = []
    for path in files:
        all_violations.extend(_check_file_for_sandbox_id_access(path))

    assert not all_violations, (
        f"Found {len(all_violations)} direct session.sandbox_id access(es):\n"
        + "\n".join(f"  - {v}" for v in all_violations)
        + "\n\nUse session.sandbox_binding.* instead. See spec §9.5 gate 2."
    )
