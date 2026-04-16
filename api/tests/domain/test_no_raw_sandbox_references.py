"""CI Gate 1: No raw Sandbox/DockerSandbox imports in application/domain/interfaces layers.

Enforces I3 (Single-Writer SandboxLifecycleService). All sandbox access must
go through the lifecycle service; direct Sandbox imports indicate a bypass.

Exception list (spec §9.5 gate 1):
- sandbox_lifecycle_service.py — the single writer itself
- sandbox_registry.py — internal to lifecycle service
- docker_sandbox.py — infrastructure implementation
- sandbox_handle.py — handle implementation
- skill_creator_service.py — temp_sandbox ephemeral exception (§9.3 #13)
"""
from __future__ import annotations

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[2] / "app"

SCAN_DIRS = [
    API_ROOT / "application",
    API_ROOT / "domain" / "services",
    API_ROOT / "interfaces",
]

# SandboxLifecycleService lives in application/services/ (not domain/services/)
# per Clean Architecture: it imports from infrastructure.

EXCEPTION_FILES = frozenset({
    "sandbox_lifecycle_service.py",
    "sandbox_registry.py",
    "docker_sandbox.py",
    "sandbox_handle.py",
    "skill_creator_service.py",
    # DI wiring needs DockerSandbox as the implementation class
    "service_dependencies.py",
    # AgentService needs Type[Sandbox] for sandbox_cls (fallback) during transition
    "agent_service.py",
    # AgentTaskRunner accepts SandboxHandle | Sandbox union type during transition
    "agent_task_runner.py",
})

FORBIDDEN_IMPORTS = [
    ("app.domain.external.sandbox", "Sandbox"),
    ("app.infrastructure.external.sandbox.docker_sandbox", "DockerSandbox"),
]


def _collect_python_files() -> list[Path]:
    files = []
    for scan_dir in SCAN_DIRS:
        if scan_dir.exists():
            files.extend(scan_dir.rglob("*.py"))
    return files


def _check_file_for_raw_sandbox_imports(path: Path) -> list[str]:
    """Return list of violation descriptions found in the file."""
    if path.name in EXCEPTION_FILES:
        return []

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                for forbidden_module, forbidden_name in FORBIDDEN_IMPORTS:
                    if (
                        node.module == forbidden_module
                        and alias.name == forbidden_name
                    ):
                        violations.append(
                            f"{path.relative_to(API_ROOT.parent)}:{node.lineno} "
                            f"imports {forbidden_name} from {forbidden_module}"
                        )
    return violations


def test_no_raw_sandbox_imports() -> None:
    """No file in application/domain/interfaces should import raw Sandbox or DockerSandbox.

    Use SandboxHandle (from app.domain.external.sandbox) instead.
    Access sandboxes through SandboxLifecycleService.acquire().
    """
    files = _collect_python_files()
    assert files, "No Python files found to scan — check SCAN_DIRS"

    all_violations = []
    for path in files:
        all_violations.extend(_check_file_for_raw_sandbox_imports(path))

    assert not all_violations, (
        f"Found {len(all_violations)} raw Sandbox/DockerSandbox import(s):\n"
        + "\n".join(f"  - {v}" for v in all_violations)
        + "\n\nUse SandboxHandle instead. See spec §9.5 gate 1."
    )
