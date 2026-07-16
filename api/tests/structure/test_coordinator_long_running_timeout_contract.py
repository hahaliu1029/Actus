"""Coordinator long-running timeout invariants.

This test intentionally uses AST/source inspection instead of constructing the
composition root.  It is a cheap CI guard against turning a rolling liveness or
cleanup lease back into a fixed task deadline.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
API_APP = REPO_ROOT / "api" / "app"


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _literal_class_default(class_node: ast.ClassDef, field_name: str) -> object:
    field = next(
        node for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == field_name
    )
    assert field.value is not None
    return ast.literal_eval(field.value)


def _kwonly_default(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_name: str,
) -> ast.expr | None:
    defaults = dict(zip(method.args.kwonlyargs, method.args.kw_defaults, strict=True))
    return next(default for argument, default in defaults.items() if argument.arg == parameter_name)


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_default_wallclock_budgets_are_unlimited() -> None:
    limits = _class(
        _tree("api/app/domain/services/coordinator_limits.py"),
        "CoordinatorLimits",
    )
    assert _literal_class_default(limits, "max_wallclock_seconds_per_child") == 0
    assert _literal_class_default(limits, "max_total_wallclock_seconds_per_run") == 0

    execution = _class(
        _tree("api/app/domain/models/app_config.py"),
        "ExecutionConfig",
    )
    total_timeout = next(
        node for node in execution.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "total_timeout_seconds"
    )
    assert isinstance(total_timeout.value, ast.Call)
    default = next(
        keyword.value for keyword in total_timeout.value.keywords
        if keyword.arg == "default"
    )
    assert ast.literal_eval(default) == 0


def test_waiter_and_orchestrator_have_no_implicit_deadline() -> None:
    waiter = _class(
        _tree("api/app/application/services/coordinator_terminal_envelope_waiter.py"),
        "CoordinatorTerminalEnvelopeWaiter",
    )
    waiter_default = _kwonly_default(_method(waiter, "await_terminal"), "timeout")
    assert isinstance(waiter_default, ast.Constant) and waiter_default.value is None

    orchestrator = _class(
        _tree("api/app/application/services/coordinator_run_orchestrator.py"),
        "CoordinatorRunOrchestrator",
    )
    run_default = _kwonly_default(_method(orchestrator, "run"), "timeout_seconds")
    assert isinstance(run_default, ast.Constant) and run_default.value is None


def test_child_wallclock_watchdog_only_starts_for_positive_budget() -> None:
    tree = _tree("api/app/application/services/coordinator_child_runner.py")
    start_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "start_wallclock_watchdog"
    ]
    assert len(start_calls) == 1

    positive_guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_max_wc > 0"
    ]
    assert len(positive_guards) == 1
    assert start_calls[0] in set(ast.walk(positive_guards[0]))


def test_coordinator_child_has_external_terminal_and_heartbeat_owners() -> None:
    source = (
        REPO_ROOT / "api/app/application/services/child_agent_runner_factory.py"
    ).read_text(encoding="utf-8")
    expected = "tool_filter_preset == COORDINATOR_STEP_PRESET"
    assert f"external_terminal_owner = {expected}" in source
    assert f"external_heartbeat_owner = {expected}" in source


def test_sandbox_activity_resets_cleanup_window_instead_of_extending_it() -> None:
    tree = _tree("sandbox/app/core/middleware.py")
    method_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "reset_timeout" in method_calls
    assert "extend_timeout" not in method_calls

    docker_source = (
        REPO_ROOT / "api/app/infrastructure/external/sandbox/docker_sandbox.py"
    ).read_text(encoding="utf-8")
    assert '"SERVER_TIMEOUT_MINUTES"' in docker_source
    assert '"SERVICE_TIMEOUT_MINUTES"' not in docker_source


def test_ordinary_probe_quota_is_scoped_to_exact_run_id() -> None:
    quota = _class(
        _tree("api/app/infrastructure/cache/probe_quota.py"),
        "ProbeQuotaService",
    )
    for method_name in ("acquire", "renew", "release"):
        method = _method(quota, method_name)
        positional = [argument.arg for argument in method.args.args]
        assert positional == ["self", "user_id", "probe_run_id"]
        assert not method.args.defaults


def test_domain_does_not_add_new_framework_or_infrastructure_imports() -> None:
    """Pin existing layering debt while forbidding it from spreading.

    The allowlist is pre-existing debt outside this timeout change.  Removing an
    entry from production is allowed; adding a new one fails this gate.
    """
    allowed: dict[str, set[str]] = {
        "domain/external/file_storage.py": {"fastapi"},
        "domain/external/observability.py": {
            "app.infrastructure.observability.context",
        },
        "domain/services/flows/planner_react.py": {
            "app.infrastructure.external.llm.actus_recovery_chat_model",
        },
        "domain/services/graphs/react_graph.py": {
            "app.infrastructure.external.llm.message_sanitizer",
        },
        "domain/services/tools/memory_tools.py": {"sqlalchemy.ext.asyncio"},
        "domain/services/agent_task_runner.py": {
            "app.infrastructure.external.embedding.openai_embedding_provider",
            "app.infrastructure.external.embedding.redis_embedding_cache",
            "app.infrastructure.external.embedding.skill_embedding_index",
            "app.infrastructure.external.file_view.image_bytes_resolver",
            "app.infrastructure.external.llm.message_sanitizer",
            "app.infrastructure.external.memory.redis_recall_cache",
            "app.infrastructure.repositories.db_user_tool_enablement_repository",
            "app.infrastructure.repositories.file_skill_repository",
            "app.infrastructure.storage.postgres",
            "app.infrastructure.storage.redis",
            "app.infrastructure.telemetry.prompt_telemetry",
            "fastapi",
        },
    }
    violations: list[str] = []
    for path in (API_APP / "domain").rglob("*.py"):
        relative = path.relative_to(API_APP).as_posix()
        for module in _imported_modules(path):
            forbidden = (
                module in {"fastapi", "sqlalchemy"}
                or module.startswith(("fastapi.", "sqlalchemy."))
                or module.startswith(("app.infrastructure", "app.interfaces"))
            )
            if forbidden and module not in allowed.get(relative, set()):
                violations.append(f"{relative}: {module}")
    assert not violations, "new domain layering violations:\n" + "\n".join(violations)
