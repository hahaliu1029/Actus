from __future__ import annotations

import ast
import inspect

import app.application.services.skill_creator_service as scs


def test_skill_creator_temp_sandbox_uses_compile_runtime_policy():
    """C5c INV-0/coverage: the skill-validation temp sandbox must be created with
    runtime_policy=compile_runtime_policy(get_settings()) so flag-ON hardens the
    generated-code sandbox and flag-OFF (helper → None) stays byte-identical.
    AST-assert the call site (driving the full generator is integration, CI-only)."""
    tree = ast.parse(inspect.getsource(scs))
    sites = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "create"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "DockerSandbox"
    ]
    assert sites, "no DockerSandbox.create(...) call found in skill_creator_service"
    for call in sites:
        kw = {k.arg: k.value for k in call.keywords}
        assert "runtime_policy" in kw, "DockerSandbox.create must pass runtime_policy (C5c)"
        rp = kw["runtime_policy"]
        assert isinstance(rp, ast.Call) and getattr(rp.func, "id", None) == "compile_runtime_policy", (
            "runtime_policy must be compile_runtime_policy(...)"
        )
