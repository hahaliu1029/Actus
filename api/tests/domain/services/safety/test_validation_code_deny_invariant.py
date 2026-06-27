"""C5b §8.8 (INV-5) — deny-set drift guard.

Every production ValidationResult(...) constructor in shell_ast_validator.py must
declare LITERAL allowed/code kwargs satisfying `allowed == (code == "ok")`, and
DENY_VALIDATION_CODES must equal the set of all non-ok ValidationCode literals.

A future "warn" code (allowed=True, non-ok), or a computed allowed/code that could
hide one, fails this guard — forcing deny-set reconciliation before the always-on
gate can mis-decide. (Verified at spec time: 13 literal constructors, all consistent.)
"""
from __future__ import annotations

import ast
import pathlib
from typing import get_args

from app.domain.services.safety.shell_ast_validator import (
    DENY_VALIDATION_CODES,
    ValidationCode,
)


def _validator_src() -> str:
    # Robust ancestor search — this file is at api/tests/domain/services/safety/, so a
    # fixed parents[N] is fragile (parents[3] here = api/tests, not the repo root). Mirror
    # the resolve() pattern in test_sandbox_policy_domain_purity.py:20. [codex planR1 P1]
    rel = ("api", "app", "domain", "services", "safety", "shell_ast_validator.py")
    for ancestor in pathlib.Path(__file__).resolve().parents:
        candidate = ancestor.joinpath(*rel)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError("could not locate shell_ast_validator.py")


def test_deny_set_equals_all_non_ok_codes() -> None:
    assert set(DENY_VALIDATION_CODES) == {c for c in get_args(ValidationCode) if c != "ok"}


def test_every_validation_result_is_literal_and_consistent() -> None:
    tree = ast.parse(_validator_src())
    ctors = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ValidationResult"
    ]
    assert ctors, "expected ValidationResult(...) constructors in the validator"
    for call in ctors:
        kw = {k.arg: k.value for k in call.keywords}
        allowed_node = kw.get("allowed")
        code_node = kw.get("code")
        assert isinstance(allowed_node, ast.Constant), (
            f"ValidationResult at line {call.lineno}: `allowed` must be a literal "
            f"(found {type(allowed_node).__name__ if allowed_node is not None else 'missing'})"
        )
        assert isinstance(code_node, ast.Constant), (
            f"ValidationResult at line {call.lineno}: `code` must be a literal "
            f"(found {type(code_node).__name__ if code_node is not None else 'missing'})"
        )
        assert allowed_node.value == (code_node.value == "ok"), (
            f"ValidationResult at line {call.lineno}: allowed={allowed_node.value!r} "
            f"violates `allowed == (code == 'ok')` for code={code_node.value!r}"
        )
