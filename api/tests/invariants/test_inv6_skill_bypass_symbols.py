"""INV-6 — Skill bypass symbols grep gate.

PE-1: WARNING-ONLY. Surfaces violations so PE-1b can flip the constant
to hard-fail. Each forbidden symbol indicates code reading skill risk
metadata outside the PE source adapter — the bypass that PE-1 aims to
internalize.

Spec §5.3 — `PE_1_GRACE_MODE = True` for PE-1; PE-1b flips to False.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

from tests.invariants._whitelists import REPO_ROOT

FORBIDDEN_SYMBOLS: tuple[str, ...] = (
    "risk_level_meta",
    "SkillRiskAssessor",
    "risk_enforce",
)

# Only these files may *consume* the forbidden symbols. Anything else is a
# bypass survivor and must be migrated through the PE source adapter.
WHITELIST: frozenset[str] = frozenset({
    # PE source adapter implementation
    "api/app/domain/services/permission/sources/skill_source.py",
    "api/app/domain/services/permission/sources/skill_metadata.py",
    # PE-1 default_engine emits the "risk_enforce" string constant at step 9
    "api/app/domain/services/permission/default_engine.py",
    # SkillRiskAssessor definition itself
    "api/app/domain/services/skill_risk_assessor.py",
    # Legacy R3 branch retained for the 14-day grace period (PE-1b deletes)
    "api/app/domain/services/graphs/react_graph.py",
})

# CI grace gate — flip to False in PE-1b (single-line change + delete legacy).
PE_1_GRACE_MODE: bool = True


def _scan_file(path: Path, symbols: tuple[str, ...]) -> list[tuple[int, str]]:
    """Return (lineno, symbol) for each forbidden symbol found in `path`.

    Uses AST + raw-text scan to catch both Name/Attribute references and
    string-literal forms (e.g., the literal "risk_enforce" string at the
    PE step 9 dispatch).
    """
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    hits: list[tuple[int, str]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hits
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in symbols:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in symbols:
            hits.append((node.lineno, node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in symbols:
                hits.append((node.lineno, node.value))
    return hits


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def test_inv6_react_graph_does_not_consume_skill_bypass_symbols():
    api_app = REPO_ROOT / "api" / "app"
    violations: list[tuple[str, int, str]] = []
    for py_file in api_app.rglob("*.py"):
        rel = _relpath(py_file)
        if rel in WHITELIST:
            continue
        for lineno, symbol in _scan_file(py_file, FORBIDDEN_SYMBOLS):
            violations.append((rel, lineno, symbol))

    if violations:
        if PE_1_GRACE_MODE:
            warnings.warn(
                "INV-6 (skill bypass symbols) violations (PE-1 grace, will "
                "fail in PE-1b):\n"
                + "\n".join(
                    f"  {f}:{ln} — {s}" for f, ln, s in sorted(violations)
                ),
                stacklevel=2,
            )
        else:
            assert not violations, (
                "INV-6 violations (PE-1b hard enforce):\n"
                + "\n".join(
                    f"  {f}:{ln} — {s}" for f, ln, s in sorted(violations)
                )
            )


def test_inv6_pe_1_grace_mode_is_true_in_pe_1():
    """PE-1 ships with grace_mode=True. PE-1b flips this single line."""
    assert PE_1_GRACE_MODE is True


def test_inv6_whitelist_contents_documented():
    """Each whitelist entry encodes a deliberate consumer; reviewers
    must understand why before adding more (security decision)."""
    # Sanity: the four whitelist entries are the four we expect.
    assert WHITELIST == frozenset({
        "api/app/domain/services/permission/sources/skill_source.py",
        "api/app/domain/services/permission/sources/skill_metadata.py",
        "api/app/domain/services/permission/default_engine.py",
        "api/app/domain/services/skill_risk_assessor.py",
        "api/app/domain/services/graphs/react_graph.py",
    })
