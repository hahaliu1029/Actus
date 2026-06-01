"""INV-6 — Skill bypass symbols grep gate (PE-1b hard enforce).

PE-1 shipped this gate in warning-only mode; PE-1b (this revision) deletes the
legacy R3 Skill Stage P branch from react_graph.py and flips the gate to a hard
fail. The gate enforces a per-file, per-symbol allow-map (``ALLOW``) instead of a
whole-file whitelist:

  - The PE source adapter + skill-risk owner files may use any skill risk symbol.
  - ``tool_result.py`` may name the ``risk_enforce`` ``DecisionReason`` type.
  - ``react_graph.py`` may use ``risk_level_meta`` / ``risk_enforce`` for its
    NATIVE legacy fallback gate ONLY. The skill-only symbols (see
    ``REACT_GRAPH_SKILL_BANNED``) are hard-banned there, so a reintroduced R3
    skill branch is caught. react_graph.py's narrow allowance is removed entirely
    in PE-2/PE-3, when the whole-batch fallback that still needs the native gate
    is retired.

Spec §5.3 (PE-1 warning-only → PE-1b hard fail).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.invariants._whitelists import REPO_ROOT

FORBIDDEN_SYMBOLS: tuple[str, ...] = (
    "risk_level_meta",
    "SkillRiskAssessor",
    "risk_enforce",
)

# PE-1b: skill-only bypass symbols that react_graph.py must NEVER consume.
# Unlike risk_level_meta / risk_enforce — which the native legacy fallback gate
# still uses legitimately until PE-2/PE-3 retire the whole-batch fallback — these
# four are unambiguously reads of *skill* risk metadata. Their reappearance in
# react_graph.py means the deleted R3 Skill Stage P branch has crept back.
REACT_GRAPH_SKILL_BANNED: tuple[str, ...] = (
    "SkillRiskAssessor",
    "runtime_type",
    "trust_origin",
    "refresh_risk_if_stale",
)

_REACT_GRAPH_REL = "api/app/domain/services/graphs/react_graph.py"

# Per-file, per-symbol allow-map. A forbidden symbol in any other api/app file —
# OR a non-allowed forbidden symbol inside an allow-listed file — is a violation.
# Each entry is a deliberate security decision; widening it re-admits a bypass.
ALLOW: dict[str, frozenset[str]] = {
    # Each file is allowed ONLY the forbidden symbols it actually consumes today
    # (per-symbol, not whole-file). Adding any other forbidden symbol to one of
    # these files — e.g. SkillRiskAssessor to default_engine.py — fails INV-6 and
    # forces a security review (codex per feedback_pr_boundary_codex_audit.md).
    #
    # PE source adapter — internalizes skill risk; instantiates SkillRiskAssessor.
    "api/app/domain/services/permission/sources/skill_source.py": frozenset(
        {"SkillRiskAssessor"}
    ),
    # Skill metadata helper — audited PE skill-risk owner; consumes none today
    # (only a docstring mention of SkillRiskAssessor, which the AST scan ignores).
    "api/app/domain/services/permission/sources/skill_metadata.py": frozenset(),
    # DefaultEngine emits the "risk_enforce" reason string at its ask step.
    "api/app/domain/services/permission/default_engine.py": frozenset(
        {"risk_enforce"}
    ),
    # SkillRiskAssessor definition file — the class name is a ClassDef (not a
    # Name/Attribute/Constant), so it scans clean; consumes none today.
    "api/app/domain/services/skill_risk_assessor.py": frozenset(),
    # DecisionReason type definition — legitimately names the risk_enforce reason.
    "api/app/domain/models/tool_result.py": frozenset({"risk_enforce"}),
    # NATIVE legacy fallback gate ONLY — risk_level_meta + risk_enforce for native
    # tools. SkillRiskAssessor stays banned (R3 deleted). Removed in PE-2/PE-3.
    _REACT_GRAPH_REL: frozenset({"risk_level_meta", "risk_enforce"}),
}


def _scan_file(path: Path, symbols: tuple[str, ...]) -> list[tuple[int, str]]:
    """Return (lineno, symbol) for each forbidden symbol found in `path`.

    AST scan catching Name/Attribute references and string-literal forms
    (e.g., the literal "risk_enforce" string at the PE step 9 dispatch).
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


def test_inv6_no_skill_bypass_symbols_outside_allow():
    """PE-1b hard enforce: any forbidden symbol outside its per-file allowance
    fails. Each bypass survivor indicates code reading skill risk metadata
    outside the PE source adapter."""
    api_app = REPO_ROOT / "api" / "app"
    violations: list[tuple[str, int, str]] = []
    for py_file in api_app.rglob("*.py"):
        rel = _relpath(py_file)
        allowed = ALLOW.get(rel, frozenset())
        for lineno, symbol in _scan_file(py_file, FORBIDDEN_SYMBOLS):
            if symbol in allowed:
                continue
            violations.append((rel, lineno, symbol))

    assert not violations, (
        "INV-6 violations (PE-1b hard enforce):\n"
        + "\n".join(f"  {f}:{ln} — {s}" for f, ln, s in sorted(violations))
    )


def test_inv6_react_graph_allowance_is_native_only():
    """react_graph.py's allowance is the NATIVE fallback subset only: it must
    NOT exempt SkillRiskAssessor (the R3 skill branch is deleted). Guards
    against someone widening the allowance to re-admit a skill bypass."""
    allowed = ALLOW[_REACT_GRAPH_REL]
    assert "SkillRiskAssessor" not in allowed
    assert allowed == frozenset({"risk_level_meta", "risk_enforce"})


def test_inv6_allow_map_contents_documented():
    """Each allow-map entry encodes a deliberate consumer; reviewers must
    understand why before widening it (security decision)."""
    assert set(ALLOW) == {
        "api/app/domain/services/permission/sources/skill_source.py",
        "api/app/domain/services/permission/sources/skill_metadata.py",
        "api/app/domain/services/permission/default_engine.py",
        "api/app/domain/services/skill_risk_assessor.py",
        "api/app/domain/models/tool_result.py",
        _REACT_GRAPH_REL,
    }


def test_inv6_react_graph_has_no_skill_bypass_symbols():
    """PE-1b: the legacy R3 Skill Stage P branch is deleted, so react_graph.py
    must not read *skill* risk metadata. The native legacy fallback gate may
    still consume risk_level_meta / risk_enforce (see ALLOW), but the four
    skill-only symbols in REACT_GRAPH_SKILL_BANNED are hard-banned: their
    reappearance means the R3 branch crept back in.
    """
    react_graph = (
        REPO_ROOT
        / "api"
        / "app"
        / "domain"
        / "services"
        / "graphs"
        / "react_graph.py"
    )
    hits = _scan_file(react_graph, REACT_GRAPH_SKILL_BANNED)
    assert not hits, (
        "react_graph.py consumes skill bypass symbols (R3 Skill Stage P "
        "crept back?):\n"
        + "\n".join(f"  react_graph.py:{ln} — {s}" for ln, s in sorted(hits))
    )
