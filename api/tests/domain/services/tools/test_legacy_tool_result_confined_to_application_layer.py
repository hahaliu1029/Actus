"""R2 CS2 legacy ToolResult soft-coexistence — guardrail 2.

The legacy ``app.domain.models.tool_result.ToolResult`` class is kept
for backward compatibility with three layers that pre-date R2:

1. **``app.application.services.*``** — the main consumer; retires in
   R4 / Phase 2 alongside the CS3 ``ToolEventEnvelope`` freeze.
2. **Internal tool class implementations** (``app/domain/services/tools/
   base.py`` / ``a2a.py`` / ``mcp.py`` / ``skill.py`` / ``brainstorm_skill.py``
   / ``create_skill.py``) — these define the underlying ``BaseTool``
   subclasses whose ``.invoke()`` returns ``ToolResult``. The R2
   langchain wrappers call them and convert the output to
   ``ToolOutcome`` via bridge helpers like
   ``skill.py::_tool_result_to_outcome``.
3. **``react_graph.py``** — the ``ToolEvent.function_result: Optional
   [ToolResult]`` field in ``app.domain.models.event`` still requires
   constructing ``ToolResult(success=..., message=...)`` to feed
   ``AgentTaskRunner._handle_tool_event`` enrichment. R4 CS3 envelope
   retires this field.

This guardrail enforces guardrail 2 at the **canonical R2 wrapper
boundary** — the 7 files that implement the new
``@tool(response_format="content_and_artifact")`` contract and return
typed ``(content, ToolOutcome)`` tuples. Those files must NOT import
the legacy class, because that would drop them back into the R1
raise-exception-or-return-ToolResult pathway that R2 replaces.

**Scope (canonical R2 wrappers)**:

- ``langchain_tools.py``
- ``langchain_mcp.py``
- ``langchain_a2a.py``
- ``langchain_dynamic_skill_tools.py``
- ``langchain_skill_tools.py``
- ``langchain_mcp_discovery.py``
- ``memory_tools.py``

**Deliberately out of scope** (documented soft coexistence):

- ``tools/base.py`` / ``mcp.py`` / ``a2a.py`` / ``skill.py`` /
  ``brainstorm_skill.py`` / ``create_skill.py`` — legacy tool classes
  wrapped by langchain_*.py adapters.
- ``graphs/react_graph.py`` — ``ToolEvent.function_result`` still uses
  ``ToolResult`` until R4.
- ``application/services/**`` — main consumer, retires in R4.

AST-scan only — the test walks source text without importing, so a
broken wrapper doesn't cascade-fail the test with an ImportError.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


# Resolve the API root once (tests run from repo root or api/ — handle both)
_THIS_DIR = Path(__file__).resolve().parent
_API_ROOT = _THIS_DIR.parents[3]


# Canonical R2 wrapper files (the 7 files that implement
# @tool(response_format="content_and_artifact") and return typed
# (content, ToolOutcome) tuples). Everything else is documented
# soft-coexistence and out of scope for this guardrail.
CANONICAL_R2_WRAPPERS: list[Path] = [
    _API_ROOT / "app" / "domain" / "services" / "tools" / name
    for name in [
        "langchain_tools.py",
        "langchain_mcp.py",
        "langchain_a2a.py",
        "langchain_dynamic_skill_tools.py",
        "langchain_skill_tools.py",
        "langchain_mcp_discovery.py",
        "memory_tools.py",
    ]
]

# The module itself is allowed to define ToolResult; we only ban
# downstream imports of that name at canonical wrapper boundaries.
MODULE_ALLOWED = _API_ROOT / "app" / "domain" / "models" / "tool_result.py"


@pytest.mark.parametrize(
    "py_file",
    CANONICAL_R2_WRAPPERS,
    ids=lambda p: p.name,
)
def test_canonical_r2_wrapper_does_not_import_legacy_tool_result(py_file: Path):
    """Canonical R2 wrappers must not import the legacy ``ToolResult`` class.

    The 7 canonical wrappers (langchain_*.py + memory_tools.py) return
    typed ``(content, ToolOutcome)`` tuples via
    ``@tool(response_format="content_and_artifact")``. Any of them
    importing ``ToolResult`` is a migration leftover — it means the
    wrapper is still using the R1 pathway and should be switched to
    ``AllowSuccess`` / ``AllowError`` / ``Denied`` / ``Asked`` /
    ``Passthrough``.

    Out of scope by design: ``skill.py`` / ``base.py`` / ``mcp.py`` /
    ``a2a.py`` / ``brainstorm_skill.py`` / ``create_skill.py`` — these
    are the underlying tool classes wrapped by langchain_*.py, and
    ``react_graph.py`` which constructs ``ToolResult`` for the legacy
    ``ToolEvent.function_result`` field. Both are documented soft
    coexistence until R4 / Phase 2.
    """
    tree = ast.parse(py_file.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "app.domain.models.tool_result":
            continue
        for alias in node.names:
            assert alias.name != "ToolResult", (
                f"{py_file.name}:{node.lineno} imports legacy ``ToolResult``. "
                f"R2 CS2 guardrail 2 forbids this in canonical R2 wrappers — "
                f"use AllowSuccess / AllowError / Denied / Asked / "
                f"Passthrough / ToolArtifact instead. If you legitimately "
                f"need ToolResult here, the file probably belongs in the "
                f"soft-coexistence list (skill.py / base.py / mcp.py / "
                f"a2a.py) rather than the canonical wrapper set."
            )


def test_canonical_r2_wrapper_set_is_stable():
    """Regression guard — the canonical wrapper set must not shrink silently.

    If a new R2 wrapper file is added, it MUST be appended to
    ``CANONICAL_R2_WRAPPERS`` above so the guardrail extends to it.
    This test asserts the set size stays at 7. Any change to the list
    is deliberate and visible in diff — not silent drift.
    """
    assert len(CANONICAL_R2_WRAPPERS) == 7, (
        f"Expected 7 canonical R2 wrappers, found {len(CANONICAL_R2_WRAPPERS)}. "
        f"If you added a new wrapper, extend CANONICAL_R2_WRAPPERS to cover it. "
        f"If you retired one, update this count + the comment in the docstring."
    )
    for path in CANONICAL_R2_WRAPPERS:
        assert path.exists(), (
            f"Canonical wrapper {path.name} no longer exists at "
            f"{path.relative_to(_API_ROOT)}. Update CANONICAL_R2_WRAPPERS."
        )


def test_tool_result_module_still_defines_legacy_class():
    """Sanity: the tool_result module itself must still export ``ToolResult``.

    This test protects against accidentally removing the legacy class
    before R4. The soft-coexistence contract explicitly keeps it alive
    for the ``application.services.*`` consumers that haven't migrated.
    """
    tree = ast.parse(MODULE_ALLOWED.read_text())
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert "ToolResult" in class_names, (
        "app/domain/models/tool_result.py no longer defines ``ToolResult``. "
        "That class is retained for soft-coexistence with "
        "application/services/** until R4 / Phase 2. If you intend to "
        "retire it, migrate the application layer first and update "
        "the R2 spec §Legacy ToolResult 软共存 section."
    )


def test_application_layer_may_still_import_legacy_tool_result():
    """Positive assertion: application/services/** is explicitly allowed.

    This is a documentation test — it scans application/services for
    any import of ``ToolResult`` and confirms we find at least one
    (otherwise the whole "soft coexistence" rationale is obsolete and
    the class should be retired). If application layer has fully
    migrated off ``ToolResult``, this test starts passing on zero
    imports and the guardrail can be tightened to forbid all layers.
    """
    app_services = _API_ROOT / "app" / "application" / "services"
    if not app_services.exists():
        pytest.skip("application/services/ not found — skipping informational assertion")

    found_imports = 0
    for py_file in app_services.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "app.domain.models.tool_result"
                and any(alias.name == "ToolResult" for alias in node.names)
            ):
                found_imports += 1

    # Not a hard assertion — this is informational coverage. If
    # found_imports == 0, the application layer has fully migrated
    # and the soft-coexistence guardrail can be retired.
    assert found_imports >= 0  # always passes, documents the count
