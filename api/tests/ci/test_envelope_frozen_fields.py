"""C3 spec §13.3 — MailboxEnvelope frozen fields immutable.

AST scan ensures the pydantic ``MailboxEnvelope`` model declaration:

  1. Includes every required field from C1 ADR §6.3 + R2 §15.7
     (``producer_role``), so wire-format compatibility is enforced at import
     time.
  2. Declares ``frozen=True`` in its ``model_config`` so envelopes are
     immutable once emitted (replay/redelivery safety: cannot mutate an
     envelope after it has been published).

Why a *separate* test from the model's own pydantic validators: this gate
catches accidental field deletion / rename even if downstream tests still
pass with a stale Mock. The list of required fields is duplicated here on
purpose — drift between the model definition and the spec must surface as a
CI failure rather than a silent migration.
"""

from __future__ import annotations

import ast
import pathlib


_ENVELOPE_FILE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "api"
    / "app"
    / "domain"
    / "models"
    / "mailbox_envelope.py"
)

# Spec §4.1 + C1 ADR §6.3 + R2 §15.7 — the 8 frozen MailboxEnvelope fields.
_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "envelope_id",
        "type",
        "parent_session_id",
        "child_session_id",
        "correlation_id",
        "emitted_at",
        "producer_role",
        "payload",
    }
)


def _find_class_def(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _collect_annotated_field_names(cls_node: ast.ClassDef) -> set[str]:
    """Return all ``ast.AnnAssign`` target names (typed class attributes)
    declared directly in the class body. Does NOT recurse into nested
    classes (pydantic models don't legitimately need that)."""
    found: set[str] = set()
    for stmt in cls_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            found.add(stmt.target.id)
    return found


def _model_config_call(cls_node: ast.ClassDef) -> ast.Call | None:
    """Return the ``ConfigDict(...)`` call assigned to ``model_config`` in
    the class body, or ``None`` if no such assignment exists."""
    for stmt in cls_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not (
            len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "model_config"
        ):
            continue
        if isinstance(stmt.value, ast.Call):
            return stmt.value
    return None


def test_envelope_has_all_required_frozen_fields() -> None:
    """Every field listed in spec §4.1 / C1 ADR §6.3 / R2 §15.7 must be
    declared as a typed class attribute on ``MailboxEnvelope``."""
    assert _ENVELOPE_FILE.exists(), f"expected {_ENVELOPE_FILE} to exist"
    tree = ast.parse(_ENVELOPE_FILE.read_text(encoding="utf-8"))
    cls = _find_class_def(tree, "MailboxEnvelope")
    assert cls is not None, "MailboxEnvelope class definition missing"

    found = _collect_annotated_field_names(cls)
    missing = _REQUIRED_FIELDS - found
    assert not missing, (
        f"MailboxEnvelope missing frozen fields {sorted(missing)} "
        f"(spec §4.1 + C1 ADR §6.3 + R2 §15.7). Declared fields: {sorted(found)}"
    )


def test_envelope_model_config_declares_frozen_true() -> None:
    """``MailboxEnvelope.model_config = ConfigDict(extra='forbid', frozen=True)``
    is the wire-format contract — envelopes are immutable after publish."""
    assert _ENVELOPE_FILE.exists(), f"expected {_ENVELOPE_FILE} to exist"
    tree = ast.parse(_ENVELOPE_FILE.read_text(encoding="utf-8"))
    cls = _find_class_def(tree, "MailboxEnvelope")
    assert cls is not None, "MailboxEnvelope class definition missing"

    cfg_call = _model_config_call(cls)
    assert cfg_call is not None, (
        "MailboxEnvelope is missing a `model_config = ConfigDict(...)` "
        "assignment in its class body"
    )

    frozen_kw = next(
        (kw for kw in cfg_call.keywords if kw.arg == "frozen"),
        None,
    )
    assert frozen_kw is not None, (
        "MailboxEnvelope.model_config must declare `frozen=True`"
    )
    assert isinstance(frozen_kw.value, ast.Constant) and frozen_kw.value.value is True, (
        f"MailboxEnvelope.model_config frozen= must be the literal True; "
        f"got {ast.dump(frozen_kw.value)}"
    )
