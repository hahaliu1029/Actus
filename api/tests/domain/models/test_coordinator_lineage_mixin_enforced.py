"""[C2 PR-9 §15.2 Gate #6] Every ``Coordinator*Event`` class in
``domain/models/event.py`` MUST inherit ``CoordinatorLineageMixin``.

PR-8 introduced ``CoordinatorLineageMixin`` (event.py:372-383) — the
optional lineage tag carrier (root/parent/child session ids,
coordinator_run_id, work_unit_id) used so that any coordinator-emitted
event can be threaded through the SSE pipeline as a child-tagged event.
Spec §13.2 makes the mixin mandatory on every coordinator event so the
frontend can rely on lineage fields existing (even if ``None``).

The five events expected to be detected by this gate (PR-8 Task 8.1):
    - CoordinatorDispatchEvent
    - CoordinatorWorkerSpawnedEvent
    - CoordinatorReduceEvent
    - CoordinatorApplyEvent
    - CoordinatorSiblingCancelEvent

If a future PR adds a sixth ``Coordinator*Event`` without composing the
mixin, this gate fires.

KNOWN HEURISTIC LIMITS (gate only inspects ``ast.Name`` bases with the
exact identifier ``CoordinatorLineageMixin``):

1. Aliased import: ``from event import CoordinatorLineageMixin as _Lineage``
   followed by ``class CoordinatorXEvent(BaseEvent, _Lineage)``. The class
   DOES inherit the mixin at runtime, but the gate's base-name check
   finds ``_Lineage`` only, fires a violation, and emits a misleading
   "must inherit CoordinatorLineageMixin" message. Real bug? No.
   Cosmetic failure? Yes. Solution if encountered: stop aliasing or
   relax the gate to also accept ``ast.alias`` resolution via the
   module's import graph.
2. Attribute base: ``class CoordinatorXEvent(BaseEvent, mod.CoordinatorLineageMixin)``
   passes silently. The ``isinstance(b, ast.Name)`` filter rejects
   ``ast.Attribute`` bases, so the mixin requirement is unenforced for
   any class that imports the mixin's containing module rather than
   the symbol directly. Plausible during circular-import refactors.
   Mitigation: keep mixin imports as direct ``from x import Mixin``
   (project convention); flagged via codex review if this becomes a
   real pattern.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.coordinator_pure


# Resolve repo-relative ``api/app/...`` paths from this test file's location so
# the gate works regardless of pytest's CWD (``api/`` vs repo root).
_API_ROOT = Path(__file__).resolve().parents[3]
_EVENT_MODULE = _API_ROOT / "app" / "domain" / "models" / "event.py"


def test_all_coordinator_events_inherit_mixin():
    src = _EVENT_MODULE.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Coordinator") and node.name.endswith("Event"):
            base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
            assert "CoordinatorLineageMixin" in base_names, \
                f"{node.name} must inherit CoordinatorLineageMixin"
