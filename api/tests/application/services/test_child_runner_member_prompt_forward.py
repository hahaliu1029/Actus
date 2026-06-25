from app.application.services.coordinator_child_runner import CoordinatorChildRunner
from app.domain.models.work_unit import WorkUnit


def test_build_child_prompt_forwards_member_system_prompt():
    # _build_child_prompt is an instance method but uses only `wu` + the static
    # assembler; call it on a bare instance via __new__ to avoid the full ctor.
    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    wu = WorkUnit(work_unit_id="w.a0.0", objective="o", phase="exploration",
                  system_prompt="You are the explorer member.")
    out = runner._build_child_prompt(wu, manifest=None)
    assert "You are the explorer member." in out
    assert "advisory" in out.lower()


def test_build_child_prompt_none_member_is_byte_identical():
    runner = CoordinatorChildRunner.__new__(CoordinatorChildRunner)
    wu_plain = WorkUnit(work_unit_id="w.a0.0", objective="o", phase="exploration")
    wu_none = WorkUnit(work_unit_id="w.a0.0", objective="o", phase="exploration", system_prompt=None)
    assert runner._build_child_prompt(wu_plain, None) == runner._build_child_prompt(wu_none, None)
