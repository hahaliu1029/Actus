from app.domain.services.prompts.assembler import PromptAssembler


def _build(**kw):
    base = dict(objective="do x", phase="exploration", allowed_paths=[], work_unit_id="w.a0.0")
    base.update(kw)
    return PromptAssembler.build_minimal_for_coordinator_child(**base)


def test_none_member_prompt_is_byte_identical():
    # INV-0: omitting member_system_prompt yields the exact pre-S4 prompt.
    assert _build() == _build(member_system_prompt=None)


def test_member_prompt_injected_under_subordinate_heading():
    out = _build(member_system_prompt="You are a precise explorer.")
    assert "You are a precise explorer." in out
    # R10-4: framed as advisory, subordinate to runtime authorization.
    assert "advisory" in out.lower()
    # ordering: identity ... member block ... behavior ... work unit
    assert out.index("Coordinator Step Worker") < out.index("You are a precise explorer.")
    assert out.index("You are a precise explorer.") < out.index("## Behavior")


def test_empty_member_prompt_is_byte_identical():
    assert _build() == _build(member_system_prompt="")
    assert _build() == _build(member_system_prompt="   ")
