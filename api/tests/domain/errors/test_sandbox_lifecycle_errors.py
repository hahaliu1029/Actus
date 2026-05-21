"""C3 PR-1 — new SandboxLifecycleError subclasses (spec §3.2 M2 + §7.3)."""

from app.domain.errors.sandbox_lifecycle import (
    SandboxLifecycleError,
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
)


def test_sandbox_already_destroyed_is_subclass():
    assert issubclass(SandboxAlreadyDestroyed, SandboxLifecycleError)


def test_sandbox_binding_missing_is_subclass():
    assert issubclass(SandboxBindingMissing, SandboxLifecycleError)


def test_sandbox_already_destroyed_carries_session_id():
    err = SandboxAlreadyDestroyed("session-xyz")
    assert err.session_id == "session-xyz"
    assert "session-xyz" in str(err)


def test_sandbox_binding_missing_carries_session_id():
    err = SandboxBindingMissing("session-abc")
    assert err.session_id == "session-abc"


def test_subclasses_are_distinct():
    assert not issubclass(SandboxAlreadyDestroyed, SandboxBindingMissing)
    assert not issubclass(SandboxBindingMissing, SandboxAlreadyDestroyed)


def test_catch_via_base_class():
    try:
        raise SandboxAlreadyDestroyed("s1")
    except SandboxLifecycleError as e:
        assert isinstance(e, SandboxAlreadyDestroyed)
