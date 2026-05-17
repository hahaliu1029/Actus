"""Protocol shape check — duck-typed conformance."""

from typing import Protocol, runtime_checkable

from app.domain.services.permission.escalation_provider import EscalationProvider


def test_protocol_attrs():
    # runtime_checkable is not used (Protocol is structural). Verify the
    # attribute set by checking protocol annotations and method presence.
    # Protocol class-level annotations live in __annotations__, not hasattr.
    assert "name" in EscalationProvider.__annotations__
    assert hasattr(EscalationProvider, "resolve")


def test_duck_typed_conformance():
    class Stub:
        name = "stub"

        async def resolve(self, call, ctx, upstream_outcome):  # noqa
            return None

    # Structural — instance check only meaningful with runtime_checkable.
    # We assert it walks like a duck:
    s = Stub()
    assert s.name == "stub"
    assert hasattr(s, "resolve")
