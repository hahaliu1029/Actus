"""PermissionEngine ABC contract."""

import pytest

from app.domain.services.permission.engine import PermissionEngine


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        PermissionEngine()  # type: ignore[abstract]


def test_required_methods_abstract():
    required = {"evaluate", "preflight_resume", "commit_resume"}
    actual = {
        name for name, attr in PermissionEngine.__dict__.items()
        if getattr(attr, "__isabstractmethod__", False)
    }
    assert required.issubset(actual)
