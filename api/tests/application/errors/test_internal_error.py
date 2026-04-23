import pytest
from app.application.errors.exceptions import AppException, InternalError


def test_internal_error_is_app_exception_subclass() -> None:
    assert issubclass(InternalError, AppException)


def test_internal_error_constructible_and_preserves_msg() -> None:
    exc = InternalError("invariant violated: foo")
    assert "invariant violated" in str(exc)
    assert exc.msg == "invariant violated: foo"


def test_internal_error_sets_500_status_code() -> None:
    """Follows the existing AppException subclass pattern (see BadRequestError/NotFoundError/...)."""
    exc = InternalError("boom")
    assert exc.code == 500
    assert exc.status_code == 500


def test_internal_error_raises_and_propagates() -> None:
    with pytest.raises(InternalError, match="boom"):
        raise InternalError("boom")


def test_config_error_is_app_exception_and_500() -> None:
    from app.application.errors.exceptions import AppException, ConfigError
    assert issubclass(ConfigError, AppException)
    exc = ConfigError("unknown provider 'x'")
    assert exc.code == 500
    assert exc.status_code == 500
    assert "unknown provider" in exc.msg
