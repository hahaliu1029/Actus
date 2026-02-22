from core.security import verify_password


def test_verify_password_returns_false_for_invalid_hash() -> None:
    assert verify_password("123456", "plain-text-password") is False
