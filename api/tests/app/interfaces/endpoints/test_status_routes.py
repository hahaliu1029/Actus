from fastapi.testclient import TestClient

from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.main import app


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


def test_get_status(client: TestClient) -> None:
    """测试获取应用状态API接口"""
    app.dependency_overrides[get_current_user] = _fake_user
    try:
        response = client.get("/api/status")
        data = response.json()

        assert response.status_code == 200
        assert data["code"] == 200
    finally:
        app.dependency_overrides.pop(get_current_user, None)
