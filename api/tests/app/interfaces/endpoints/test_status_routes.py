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

        # 接口可达，返回 200 HTTP 状态码
        assert response.status_code == 200
        # 业务码为 200（全部健康）或 503（CI 无基础设施时部分异常），均为合法结果
        assert data["code"] in (200, 503)
        assert "data" in data
    finally:
        app.dependency_overrides.pop(get_current_user, None)
