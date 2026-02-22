from fastapi.testclient import TestClient


def test_get_status(client: TestClient) -> None:
    """测试获取应用状态API接口"""
    # 发送GET请求到状态接口
    response = client.get("/api/status")
    data = response.json()

    # 断言响应状态码为200
    assert response.status_code == 200
    assert data["code"] == 200
