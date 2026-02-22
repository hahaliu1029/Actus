from typing import Generator

import pytest
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    创建一个可供所有测试用例使用的 TestClient 客户端。
    scope="session" 表示这个fixture 在整个测试用例只会实例一次，这样可以提高效率
    :return: TestClient
    """
    with TestClient(app) as c:
        yield c
