from typing import Generator

import pytest
from app.main import app
from fastapi.testclient import TestClient
from tests._tool_source_testing import reset_tool_source_registry


@pytest.fixture
def anyio_backend():
    """Pin anyio tests to asyncio only (trio is not installed)."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _clear_tool_source_registry():
    """Clear and re-bootstrap _REGISTRY before and after every test.

    Ensures every test starts from production-equivalent state. Dynamic
    registrations (mcp_/skill_/conflict-test names) added by one test do
    not pollute the next.
    """
    reset_tool_source_registry()
    yield
    reset_tool_source_registry()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    创建一个可供所有测试用例使用的 TestClient 客户端。
    scope="session" 表示这个fixture 在整个测试用例只会实例一次，这样可以提高效率
    :return: TestClient
    """
    with TestClient(app) as c:
        yield c
