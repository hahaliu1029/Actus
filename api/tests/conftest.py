import uuid
from typing import Generator
from uuid import UUID

import pytest
from app.main import app
from fastapi.testclient import TestClient
from tests._tool_source_testing import reset_tool_source_registry

# M1 memory 系统用的固定测试 user_id（UUID v4 格式）。
# 规范：所有 memory / flush 相关测试统一使用 UUID；同用户场景用
# ``TEST_USER_ID_FIXED``，跨用户对比（权限隔离）用 ``TEST_OTHER_USER_ID_FIXED``，
# 集成测试需要独立用户时用 ``random_uuid_user_id`` fixture 或直接 uuid.uuid4()。
TEST_USER_ID_FIXED: str = str(UUID("00000000-0000-4000-8000-000000000001"))
TEST_OTHER_USER_ID_FIXED: str = str(UUID("00000000-0000-4000-8000-000000000002"))


@pytest.fixture
def random_uuid_user_id() -> str:
    """每次调用返回一个全新的 UUID v4 字符串，用于需要独立用户的集成测试。"""
    return str(uuid.uuid4())


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
