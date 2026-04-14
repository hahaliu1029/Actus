"""Verify CheckpointerPool is opened on startup and closed on shutdown."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app, lifespan


@pytest.mark.anyio
async def test_lifespan_opens_and_closes_checkpointer_pool():
    """FastAPI lifespan should call CheckpointerPool.open() on startup
    and CheckpointerPool.close() on shutdown."""
    mock_open = AsyncMock()
    mock_close = AsyncMock()
    mock_pool_instance = MagicMock()
    mock_pool_instance.open = mock_open
    mock_pool_instance.close = mock_close
    mock_pool_instance.pool = MagicMock()

    with patch(
        "app.infrastructure.checkpointer_pool.CheckpointerPool", return_value=mock_pool_instance
    ), patch(
        "app.main.get_redis", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock())
    ), patch(
        "app.main.get_postgres", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock(), session_factory=MagicMock())
    ), patch(
        "app.main.get_minio", return_value=MagicMock(init=AsyncMock(), shutdown=AsyncMock())
    ), patch(
        "app.main.command"  # skip Alembic migrations
    ), patch(
        "app.interfaces.service_dependencies._build_agent_service", return_value=MagicMock(shutdown=AsyncMock())
    ):
        async with lifespan(app):
            # Startup complete — open() should have been called
            mock_open.assert_awaited_once()

        # Lifespan exited — close() should have been called
        mock_close.assert_awaited_once()
