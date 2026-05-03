import pytest
from app.infrastructure.repositories.db_conversation_compaction_repository import (
    DBConversationCompactionRepository,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_uow_exposes_compaction_repo(uow_factory):
    async with uow_factory() as uow:
        assert isinstance(uow.compaction, DBConversationCompactionRepository)
