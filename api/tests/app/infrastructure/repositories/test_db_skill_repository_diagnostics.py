"""B9 PR-0：DB repo 诊断实现——DB 条目无文件损坏概念，恒 ok 包装（spec §8 R10#4）。"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.repositories.db_skill_repository import DBSkillRepository


@pytest.mark.anyio
async def test_db_diagnostics_wraps_list_as_ok(monkeypatch):
    repo = DBSkillRepository(db_session=MagicMock())
    fake_skill = MagicMock(id="s1")
    monkeypatch.setattr(repo, "list", AsyncMock(return_value=[fake_skill]))
    diags = await repo.list_with_diagnostics()
    assert len(diags) == 1
    assert diags[0].ok is True
    assert diags[0].skill_key == "s1"
    assert diags[0].skill is fake_skill
    assert diags[0].error_code is None
