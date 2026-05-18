"""PE-1 §3.1 — SkillRiskRefresher hash-aware typed result wrapper.

Owns content_hash comparison itself; returns 'fresh' when unchanged so
healthy unchanged skills don't get force-HIGH on every call (Round 1 P0#1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.domain.services.permission.skill_refresher import SkillRiskRefresher
from app.domain.services.risk_assessor import RiskLevel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSkillTool:
    """Stub mirroring SkillTool's resolve_skill_dir + refresh_risk_if_stale."""

    def __init__(self, *, skill_dir: Path | None, refresh_raises: BaseException | None = None,
                 refresh_return: str | None = None):
        self._skill_dir = skill_dir
        self._refresh_raises = refresh_raises
        self._refresh_return = refresh_return
        self.refresh_calls: list[str] = []

    def resolve_skill_dir(self, tool_name: str) -> Path | None:
        return self._skill_dir

    def refresh_risk_if_stale(self, tool_name: str) -> str | None:
        self.refresh_calls.append(tool_name)
        if self._refresh_raises is not None:
            raise self._refresh_raises
        return self._refresh_return


@pytest.fixture
def real_skill_dir(tmp_path: Path) -> Path:
    """Create a directory with a stable hashable file so SkillsGuard hash is deterministic."""
    d = tmp_path / "sk_real"
    d.mkdir()
    (d / "manifest.json").write_text('{"tools": []}', encoding="utf-8")
    return d


@pytest.fixture
def stable_hash(real_skill_dir: Path) -> str:
    from app.domain.services.skills_guard import SkillsGuard
    return SkillsGuard.compute_content_hash(real_skill_dir)


class TestRefresherReturnsFreshOnHashMatch:
    async def test_returns_fresh_when_hash_matches(self, real_skill_dir, stable_hash):
        st = _FakeSkillTool(skill_dir=real_skill_dir)
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash=stable_hash)
        assert out.status == "fresh"
        assert out.risk_level is None
        # MUST NOT call refresh_risk_if_stale when hash matches
        assert st.refresh_calls == []


class TestRefresherReturnsRefreshedOnHashDiff:
    async def test_returns_refreshed_with_new_risk_level(self, real_skill_dir, stable_hash):
        st = _FakeSkillTool(skill_dir=real_skill_dir, refresh_return="high")
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="sha256:WRONG")
        assert out.status == "refreshed"
        assert out.risk_level == RiskLevel.HIGH
        assert st.refresh_calls == ["t1"]

    async def test_returns_refreshed_for_medium(self, real_skill_dir):
        st = _FakeSkillTool(skill_dir=real_skill_dir, refresh_return="medium")
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="sha256:OUTDATED")
        assert out.status == "refreshed"
        assert out.risk_level == RiskLevel.MEDIUM

    async def test_returns_unknown_when_refresh_returns_none_after_diff(self, real_skill_dir):
        """Anomaly path: hash differs but refresher returned None (race / sync_manager state)."""
        st = _FakeSkillTool(skill_dir=real_skill_dir, refresh_return=None)
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="sha256:OLD")
        assert out.status == "unknown"
        assert "refresh_returned_none_after_hash_diff" in (out.error or "")


class TestRefresherReturnsUnknownWhenSkillDirUnresolvable:
    async def test_returns_unknown_when_resolve_returns_none(self):
        st = _FakeSkillTool(skill_dir=None)
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="anything")
        assert out.status == "unknown"
        assert out.error == "skill_dir_unresolvable"
        assert st.refresh_calls == []

    async def test_returns_unknown_when_skill_dir_does_not_exist(self, tmp_path):
        st = _FakeSkillTool(skill_dir=tmp_path / "ghost_dir")
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="x")
        assert out.status == "unknown"
        assert out.error == "skill_dir_missing"

    async def test_returns_unknown_when_compute_content_hash_returns_none(self, real_skill_dir):
        st = _FakeSkillTool(skill_dir=real_skill_dir)
        r = SkillRiskRefresher(st)
        with patch(
            "app.domain.services.permission.skill_refresher.SkillsGuard.compute_content_hash",
            return_value=None,
        ):
            out = await r.refresh("t1", expected_content_hash="x")
        assert out.status == "unknown"
        assert "cannot_compute_hash" in (out.error or "")


class TestRefresherReturnsFailedOnException:
    async def test_oserror_returns_failed(self, real_skill_dir):
        st = _FakeSkillTool(skill_dir=real_skill_dir,
                            refresh_raises=OSError("permission denied"))
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="sha256:OLD")
        assert out.status == "failed"
        assert "permission denied" in (out.error or "")

    async def test_value_error_returns_failed(self, real_skill_dir):
        st = _FakeSkillTool(skill_dir=real_skill_dir,
                            refresh_raises=ValueError("invalid risk"))
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="sha256:OLD")
        assert out.status == "failed"

    async def test_unexpected_exception_returns_failed(self, real_skill_dir):
        st = _FakeSkillTool(skill_dir=real_skill_dir,
                            refresh_raises=RuntimeError("unexpected"))
        r = SkillRiskRefresher(st)
        out = await r.refresh("t1", expected_content_hash="sha256:OLD")
        assert out.status == "failed"
        assert "unexpected" in (out.error or "")
