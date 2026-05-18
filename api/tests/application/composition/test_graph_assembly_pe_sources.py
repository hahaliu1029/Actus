"""PE-1 §2.6 — graph_assembly.build_permission_engine accepts sources +
validate_pe_source_registry fails fast on registry mismatch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.composition.graph_assembly import (
    build_permission_engine,
    validate_pe_source_registry,
)
from app.domain.services.permission.errors import PermissionConfigurationError
from app.domain.services.permission.sources import (
    NativeSource,
    PE_SUPPORTED_SOURCES_AFTER_PE_1,
    SkillSource,
)


def _stub_writer():
    w = AsyncMock()
    return w


def _stub_queue():
    q = AsyncMock()
    return q


def _stub_reader():
    r = AsyncMock()
    return r


def _stub_ssm():
    s = AsyncMock()
    return s


def _stub_uow_factory():
    return MagicMock()


# ---------- validate_pe_source_registry ----------

class TestValidateSourceRegistry:
    def test_passes_when_all_supported_sources_registered(self):
        sources = {"native": NativeSource(),
                   "skill": SkillSource(refresher=MagicMock(), redis=MagicMock())}
        # Should not raise
        validate_pe_source_registry(sources)

    def test_raises_when_skill_missing(self):
        sources = {"native": NativeSource()}
        with pytest.raises(PermissionConfigurationError) as exc_info:
            validate_pe_source_registry(sources)
        assert "skill" in str(exc_info.value)

    def test_raises_when_native_missing(self):
        sources = {"skill": SkillSource(refresher=MagicMock(), redis=MagicMock())}
        with pytest.raises(PermissionConfigurationError) as exc_info:
            validate_pe_source_registry(sources)
        assert "native" in str(exc_info.value)

    def test_raises_when_empty(self):
        with pytest.raises(PermissionConfigurationError):
            validate_pe_source_registry({})

    def test_extra_sources_beyond_supported_set_pass(self):
        """Adding 'mcp' before PE-2 should not fail validation — we only
        check that supported set is covered."""
        sources = {
            "native": NativeSource(),
            "skill": SkillSource(refresher=MagicMock(), redis=MagicMock()),
            "mcp": MagicMock(),  # not yet supported, but harmless
        }
        validate_pe_source_registry(sources)


# ---------- build_permission_engine ----------

class TestBuildPermissionEngineAcceptsSources:
    def test_passes_sources_to_engine_ctor(self):
        sources = {"native": NativeSource(),
                   "skill": SkillSource(refresher=MagicMock(), redis=MagicMock())}
        pe = build_permission_engine(
            uow_factory=_stub_uow_factory(),
            writer=_stub_writer(),
            queue=_stub_queue(),
            session_machine=_stub_ssm(),
            reader=_stub_reader(),
            summary_llm=None,
            sources=sources,
        )
        assert pe._sources is sources

    def test_default_sources_none_does_not_raise_at_build_time(self):
        """PE-1 §2.6 — validation is caller-driven (post late-registration).
        build_permission_engine accepts empty/partial sources; caller calls
        validate_pe_source_registry explicitly after wiring is complete."""
        pe = build_permission_engine(
            uow_factory=_stub_uow_factory(),
            writer=_stub_writer(),
            queue=_stub_queue(),
            session_machine=_stub_ssm(),
            reader=_stub_reader(),
            summary_llm=None,
        )
        assert pe is not None
