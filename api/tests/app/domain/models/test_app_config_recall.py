"""B8 PR-1: MemoryConfig recall_* 字段契约（P-9）。"""
import pytest
from pydantic import ValidationError

from app.domain.models.app_config import MemoryConfig


class TestRecallConfigDefaults:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.recall_mode == "off"
        assert cfg.recall_top_k == 5
        assert cfg.recall_threshold == 0.35
        assert cfg.recall_timeout_seconds == 0.75
        assert cfg.recall_cache_ttl_seconds == 86400
        assert cfg.recall_query_max_chars == 2000

    def test_mode_literal_rejects_unknown(self):
        with pytest.raises(ValidationError):
            MemoryConfig(recall_mode="ON")

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("recall_top_k", 0),
            ("recall_top_k", 21),
            ("recall_threshold", -0.1),
            ("recall_threshold", 1.1),
            ("recall_timeout_seconds", 0.05),
            ("recall_timeout_seconds", 16.0),
            ("recall_cache_ttl_seconds", 59),
            ("recall_query_max_chars", 99),
        ],
    )
    def test_bounds(self, field, bad):
        with pytest.raises(ValidationError):
            MemoryConfig(**{field: bad})
