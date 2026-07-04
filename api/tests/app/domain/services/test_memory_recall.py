"""B8 PR-1: 纯函数契约（P-2）。无 I/O、确定性。"""
import hashlib

import pytest

from app.domain.models.app_config import MemoryConfig
from app.domain.models.memory_recall import RecallQueryMaterial
from app.domain.services.memory_recall import (
    RECALL_ITEM_RENDER_CAP,
    RECALL_QUERY_MAX_CHARS,
    build_params_version,
    build_recall_query,
    clean_session_title,
    compute_query_hash,
    normalize_recall_query,
)


def _material(message="帮我整理周报", original_request=None, session_title=None, entry="graph"):
    return RecallQueryMaterial(
        message=message,
        original_request=original_request,
        session_title=session_title,
        entry=entry,
    )


class TestCleanSessionTitle:
    @pytest.mark.parametrize("bad", [None, "", "   ", "新对话", "Task"])
    def test_placeholders_become_none(self, bad):
        assert clean_session_title(bad) is None

    def test_real_title_stripped_passthrough(self):
        assert clean_session_title("  季度数据分析  ") == "季度数据分析"


class TestBuildRecallQuery:
    def test_order_title_then_original_then_message(self):
        q = build_recall_query(
            _material(message="输出为ppt", original_request="分析季度销售数据", session_title="销售分析"),
        )
        assert q == "销售分析\n分析季度销售数据\n输出为ppt"

    def test_original_request_deduped_when_equal_to_message(self):
        q = build_recall_query(_material(message="继续", original_request="继续"))
        assert q == "继续"

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_original_request_absent_variants(self, empty):
        q = build_recall_query(_material(message="继续", original_request=empty))
        assert q == "继续"

    def test_placeholder_title_dropped(self):
        q = build_recall_query(_material(message="继续", session_title="新对话"))
        assert q == "继续"

    def test_max_chars_truncation(self):
        long_msg = "长" * 3000
        q = build_recall_query(_material(message=long_msg))
        assert len(q) == RECALL_QUERY_MAX_CHARS

    def test_max_chars_kwarg_is_authoritative(self):
        q = build_recall_query(_material(message="x" * 500), max_chars=100)
        assert len(q) == 100

    def test_message_stripped(self):
        q = build_recall_query(_material(message="  继续  "))
        assert q == "继续"


class TestNormalizeRecallQuery:
    def test_whitespace_folded_and_stripped(self):
        assert normalize_recall_query("  A\n\tB   C ") == "a b c"

    def test_ascii_lowered_cjk_unchanged(self):
        assert normalize_recall_query("整理 Q4 REPORT") == "整理 q4 report"


class TestComputeQueryHash:
    def test_sha256_of_version_pipe_query(self):
        expected = hashlib.sha256("v1:k5|查询".encode("utf-8")).hexdigest()
        assert compute_query_hash("查询", params_version="v1:k5") == expected

    def test_params_version_sensitivity(self):
        a = compute_query_hash("查询", params_version="A")
        b = compute_query_hash("查询", params_version="B")
        assert a != b


class TestBuildParamsVersion:
    def test_canonical_format_with_defaults(self):
        cfg = MemoryConfig()
        assert build_params_version(cfg) == "v1:k5:t0.35:h30:m0.7:c3:r120"

    def test_changes_with_top_k(self):
        assert build_params_version(MemoryConfig(recall_top_k=6)) != build_params_version(MemoryConfig())


class TestConstants:
    def test_pinned_values(self):
        assert RECALL_QUERY_MAX_CHARS == 2000
        assert RECALL_ITEM_RENDER_CAP == 120
