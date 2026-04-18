"""M2-PR3: per-section unit tests for the 3 memory prompt sections.

Each section (memory_rules, memory_user_profile, memory_fact_index) is
tested in isolation against synthetic ``RenderContext`` fixtures built
over a ``MemorySnapshot``. The combined assembly behavior under budget
pressure (priority-DESC drop order) is covered in
``test_sections_m2_budget_drops.py``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.domain.models.memory_chunk import MemoryChunk
from app.domain.services.prompts.memory_snapshot import MemorySnapshot
from app.domain.services.prompts.section import RenderContext
from app.domain.services.prompts.sections._memory_section_helpers import (
    assemble_bullets_within_budget,
    estimate_tokens_cjk_aware,
    sanitize_bullet_content,
)
from app.domain.services.prompts.sections.memory_fact_index import (
    MEMORY_FACT_INDEX_BUDGET,
    memory_fact_index_section,
)
from app.domain.services.prompts.sections.memory_rules import (
    MEMORY_RULE_SECTION_BUDGET,
    memory_rules_section,
)
from app.domain.services.prompts.sections.memory_user_profile import (
    MEMORY_USER_SECTION_BUDGET,
    memory_user_profile_section,
)


# ---- Fixtures ---------------------------------------------------------- #


def _chunk(
    content: str,
    *,
    category: str | None = None,
    pinned: bool = False,
    chunk_id: str | None = None,
) -> MemoryChunk:
    now = datetime.now(timezone.utc)
    return MemoryChunk(
        id=chunk_id or str(uuid.uuid4()),
        user_id="u1",
        content=content,
        content_hash="h",
        source="session_flush",
        metadata={},
        created_at=now,
        updated_at=now,
        session_id=None,
        embedding=None,
        category=category,
        auto_promoted_at=None,
        fs_synced=True,
        pinned=pinned,
    )


def _ctx(
    *,
    snapshot: MemorySnapshot | None = None,
    lang: str = "zh",
) -> RenderContext:
    return RenderContext(
        lang=lang,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
    )


# ---- Shared helpers ---------------------------------------------------- #


class TestEstimateTokens:
    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens_cjk_aware("") == 0

    def test_very_short_string_has_minimum_of_one(self) -> None:
        """PR-3 self-review: before the fix, ``int(0.25) == 0`` meant a
        single-char ASCII bullet contributed 0 tokens to budget accounting.
        Canonical ``TokenEstimator.estimate`` uses ``max(round(t), 1)`` for
        non-empty input; this helper must match so assembler and section
        don't disagree on whether a bullet fits."""
        assert estimate_tokens_cjk_aware("a") == 1

    def test_ascii_weight_is_light(self) -> None:
        """4 ASCII chars ≈ 1 token (0.25 per char)."""
        tokens = estimate_tokens_cjk_aware("abcd")
        # 4 * 0.25 = 1.0 → round(1.0) = 1
        assert tokens == 1

    def test_cjk_weight_is_heavy(self) -> None:
        """2 CJK chars ≈ 3 tokens (1.5 per char)."""
        tokens = estimate_tokens_cjk_aware("你好")
        # 2 * 1.5 = 3.0
        assert tokens == 3

    def test_mixed_string(self) -> None:
        """CJK + ASCII + punctuation — values should add up roughly."""
        tokens = estimate_tokens_cjk_aware("你好, world")
        # "你好" = 3, ", " = 0.5, "world" = 1.25 → round(4.75) = 5
        assert tokens == 5

    def test_matches_canonical_token_estimator_rounding(self) -> None:
        """Sanity check: for a typical mixed string, this helper agrees
        with the canonical ``TokenEstimator(strategy='hybrid').estimate``
        that ``PromptAssembler`` uses. Drift here means sections and the
        assembler's budget check disagree."""
        from app.domain.services.graphs.token_estimator import TokenEstimator

        canonical = TokenEstimator(strategy="hybrid")
        for sample in ["a", "ab", "你好", "- 测试 a bullet", "- "]:
            assert (
                estimate_tokens_cjk_aware(sample) == canonical.estimate(sample)
            ), f"estimator drift on input {sample!r}"


class TestSanitizeBulletContent:
    def test_empty_returns_empty(self) -> None:
        assert sanitize_bullet_content("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert sanitize_bullet_content("   \n\t  ") == ""

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert sanitize_bullet_content("  hello  ") == "hello"

    def test_collapses_internal_newlines(self) -> None:
        """Multi-line content must flatten to a single line so the bullet
        structure stays intact."""
        assert sanitize_bullet_content("line1\nline2") == "line1 line2"

    def test_collapses_tabs_and_multiple_spaces(self) -> None:
        assert sanitize_bullet_content("a\tb    c") == "a b c"

    def test_collapses_blank_line_to_single_space(self) -> None:
        """Attacker-ish content with ``\\n\\n## fake`` flattens to a
        single line, defending the bullet structure without parsing
        markdown."""
        assert sanitize_bullet_content("line\n\n## fake") == "line ## fake"


class TestAssembleBullets:
    def test_returns_none_on_empty_bullets(self) -> None:
        assert assemble_bullets_within_budget(
            header="## H", bullets=[], max_tokens=100
        ) is None

    def test_joins_header_and_bullets_with_newline(self) -> None:
        result = assemble_bullets_within_budget(
            header="## H", bullets=["- a", "- b"], max_tokens=1000
        )
        assert result is not None
        text, kept = result
        assert text == "## H\n- a\n- b"
        assert kept == 2

    def test_drops_tail_when_over_budget(self) -> None:
        """Given a tight budget, tail bullets drop and ``None`` is never
        returned just because budget is exceeded — partial output is fine."""
        # Each bullet is ~1.5 tokens (1 CJK char + 2 ASCII chars in "- ").
        # Budget cuts off somewhere in the middle of the list.
        result = assemble_bullets_within_budget(
            header="## H",  # ~5 tokens
            bullets=[f"- 条{i}" for i in range(20)],  # each ~2 tokens
            max_tokens=15,
        )
        assert result is not None
        text, kept = result
        assert text.startswith("## H")
        # Dropped at least some tail
        assert kept < 20

    def test_returns_none_if_even_first_bullet_does_not_fit(self) -> None:
        """Matches the section contract: no "header-only" output."""
        assert assemble_bullets_within_budget(
            header="## LongHeader " * 50,
            bullets=["- a"],
            max_tokens=5,
        ) is None

    def test_kept_count_is_authoritative_even_when_header_contains_bullet_like_pattern(
        self,
    ) -> None:
        """Regression for PR-3 self-review: callers previously derived the
        kept count via ``text.count("\\n- ")`` which lied if the header
        itself embedded that substring. The returned ``kept`` must be
        the true number of bullets emitted, regardless of header shape."""
        tricky_header = "## Rules\n- NOTE: legacy items"  # header has ``\n- ``
        result = assemble_bullets_within_budget(
            header=tricky_header,
            bullets=["- real1", "- real2"],
            max_tokens=1000,
        )
        assert result is not None
        text, kept = result
        # Naive ``\n- `` count on text is 3 (one from header, two from bullets);
        # authoritative kept count is exactly 2.
        assert kept == 2
        assert text.count("\n- ") == 3


# ---- memory_rules ------------------------------------------------------ #


class TestMemoryRules:
    def test_returns_none_when_snapshot_is_none(self) -> None:
        output = memory_rules_section.render(_ctx(snapshot=None))
        assert output.text is None

    def test_returns_none_when_no_rule_chunks(self) -> None:
        snapshot = MemorySnapshot()  # empty
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.text is None

    def test_zh_header(self) -> None:
        snapshot = MemorySnapshot(
            rule_chunks=(_chunk("以后别 force push 到 main", category="rule"),)
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot, lang="zh"))
        assert output.text is not None
        assert "## 项目规则" in output.text
        assert "以后别 force push 到 main" in output.text

    def test_en_header(self) -> None:
        snapshot = MemorySnapshot(
            rule_chunks=(_chunk("never force push to main", category="rule"),)
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot, lang="en"))
        assert output.text is not None
        assert "## Project Rules" in output.text
        assert "never force push to main" in output.text
        assert "## 项目规则" not in output.text

    def test_bullet_format(self) -> None:
        snapshot = MemorySnapshot(
            rule_chunks=(
                _chunk("rule A", category="rule"),
                _chunk("rule B", category="rule"),
            )
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        assert "\n- rule A" in output.text
        assert "\n- rule B" in output.text

    def test_metadata_records_counts(self) -> None:
        snapshot = MemorySnapshot(
            rule_chunks=tuple(
                _chunk(f"rule {i}", category="rule") for i in range(5)
            )
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.metadata["memory_rule_count"] == 5
        assert output.metadata["memory_rule_emitted"] == 5
        assert output.metadata["memory_rule_dropped"] == 0

    def test_section_priority_protected_by_critical_min(self) -> None:
        """M2-PR3 design: rules are hard constraints — priority=8 means
        they survive budget pressure in FULL mode (CRITICAL_PRIORITY_MIN=8)."""
        assert memory_rules_section.priority == 8

    def test_section_identity(self) -> None:
        s = memory_rules_section
        assert s.id == "memory_rules"
        assert s.cacheable is False
        assert s.dynamic is True
        assert s.max_tokens == MEMORY_RULE_SECTION_BUDGET

    def test_multiline_content_flattens_to_single_bullet(self) -> None:
        """Regression for PR-3 self-review: a rule saved as
        ``"line 1\\nline 2"`` must render as ONE bullet with a space,
        not two separate lines that would break the bullet structure."""
        snapshot = MemorySnapshot(
            rule_chunks=(_chunk("line 1\nline 2", category="rule"),)
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        # The sanitized bullet is on a single line
        bullet_line = [
            ln for ln in output.text.splitlines() if ln.startswith("- ")
        ]
        assert bullet_line == ["- line 1 line 2"]
        assert output.metadata["memory_rule_emitted"] == 1

    def test_empty_content_filtered_out(self) -> None:
        """All-whitespace content must not emit a bare ``"- "`` bullet.
        Regression for PR-3 self-review."""
        snapshot = MemorySnapshot(
            rule_chunks=(
                _chunk("valid rule", category="rule"),
                _chunk("   ", category="rule"),  # all whitespace → filtered
                _chunk("\n\t\n", category="rule"),  # all whitespace → filtered
            )
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        # Only the 1 valid rule emitted
        assert output.metadata["memory_rule_emitted"] == 1
        # Total count unchanged — denominator reflects input, not survivors
        assert output.metadata["memory_rule_count"] == 3
        assert output.metadata["memory_rule_dropped"] == 2
        # No bare "- " bullet in output
        assert "- \n" not in output.text and not output.text.endswith("- ")

    def test_all_empty_content_returns_none(self) -> None:
        """If every rule chunk sanitizes to empty, section emits nothing
        (no orphan header)."""
        snapshot = MemorySnapshot(
            rule_chunks=(
                _chunk("   ", category="rule"),
                _chunk("", category="rule"),
            )
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.text is None

    def test_rule_tail_dropped_when_budget_exceeded(self) -> None:
        """20 very-long rule chunks: section should emit fewer than 20
        and record the dropped count."""
        # Each content is 300 CJK chars = 450 tokens. 20 such bullets
        # would be ~9000 tokens, far beyond the 2500 cap.
        big_content = "长" * 300
        snapshot = MemorySnapshot(
            rule_chunks=tuple(
                _chunk(big_content, category="rule") for _ in range(20)
            )
        )
        output = memory_rules_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        emitted = output.metadata["memory_rule_emitted"]
        assert 0 < emitted < 20
        assert output.metadata["memory_rule_dropped"] == 20 - emitted


# ---- memory_user_profile ----------------------------------------------- #


class TestMemoryUserProfile:
    def test_returns_none_when_snapshot_is_none(self) -> None:
        output = memory_user_profile_section.render(_ctx(snapshot=None))
        assert output.text is None

    def test_returns_none_when_no_user_chunks(self) -> None:
        output = memory_user_profile_section.render(
            _ctx(snapshot=MemorySnapshot())
        )
        assert output.text is None

    def test_zh_header_mentions_pin_marker(self) -> None:
        """Header should explain the ★ marker to the LLM."""
        snapshot = MemorySnapshot(
            user_chunks=(_chunk("用户偏好 Go", category="user"),)
        )
        output = memory_user_profile_section.render(
            _ctx(snapshot=snapshot, lang="zh")
        )
        assert output.text is not None
        assert "## 用户画像" in output.text
        assert "★" in output.text  # marker explanation in header

    def test_en_header_mentions_pin_marker(self) -> None:
        snapshot = MemorySnapshot(
            user_chunks=(_chunk("prefers Go", category="user"),)
        )
        output = memory_user_profile_section.render(
            _ctx(snapshot=snapshot, lang="en")
        )
        assert output.text is not None
        assert "## User Profile" in output.text
        assert "★" in output.text

    def test_pinned_chunk_gets_star_marker(self) -> None:
        """Pinned user memories render with a leading ★ in their bullet;
        unpinned do not."""
        pinned = _chunk("永久偏好 Go", category="user", pinned=True)
        unpinned = _chunk("想学 Rust", category="user", pinned=False)
        snapshot = MemorySnapshot(user_chunks=(pinned, unpinned))
        output = memory_user_profile_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        assert "- ★ 永久偏好 Go" in output.text
        assert "- 想学 Rust" in output.text
        assert "- ★ 想学 Rust" not in output.text

    def test_metadata_reports_pinned_count(self) -> None:
        pinned = _chunk("a", category="user", pinned=True)
        unpinned = _chunk("b", category="user")
        snapshot = MemorySnapshot(user_chunks=(pinned, unpinned))
        output = memory_user_profile_section.render(_ctx(snapshot=snapshot))
        assert output.metadata["memory_user_count"] == 2
        assert output.metadata["memory_user_pinned_count"] == 1
        assert output.metadata["memory_user_emitted"] == 2

    def test_section_identity(self) -> None:
        s = memory_user_profile_section
        assert s.id == "memory_user_profile"
        assert s.priority == 7
        assert s.cacheable is False
        assert s.dynamic is True
        assert s.max_tokens == MEMORY_USER_SECTION_BUDGET

    def test_multiline_user_content_flattens(self) -> None:
        """User profile: multiline saved memory flattens on render so the
        bullet / ★-marker structure stays intact."""
        snapshot = MemorySnapshot(
            user_chunks=(
                _chunk(
                    "喜欢 Go\n10 年经验",
                    category="user",
                    pinned=True,
                ),
            )
        )
        output = memory_user_profile_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        bullet_line = [
            ln for ln in output.text.splitlines() if ln.startswith("- ")
        ]
        assert bullet_line == ["- ★ 喜欢 Go 10 年经验"]

    def test_empty_user_content_filtered(self) -> None:
        """All-whitespace user chunk is skipped; pinned-ness doesn't
        override the emptiness filter (a blank pinned bullet is useless)."""
        snapshot = MemorySnapshot(
            user_chunks=(
                _chunk("valid", category="user"),
                _chunk("  ", category="user", pinned=True),
            )
        )
        output = memory_user_profile_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        assert output.metadata["memory_user_emitted"] == 1
        # Pinned count still reflects the raw snapshot state
        assert output.metadata["memory_user_pinned_count"] == 1

    def test_tail_drop_preserves_pinned_when_they_are_first(self) -> None:
        """Pinned at front, unpinned tail: under budget pressure, the
        pinned ones must remain. (Upstream MemorySnapshot sorts
        pinned-first; this test just confirms the section's tail-drop
        preserves that ordering invariant.)"""
        big = "长" * 300  # each bullet ~450 tokens
        pinned = _chunk(f"{big}-P", category="user", pinned=True)
        unpinned_chunks = tuple(
            _chunk(f"{big}-U{i}", category="user") for i in range(5)
        )
        # Pinned at index 0, unpinned tail
        snapshot = MemorySnapshot(user_chunks=(pinned, *unpinned_chunks))
        output = memory_user_profile_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        # Pinned bullet emitted (with ★ marker)
        assert "★" in output.text
        # At least one unpinned dropped (since 6 * 450 > 1500 tokens)
        assert output.metadata["memory_user_dropped"] >= 1


# ---- memory_fact_index ------------------------------------------------- #


class TestMemoryFactIndex:
    def test_returns_none_when_snapshot_is_none(self) -> None:
        output = memory_fact_index_section.render(_ctx(snapshot=None))
        assert output.text is None

    def test_returns_none_when_no_fact_chunks(self) -> None:
        output = memory_fact_index_section.render(
            _ctx(snapshot=MemorySnapshot())
        )
        assert output.text is None

    def test_zh_header(self) -> None:
        snapshot = MemorySnapshot(
            fact_chunks=(
                _chunk("DB 是 PostgreSQL 17", category="fact", chunk_id="f1"),
            )
        )
        output = memory_fact_index_section.render(
            _ctx(snapshot=snapshot, lang="zh")
        )
        assert output.text is not None
        assert "## 事实索引" in output.text

    def test_en_header(self) -> None:
        snapshot = MemorySnapshot(
            fact_chunks=(
                _chunk("DB is PostgreSQL 17", category="fact", chunk_id="f1"),
            )
        )
        output = memory_fact_index_section.render(
            _ctx(snapshot=snapshot, lang="en")
        )
        assert output.text is not None
        assert "## Fact Index" in output.text

    def test_bullet_format_includes_id_and_short_preview(self) -> None:
        snapshot = MemorySnapshot(
            fact_chunks=(
                _chunk(
                    "数据库是 PostgreSQL 17，在 docker-compose.yml 里",
                    category="fact",
                    chunk_id="f1",
                ),
            )
        )
        output = memory_fact_index_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        # ``[f1]`` prefix, preview char-bounded (first 15 chars + ellipsis)
        assert "- [f1] " in output.text
        # Ellipsis present because content is >15 chars
        assert "…" in output.text

    def test_short_content_has_no_ellipsis(self) -> None:
        short = "DB=PG17"  # < 15 chars
        snapshot = MemorySnapshot(
            fact_chunks=(_chunk(short, category="fact", chunk_id="f1"),)
        )
        output = memory_fact_index_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        assert "- [f1] DB=PG17" in output.text
        assert "…" not in output.text

    def test_collapses_internal_whitespace(self) -> None:
        """Preview should be single-line: newlines + tab + double-space
        collapse to single space."""
        content = "line\t1\n\nline   2"
        snapshot = MemorySnapshot(
            fact_chunks=(_chunk(content, category="fact", chunk_id="f1"),)
        )
        output = memory_fact_index_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        assert "\n" not in output.text.split("- [f1] ", 1)[1].split("\n")[0]

    def test_metadata_records_counts(self) -> None:
        snapshot = MemorySnapshot(
            fact_chunks=tuple(
                _chunk(f"fact {i}", category="fact") for i in range(3)
            )
        )
        output = memory_fact_index_section.render(_ctx(snapshot=snapshot))
        assert output.metadata["memory_fact_count"] == 3
        assert output.metadata["memory_fact_emitted"] == 3
        assert output.metadata["memory_fact_dropped"] == 0

    def test_section_identity(self) -> None:
        s = memory_fact_index_section
        assert s.id == "memory_fact_index"
        assert s.priority == 5
        assert s.cacheable is False
        assert s.dynamic is True
        assert s.max_tokens == MEMORY_FACT_INDEX_BUDGET

    def test_tail_dropped_when_budget_exceeded(self) -> None:
        """50 facts with longish IDs: compact format still has tail-drop
        behavior when content + bullet markup runs past 1000 tokens."""
        # Contents are short but ~30 ascii chars each: bullet ≈ 25 chars
        # → ~7 tokens each. 50 * 7 = 350 tokens. We need bigger previews.
        long_content = "X" * 200  # preview still gets truncated to 15 chars
        # but we still need enough bullets to exceed budget. The preview
        # collapse keeps each bullet small, so we fabricate long IDs.
        snapshot = MemorySnapshot(
            fact_chunks=tuple(
                _chunk(long_content, category="fact", chunk_id="f" * 80 + str(i))
                for i in range(50)
            )
        )
        output = memory_fact_index_section.render(_ctx(snapshot=snapshot))
        assert output.text is not None
        emitted = output.metadata["memory_fact_emitted"]
        # Each bullet ~ (80 ascii id + 16 ascii brackets+space + 15 X chars + 3 ASCII ellipsis char group)
        # Bullets cumulatively > 1000 tokens → some drop.
        assert emitted < 50


# ---- Bilingual parity (smoke) ------------------------------------------ #


class TestBilingualParity:
    """Each memory section must produce recognizably different output in
    zh vs en so language dispatch is not accidentally short-circuited.
    """

    @pytest.mark.parametrize(
        "section, content",
        [
            (memory_rules_section, ("rule", "rule_chunks")),
            (memory_user_profile_section, ("user", "user_chunks")),
            (memory_fact_index_section, ("fact", "fact_chunks")),
        ],
    )
    def test_zh_and_en_headers_differ(self, section, content) -> None:
        category, field = content
        chunks = (_chunk(f"{category} content", category=category),)
        snapshot = MemorySnapshot(**{field: chunks})

        zh_output = section.render(_ctx(snapshot=snapshot, lang="zh"))
        en_output = section.render(_ctx(snapshot=snapshot, lang="en"))
        assert zh_output.text is not None
        assert en_output.text is not None
        # ZH has CJK; EN has only ASCII in header (content may be mixed)
        zh_first_line = zh_output.text.split("\n", 1)[0]
        en_first_line = en_output.text.split("\n", 1)[0]
        assert any("\u4e00" <= ch <= "\u9fff" for ch in zh_first_line)
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in en_first_line)
