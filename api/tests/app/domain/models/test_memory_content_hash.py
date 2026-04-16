"""Tests for the shared memory_content_hash() function."""
from __future__ import annotations

import hashlib

from app.domain.models.memory_chunk import memory_content_hash


class TestMemoryContentHash:
    def test_returns_sha256_hex(self) -> None:
        result = memory_content_hash("hello")
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 64

    def test_empty_string(self) -> None:
        result = memory_content_hash("")
        expected = hashlib.sha256("".encode("utf-8")).hexdigest()
        assert result == expected

    def test_cjk_content(self) -> None:
        text = "这是一段中文记忆内容"
        result = memory_content_hash(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_deterministic(self) -> None:
        assert memory_content_hash("abc") == memory_content_hash("abc")

    def test_different_input_different_hash(self) -> None:
        assert memory_content_hash("a") != memory_content_hash("b")
