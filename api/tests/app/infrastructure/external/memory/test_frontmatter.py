"""frontmatter 序列化测试（PR-5A）。

覆盖：
- Canonical key ordering 稳定
- body 为空时文件只有 frontmatter 块
- Unicode / emoji 全通透
- 未知 key 追加在 canonical key 之后，不丢
- YAML 特殊字符（冒号、破折号开头的 title）正确引号化
"""
from __future__ import annotations

import pytest
import yaml

from app.infrastructure.external.memory.frontmatter import (
    FRONTMATTER_KEYS,
    serialize_memory_file,
)


class TestSerializeMemoryFile:
    def test_canonical_key_order_is_stable(self):
        """字段顺序由 FRONTMATTER_KEYS 决定，输入顺序不影响输出。"""
        # 故意把 id 和 title 放在最后
        fm = {
            "tags": ["go"],
            "pinned": False,
            "updated_at": "2026-04-17T09:50:00Z",
            "created_at": "2026-04-17T09:48:57Z",
            "source": "manual",
            "category": "user",
            "title": "用户偏好",
            "id": "01HXYZ",
        }
        out = serialize_memory_file(fm, "body")

        # Extract keys from rendered YAML block in order they appear
        yaml_block = out.split("---\n")[1]  # between first and second ---
        parsed_keys = [
            line.split(":")[0]
            for line in yaml_block.strip().split("\n")
            if not line.startswith("-") and ":" in line and not line.startswith(" ")
        ]
        # Canonical order: id, title, category, source, created_at, updated_at, tags, pinned
        assert parsed_keys == list(FRONTMATTER_KEYS)

    def test_empty_body_produces_only_frontmatter(self):
        fm = {"id": "x", "category": "user"}
        out = serialize_memory_file(fm, "")
        # Structure: "---\n<yaml>\n---\n\n"（trailing newline, no body section）
        assert out.startswith("---\n")
        assert out.count("---\n") == 2  # 两个 fence
        assert out.endswith("\n")
        # 解析回字典: str.split("---") → ['', '<yaml>', '<body>']
        _, yaml_part, rest = out.split("---")
        loaded = yaml.safe_load(yaml_part)
        assert loaded == {"id": "x", "category": "user"}
        assert rest.strip() == ""

    def test_body_trailing_newline_deduplicated(self):
        """body 末尾多个 \\n 归一成单个——避免反复写入撑膨胀。"""
        fm = {"id": "x"}
        out1 = serialize_memory_file(fm, "hello")
        out2 = serialize_memory_file(fm, "hello\n")
        out3 = serialize_memory_file(fm, "hello\n\n\n")
        assert out1 == out2 == out3

    def test_unicode_and_emoji_roundtrip(self):
        fm = {
            "id": "a",
            "title": "中文标题 🎉",
            "category": "user",
        }
        out = serialize_memory_file(fm, "body 中文 emoji 🚀")
        # yaml.safe_dump allow_unicode=True 保证不转义
        assert "中文标题 🎉" in out
        assert "body 中文 emoji 🚀" in out

    def test_unknown_keys_appended_after_canonical(self):
        """扩展字段（比如 future ``auto_promoted_at``）不会丢，但放在 canonical
        之后——新字段要进 FRONTMATTER_KEYS 才拿到 canonical 槽位。"""
        fm = {
            "auto_promoted_at": "2026-04-17T10:00:00Z",  # unknown
            "id": "a",
            "category": "user",
        }
        out = serialize_memory_file(fm, "")
        # id 和 category 在 canonical 前面
        id_pos = out.index("id:")
        category_pos = out.index("category:")
        unknown_pos = out.index("auto_promoted_at:")
        assert id_pos < category_pos < unknown_pos

    def test_yaml_special_chars_are_quoted(self):
        """title 里含冒号不会破坏 YAML 语法——yaml.dump 自动加引号。"""
        fm = {"id": "a", "title": "这是: 一个 test"}
        out = serialize_memory_file(fm, "b")
        # 反解析仍等
        yaml_block = out.split("---")[1]
        loaded = yaml.safe_load(yaml_block)
        assert loaded["title"] == "这是: 一个 test"

    def test_tags_rendered_as_yaml_list(self):
        fm = {"id": "a", "tags": ["go", "react"]}
        out = serialize_memory_file(fm, "")
        # yaml.dump(default_flow_style=False) 走块式 "- item"
        assert "- go" in out
        assert "- react" in out

    def test_empty_tags_still_included(self):
        """tags=[] 是合法状态（用户显式清空），序列化保留。"""
        fm = {"id": "a", "tags": []}
        out = serialize_memory_file(fm, "")
        yaml_block = out.split("---")[1]
        loaded = yaml.safe_load(yaml_block)
        assert loaded["tags"] == []

    def test_pinned_false_preserved(self):
        """``sort_keys=False`` + explicit False 不能被 dump 丢——设计里
        pinned 字段是 CHECK 约束的一部分，必须出现在文件里。"""
        fm = {"id": "a", "pinned": False}
        out = serialize_memory_file(fm, "")
        assert "pinned: false" in out
