from __future__ import annotations

"""Skill selector for progressive activation."""

import re
from typing import Iterable

from app.domain.models.skill import Skill


class SkillSelector:
    """根据用户输入从技能池中选择候选技能。"""

    def __init__(self, default_top_k: int = 12) -> None:
        self._default_top_k = max(1, default_top_k)

    def select(
        self,
        skills: list[Skill],
        user_message: str,
        top_k: int | None = None,
    ) -> list[Skill]:
        if not skills:
            return []

        limit = max(1, top_k or self._default_top_k)
        message_tokens = self._tokenize(user_message)
        if not message_tokens:
            return skills[:limit]

        scored = []
        for index, skill in enumerate(skills):
            score = self._score_skill(skill, message_tokens)
            scored.append((score, -index, skill))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def _score_skill(self, skill: Skill, message_tokens: set[str]) -> int:
        context_blob = str((skill.manifest or {}).get("context_blob") or "")
        text_parts = [
            skill.name,
            skill.slug,
            skill.description,
            context_blob or str((skill.manifest or {}).get("skill_md") or ""),
        ]
        for tool in (skill.manifest or {}).get("tools", []):
            if isinstance(tool, dict):
                text_parts.append(str(tool.get("name") or ""))
                text_parts.append(str(tool.get("description") or ""))

        activation = (skill.manifest or {}).get("activation") or {}
        if isinstance(activation, dict):
            for keyword in activation.get("keywords", []) or []:
                text_parts.append(str(keyword))

        skill_tokens = self._tokenize(" ".join(text_parts))
        return len(message_tokens.intersection(skill_tokens))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", text or "").lower()
        tokens = {token for token in normalized.split() if token}
        cjk_tokens = {
            ch
            for ch in normalized
            if "\u4e00" <= ch <= "\u9fff"
        }
        return tokens.union(cjk_tokens)
