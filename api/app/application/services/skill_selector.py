from __future__ import annotations

"""Skill selector for progressive activation."""

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from app.domain.models.skill import Skill


@dataclass(slots=True)
class SkillSelectionMeta:
    """Skill选择元信息。"""

    selected_skills: list[Skill]
    max_score: int
    second_score: int
    token_count: int
    effective_threshold: int

    @property
    def has_positive_match(self) -> bool:
        return self.max_score >= self.effective_threshold


class SkillSelector:
    """根据用户输入从技能池中选择候选技能。"""

    _SEGMENT_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")

    def __init__(self, default_top_k: int = 12, base_threshold: int = 3) -> None:
        self._default_top_k = max(1, default_top_k)
        self._base_threshold = max(1, base_threshold)

    def select(
        self,
        skills: list[Skill],
        user_message: str,
        top_k: int | None = None,
    ) -> list[Skill]:
        return self.select_with_meta(
            skills=skills,
            user_message=user_message,
            top_k=top_k,
        ).selected_skills

    def select_with_meta(
        self,
        skills: list[Skill],
        user_message: str,
        top_k: int | None = None,
    ) -> SkillSelectionMeta:
        if not skills:
            return SkillSelectionMeta(
                selected_skills=[],
                max_score=0,
                second_score=0,
                token_count=0,
                effective_threshold=1,
            )

        limit = max(1, top_k or self._default_top_k)
        message_tokens = self._tokenize(user_message)
        token_count = len(message_tokens)
        effective_threshold = self._compute_effective_threshold(token_count)

        if not message_tokens:
            return SkillSelectionMeta(
                selected_skills=skills[:limit],
                max_score=0,
                second_score=0,
                token_count=0,
                effective_threshold=effective_threshold,
            )

        scored: list[tuple[int, int, Skill]] = []
        for index, skill in enumerate(skills):
            score = self._score_skill(skill, message_tokens)
            scored.append((score, -index, skill))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_skills = [item[2] for item in scored[:limit]]
        max_score = scored[0][0] if scored else 0
        second_score = scored[1][0] if len(scored) > 1 else 0

        return SkillSelectionMeta(
            selected_skills=selected_skills,
            max_score=max_score,
            second_score=second_score,
            token_count=token_count,
            effective_threshold=effective_threshold,
        )

    def _compute_effective_threshold(self, token_count: int) -> int:
        return min(
            self._base_threshold,
            max(1, math.ceil(token_count * 0.5)),
        )

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
        normalized = SkillSelector._normalize_text(text)
        if not normalized:
            return set()

        tokens: set[str] = set()
        for segment in SkillSelector._SEGMENT_RE.findall(normalized):
            if not segment:
                continue
            if SkillSelector._is_cjk_segment(segment):
                if len(segment) == 1:
                    tokens.add(segment)
                    continue
                tokens.add(segment)
                for idx in range(len(segment) - 1):
                    tokens.add(segment[idx : idx + 2])
                continue
            tokens.add(segment)
        return tokens

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "").lower()
        normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _is_cjk_segment(segment: str) -> bool:
        return bool(segment) and all("\u4e00" <= ch <= "\u9fff" for ch in segment)
