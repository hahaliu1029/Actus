"""B9 PR-0：per-skill 诊断承载，spec §8。

`SkillDiagnostic` 是 skill 仓储枚举时的 per-skill 诊断结果：好条目携带
`skill`（`ok=True`），坏条目携带结构化 `error_code` + `relative_file`
（`ok=False`，`skill=None`）。integrity 探测与 GET 聚合消费它产出全量
条目（含损坏 skill 的 `health.state=error`）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from app.domain.models.skill import Skill

# integrity 专用 error_code 词表（spec §8，R10#1 按 kind 拆分）：
#   parse_error      — meta.json / manifest.json JSON 解析失败或 Skill 构造失败
#   missing_meta     — meta.json 缺失
#   missing_manifest — manifest.json 缺失
SkillIntegrityErrorCode = Literal["parse_error", "missing_meta", "missing_manifest"]


@dataclass(frozen=True)
class SkillDiagnostic:
    """单个 Skill 的诊断结果（B9 spec §8 P-1 钉子）。

    好条目：`ok=True` + `skill` 非 None，`error_code`/`relative_file` 为 None。
    坏条目：`ok=False` + `skill=None` + `error_code`（+ integrity 情况下的
    `relative_file`，指向坏文件的相对名 `meta.json` / `manifest.json`）。
    """

    skill_key: str                                  # skill 目录名（file repo）/ skill.id（db repo）
    ok: bool
    error_code: Optional[SkillIntegrityErrorCode] = None   # ok=True 时恒 None
    relative_file: Optional[Literal["meta.json", "manifest.json"]] = None  # ok=True 时恒 None
    skill: Optional[Skill] = None                   # ok=True 时必非 None；ok=False 时恒 None
