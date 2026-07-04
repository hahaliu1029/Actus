"""B8 query-time 记忆召回 —— 纯 query 构建函数。

无 I/O（镜像 ``memory_ranker`` 风格）。模块级依赖 stdlib-only；唯一
例外是 ``build_params_version`` 内对 ``CANDIDATE_MULTIPLIER`` 的
函数级局部 import（单源常量，与 memory_search 工具共享，spec R2#3——
放函数内避免把 langchain 拉进本模块的 import 时依赖）。
"""
from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from app.domain.models.memory_recall import RecallQueryMaterial

if TYPE_CHECKING:
    from app.domain.models.app_config import MemoryConfig

RECALL_QUERY_MAX_CHARS = 2000
"""query 总长 default 常量；config（recall_query_max_chars）是运行时权威，
provider 调用 build_recall_query 时显式传 cfg 值（spec R10#2）。"""

RECALL_ITEM_RENDER_CAP = 120
"""per-item content 截断长度。deliberately 非 config（减少参数矩阵）；
params_version 的 ``r{render_cap}`` 取此常量值。"""

_TITLE_PLACEHOLDERS = frozenset({"新对话", "Task"})
"""泛化占位 title：新 session 默认 "新对话"（session_service），planner
fallback "Task"（main_graph fallback plan）——进 query 是纯噪声。"""

_WHITESPACE_RE = re.compile(r"\s+")


def clean_session_title(title: str | None) -> str | None:
    """占位词清洗：None/空白/占位词 → None；其余 strip 后返回。"""
    if title is None:
        return None
    stripped = title.strip()
    if not stripped or stripped in _TITLE_PLACEHOLDERS:
        return None
    return stripped


def build_recall_query(
    material: RecallQueryMaterial, *, max_chars: int = RECALL_QUERY_MAX_CHARS,
) -> str:
    """拼接顺序固定（spec §5.1 canonical）：title → original_request → message。

    original_request 仅当非空（strip 后）且 != message（strip 后）；
    总长截断 max_chars。
    """
    message = material.message.strip()
    parts: list[str] = []
    title = clean_session_title(material.session_title)
    if title:
        parts.append(title)
    original = (material.original_request or "").strip()
    if original and original != message:
        parts.append(original)
    if message:
        parts.append(message)
    return "\n".join(parts)[:max_chars]


def normalize_recall_query(query: str) -> str:
    """连续空白折叠为单空格 + strip + lower（CJK 无大小写不受影响）。"""
    return _WHITESPACE_RE.sub(" ", query).strip().lower()


def compute_query_hash(normalized: str, *, params_version: str) -> str:
    return hashlib.sha256(f"{params_version}|{normalized}".encode("utf-8")).hexdigest()


def build_params_version(cfg: "MemoryConfig") -> str:
    """canonical params_version（spec R9#4 定稿）。

    "v1" 是 query-builder 版本，builder 逻辑变更时递增。任一影响检索
    排序或缓存内容形态的参数变更即改变 cache key（缓存自动失效）。
    浮点用原值 str()（config 来源稳定）。
    """
    from app.domain.services.tools.memory_tools import CANDIDATE_MULTIPLIER

    return (
        f"v1:k{cfg.recall_top_k}:t{cfg.recall_threshold}"
        f":h{cfg.half_life_days}:m{cfg.mmr_lambda}"
        f":c{CANDIDATE_MULTIPLIER}:r{RECALL_ITEM_RENDER_CAP}"
    )
