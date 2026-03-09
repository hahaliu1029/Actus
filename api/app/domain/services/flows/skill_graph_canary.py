"""Skill 创建子图灰度分桶工具。

按 user_id 哈希分桶，决定是否启用子图流程。
"""

from __future__ import annotations

import hashlib


def is_skill_graph_enabled(user_id: str, canary_percent: int) -> bool:
    """判断指定用户是否命中子图灰度。

    Parameters
    ----------
    user_id : 用户 ID。
    canary_percent : 灰度百分比（0-100）。0 = 全部关闭，100 = 全部启用。

    Returns
    -------
    bool : 是否启用子图。
    """
    if canary_percent <= 0:
        return False
    if canary_percent >= 100:
        return True

    # 使用 MD5 哈希取模，确保同一 user_id 始终得到相同结果
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return bucket < canary_percent
