"""D1a Task 25：plugin 展示名 best-effort 读取（registry 无 name 列，§3.1 冻结不扩）。

纯同步函数（便于单测）——仓库「所有 I/O async/await」硬约束由 **调用侧** 经
``asyncio.to_thread`` 包装满足（B9 聚合服务 `_build_plugin_items`）。读取仅发生在
Admin 聚合/plugin 列表路径、每 plugin 一次小文件读；失败恒 fallback `ext_id`，不冒泡、
不加缓存（YAGNI）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_PLUGIN_JSON_BYTES = 64 * 1024  # 64KiB 上限——超限直接 fallback（防超大文件读爆内存）


def read_plugin_display_name(plugins_root: Path, ext_id: str, version: str | None) -> str:
    """读 ``plugins_root/{ext_id}/{version}/plugin.json`` 的 ``name``；任何异常 → ``ext_id``。

    version=None（无版本无路径）→ 直接 fallback。文件缺失/JSON 非法/无 name/超 64KiB
    → fallback。best-effort：绝不抛出。
    """
    if not version:
        return ext_id
    try:
        path = Path(plugins_root) / ext_id / version / "plugin.json"
        with path.open("rb") as f:
            raw = f.read(_MAX_PLUGIN_JSON_BYTES + 1)
        if len(raw) > _MAX_PLUGIN_JSON_BYTES:
            return ext_id
        data = json.loads(raw.decode("utf-8"))
        name = data.get("name") if isinstance(data, dict) else None
        if isinstance(name, str) and name.strip():
            return name
        return ext_id
    except Exception:  # noqa: BLE001 — best-effort：任何异常 fallback ext_id（不冒泡）
        logger.debug("read_plugin_display_name fallback for %s", ext_id, exc_info=True)
        return ext_id
