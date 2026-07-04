"""B10 tool display registry — 工具卡 display policy 的 domain 单一真相.

对 38 个 canonical 工具 (与 tool_source_resolver._CANONICAL_TOOL_IDENTITIES
一一对应) 静态定死三个 machine 位:

- ``display_icon``: 受控词表 (spec §3.3), FE 维护 词表→lucide 映射,
  未知值 FE 降级 "generic".
- ``read_only``: True → FE 默认折叠. 仅当工具无持久/外部状态变更副作用
  (viewport 瞬态与只读检索 GET 不计; 有进程/会话/文件写入或变更型网络
  副作用的一律不算 — spec §3.1 R11#2+R12#2 收窄定义).
- ``destructive``: True → FE 红色高亮. 仅当工具可造成不可逆系统级损害.

一致性锁 (INV-B10-3, 由单测强制): _STATIC_RISK 的 HIGH ⊆ destructive;
read_only ∩ (_STATIC_RISK ≥ MEDIUM) = ∅.

Domain-pure: 仅依赖标准库 + 同层 tool_source_resolver (INV-B10-7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.domain.services.tools.tool_source_resolver import (
    ToolSource,
    ToolSourceUnknownError,
    resolve_tool_source,
)

# spec §3.3 受控词表 — FE fallback 兜底, 词表扩展 = additive, 无需 version bump.
DISPLAY_ICON_VOCABULARY: frozenset[str] = frozenset({
    "file", "file-edit", "terminal", "browser", "search", "message",
    "memory", "mcp", "skill", "a2a", "generic",
})


@dataclass(frozen=True)
class ToolDisplayMeta:
    display_icon: str
    read_only: bool
    destructive: bool


def _meta(icon: str, *, ro: bool = False, dx: bool = False) -> ToolDisplayMeta:
    return ToolDisplayMeta(display_icon=icon, read_only=ro, destructive=dx)


# spec §4.1 全量分类表 — 38 canonical 工具逐一定死, 与
# _CANONICAL_TOOL_IDENTITIES keys 一一对应 (单测锁定).
# 边界分类论证见 spec §4.1: shell_kill_process=destructive (终止进程不可逆);
# get_mcp_tool read_only=False (激活工具到下一执行步 = session 状态副作用);
# file_view read_only=False (R6#1: MIME 探测走 default shell exec_command,
# 会 terminate 运行中进程); browser_restart/install_skill 有副作用但可恢复→中性.
_CANONICAL_DISPLAY: dict[str, ToolDisplayMeta] = {
    # native / browser (12)
    "browser_view": _meta("browser", ro=True),
    "browser_navigate": _meta("browser"),
    "browser_click": _meta("browser"),
    "browser_input": _meta("browser"),
    "browser_move_mouse": _meta("browser"),
    "browser_press_key": _meta("browser"),
    "browser_select_option": _meta("browser"),
    "browser_scroll_up": _meta("browser", ro=True),
    "browser_scroll_down": _meta("browser", ro=True),
    "browser_console_exec": _meta("browser", dx=True),
    "browser_console_view": _meta("browser", ro=True),
    "browser_restart": _meta("browser"),
    # native / shell (5)
    "shell_execute": _meta("terminal", dx=True),
    "shell_read_output": _meta("terminal", ro=True),
    "shell_wait_process": _meta("terminal", ro=True),
    "shell_write_input": _meta("terminal"),
    "shell_kill_process": _meta("terminal", dx=True),
    # native / file (7)
    "file_read": _meta("file", ro=True),
    "file_write": _meta("file-edit"),
    "file_str_replace": _meta("file-edit"),
    "file_find_in_content": _meta("file", ro=True),
    "file_find_by_name": _meta("file", ro=True),
    "file_list": _meta("file", ro=True),
    "file_view": _meta("file"),
    # native / message (2)
    "message_notify_user": _meta("message"),
    "message_ask_user": _meta("message"),
    # native / search (1)
    "search_web": _meta("search", ro=True),
    # native / memory (3)
    "memory_search": _meta("memory", ro=True),
    "memory_get": _meta("memory", ro=True),
    "memory_save": _meta("memory"),
    # a2a (2)
    "get_remote_agent_cards": _meta("a2a", ro=True),
    "call_remote_agent": _meta("a2a"),
    # mcp discovery (2)
    "list_mcp_tools": _meta("mcp", ro=True),
    "get_mcp_tool": _meta("mcp"),
    # skill creator (3) + skill guide (1)
    "brainstorm_skill": _meta("skill"),
    "generate_skill": _meta("skill"),
    "install_skill": _meta("skill"),
    "get_skill_guide": _meta("skill", ro=True),
}

# SV4: canonical 未命中时按已解析 tool_source.source 分派 family 兜底 —
# 只给 icon, 策略位恒 False (INV-B10-1(b): 不产生折叠/高亮).
# native 不在表内: native 源 canonical miss = registry 漂移, fail-closed None.
_FAMILY_FALLBACK_ICON: dict[str, str] = {
    "mcp": "mcp",
    "skill": "skill",
    "a2a": "a2a",
}


def resolve_display_meta(
    function_name: str, tool_source: Optional[ToolSource]
) -> Optional[ToolDisplayMeta]:
    """canonical 精确命中 → family 兜底 (按 tool_source.source) → None (fail-closed).

    纯函数, 永不抛出. None 语义 = FE generic icon + 不折叠不高亮 (INV-B10-1(a)).
    """
    meta = _CANONICAL_DISPLAY.get(function_name)
    if meta is not None:
        return meta
    if tool_source is not None:
        icon = _FAMILY_FALLBACK_ICON.get(tool_source.source)
        if icon is not None:
            return _meta(icon)
    return None


@dataclass(frozen=True)
class DisplayAttachment:
    """react_graph ToolEvent 构造点的 display 附着物 (spec §4.2).

    enabled=False → 全 None attachment: pre-existing 字段不变, 新 display
    policy key 序列化为 null (INV-B10-0 语义). CALLED 构造点不消费
    ``tool_source`` (保持既有实参), CALLING/RUNNING 消费它 (flag-off 时
    None = 现状).
    """

    tool_source: Optional[ToolSource]
    display_icon: Optional[str]
    read_only: Optional[bool]
    destructive: Optional[bool]


_DISABLED_ATTACHMENT = DisplayAttachment(
    tool_source=None, display_icon=None, read_only=None, destructive=None
)


def resolve_display_attachment(
    function_name: str,
    *,
    enabled: bool,
    existing_tool_source: Optional[ToolSource] = None,
) -> DisplayAttachment:
    """enabled=True → resolver try/except 填 tool_source (existing 优先不覆盖),
    registry 命中填 display_icon/read_only/destructive.

    纯函数, 永不抛出 (INV-B10-2): 幻觉工具名 → tool_source=None →
    三元组全 None → 事件正常发射, FE generic 降级.
    """
    if not enabled:
        return _DISABLED_ATTACHMENT
    tool_source = existing_tool_source
    if tool_source is None:
        try:
            tool_source = resolve_tool_source(function_name)
        except ToolSourceUnknownError:
            tool_source = None
    meta = resolve_display_meta(function_name, tool_source)
    if meta is None:
        return DisplayAttachment(
            tool_source=tool_source,
            display_icon=None,
            read_only=None,
            destructive=None,
        )
    return DisplayAttachment(
        tool_source=tool_source,
        display_icon=meta.display_icon,
        read_only=meta.read_only,
        destructive=meta.destructive,
    )
