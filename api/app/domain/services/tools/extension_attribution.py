"""B9 扩展归因注册表（进程级，P-8 钉子）。

工具名 → (kind, extension_id) 的进程级映射，供统计热路径（Task 19）把一次工具
调用归因回其所属 MCP server / Skill。对齐 ``tool_source_resolver`` 惯例（模块级
私有 dict + 纯函数），但**独立于** frozen 的 ``ToolSource`` / CS3 结构门（INV-B9-3）：
本注册表不触碰 ``ToolSource``、不注册身份、不参与 ``resolve_tool_source`` 契约，
纯粹是一张归因侧表。

注册来源是**显式 bindings**（MCP `tool_server_bindings()` / Skill `_tool_bindings`），
**绝不**从工具名反解 kind/id（名字里的分隔符不承载归因语义——server 名可含 `_`，
skill 工具名可含 sha1 截断，二者都无法从名字反推真实 extension_id）。

上限保护（spec §9）：
- 每个 extension 最多 ``MAX_TOOLS_PER_EXTENSION`` 个工具；超出丢弃 + 每 extension
  只 warn 一次。
- 全注册表最多 ``MAX_REGISTRY_SIZE`` 条；超出丢弃 + 每次 warn。
- 幂等覆盖：同 tool_name 重注册不重复计数（只更新映射值）。

**不导出 reset/clear**：test-only reset 放在 ``api/tests/_extension_attribution_testing.py``
（镜像 ``tests/_tool_source_testing.py``），把清空归因的能力挡在 prod import path 外。
"""
from __future__ import annotations

import logging

from app.domain.models.runtime_extension import ExtensionKind

logger = logging.getLogger(__name__)

MAX_TOOLS_PER_EXTENSION = 256
MAX_REGISTRY_SIZE = 4096

# 模块级私有存储（名字冻结——tests/_extension_attribution_testing.py 按此 import）。
_REGISTRY: dict[str, tuple[ExtensionKind, str]] = {}  # tool_name → (kind, extension_id)
_PER_EXTENSION_COUNTS: dict[tuple[ExtensionKind, str], int] = {}  # (kind, id) → 已注册工具数

# 已对某 extension 发过 per-extension 超限告警的集合——保证每 extension 只 warn 一次。
_PER_EXTENSION_CAP_WARNED: set[tuple[ExtensionKind, str]] = set()


def register_extension_tool(
    tool_name: str, kind: ExtensionKind, extension_id: str
) -> None:
    """把一个工具名归因到 (kind, extension_id)。

    幂等覆盖（同 tool_name 重注册只更新映射值，不重复计数）；超 per-extension /
    总量上限则丢弃：
    - per-extension 超限 → 丢弃 + 每 extension 只 warn 一次。
    - 总量超限 → 丢弃 + warn。

    P3 修复——**同名换主**（same tool_name re-registers under a DIFFERENT owner，如
    skill 重装：slug 派生工具名不变但 skill.id 变了）：旧路径无条件早返回、不动
    ``_PER_EXTENSION_COUNTS``，导致旧 owner 计数不减、新 owner 从不计数，per-extension
    cap 簿记漂移。现按 (kind, ext_id) 判主：
    - 同主 → 保持纯幂等早返回（配额零成本）。
    - 换主 → 旧 owner 计数 -1（floor 0；归 0 剔 key + 从 CAP_WARNED 丢弃以便日后重超限
      能再 warn），再走**正常 add 路径**给新 owner（两道 cap 都要过；若新 owner 已达
      cap 则按既有 refuse 语义拒绝该次 re-bind 并 warn——此拒绝分支同时移除陈旧的
      ``_REGISTRY[tool_name]`` 条目，避免 ``resolve_extension`` 继续指向旧 owner）。

    绝不抛异常（调用点已 fail-open 包裹，但这里同样零抛以防污染热路径）。
    """
    ext_key = (kind, extension_id)

    existing = _REGISTRY.get(tool_name)
    if existing is not None:
        # 同主幂等覆盖：已存在同名同 owner 条目 → 只更新映射值，配额不变。
        if existing == ext_key:
            _REGISTRY[tool_name] = ext_key
            return
        # 换主：先把旧 owner 计数减掉（floor 0 + 归 0 剔 key + 清 CAP_WARNED），
        # 再落到下面的正常 add 路径重新计入新 owner（含两道 cap 检查）。
        _decrement_owner(existing)
        # 注意：此处**不**先删 _REGISTRY[tool_name]——留到 add 成功时被覆盖，
        # 或在新 owner 撞 cap 的 refuse 分支里显式删除（见下）。

    # 全注册表总量上限。
    # 换主复用既有 tool_name 槽位（不新增条目），故不触发总量上限；仅全新 tool_name 才检查。
    if existing is None and len(_REGISTRY) >= MAX_REGISTRY_SIZE:
        logger.warning(
            "扩展归因注册表已满（%d 条，上限 %d）；丢弃 %r → %s/%s",
            len(_REGISTRY),
            MAX_REGISTRY_SIZE,
            tool_name,
            kind,
            extension_id,
        )
        return

    # 单 extension 工具数上限。
    current = _PER_EXTENSION_COUNTS.get(ext_key, 0)
    if current >= MAX_TOOLS_PER_EXTENSION:
        if ext_key not in _PER_EXTENSION_CAP_WARNED:
            _PER_EXTENSION_CAP_WARNED.add(ext_key)
            logger.warning(
                "扩展 %s/%s 归因工具数超上限（%d）；后续工具丢弃（per-extension 只告警一次）",
                kind,
                extension_id,
                MAX_TOOLS_PER_EXTENSION,
            )
        # 换主但新 owner 已满：拒绝 re-bind。此时旧 owner 计数已减掉（上面），若保留
        # 陈旧的 _REGISTRY[tool_name] 会让 resolve_extension 继续错误指向旧 owner——
        # 故显式移除该槽位，宁可归因缺失也不错误归因（设计选择：refuse 时不留悬挂映射）。
        if existing is not None:
            _REGISTRY.pop(tool_name, None)
        return

    _REGISTRY[tool_name] = ext_key
    _PER_EXTENSION_COUNTS[ext_key] = current + 1


def _decrement_owner(ext_key: tuple[ExtensionKind, str]) -> None:
    """把一个 owner 的 per-extension 计数减 1（floor 0）。

    计数只要下降就从 ``_PER_EXTENSION_CAP_WARNED`` 丢弃该 owner——若该 owner 之前正好
    卡在 cap 上并已 warn，减 1 后已跌破 cap，日后再次注册工具重新超限时应能再 warn 一次
    （否则残留的告警簿记会让它静默）。归 0 时另外剔除计数 key，保持表干净。
    """
    current = _PER_EXTENSION_COUNTS.get(ext_key, 0)
    # 计数下降即撤销该 owner 的"已告警"标记（rising-edge 语义：跌破 cap 后可再报）。
    _PER_EXTENSION_CAP_WARNED.discard(ext_key)
    new_count = current - 1
    if new_count <= 0:
        _PER_EXTENSION_COUNTS.pop(ext_key, None)
    else:
        _PER_EXTENSION_COUNTS[ext_key] = new_count


def resolve_extension(tool_name: str) -> tuple[ExtensionKind, str] | None:
    """反查工具名的归因 (kind, extension_id)；未注册返回 None。

    native 工具与 A2A 工具从不注册（native 无 extension；A2A agent id 是运行时参数，
    D9/R1#8 不注册），故对它们恒返回 None。
    """
    return _REGISTRY.get(tool_name)
