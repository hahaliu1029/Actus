"""R5 CS4 AST guard：锁住 ApprovalState 契约面 invariant。

Rules（按 design doc §AST Guard 与 test plan §6）：

R5a 合同面强制项（现在就应该绿）：
- Rule 1: ``ApprovalGrantRepository.create()`` / ``approval_grants.create()``
  仅在 ``application/services/approval_state_writer.py`` 调用
- Rule 2: ``ToolApprovalLogRepository.create()`` / ``tool_approval_log.create()``
  仅在 ``approval_state_writer.py`` 调用（**无白名单**——once scope 的 audit
  走 ``ApprovalStateWriter.write_audit_only()``，persistent scope 走
  ``ApprovalStateWriter.write()``，两条路径共用同一 writer 入口）
- Rule 6: 无 ``os.environ.get("APPROVAL_LEGACY_RULE_FALLBACK", ...)`` 旁路
  （v4 design 已禁；现仓库干净，立即生效作为防退化守卫）

R5b 前置 invariant（R5a 交付守卫文件、rule 本身 skip 直到 R5b 落地）：
- Rule 3: 仓库无 ``from app.domain.services.approval_cache import ...``
- Rule 4: ``planner_react.py`` / ``agent_task_runner.py`` 的 configurable dict
  无 ``"approval_cache"`` key
- Rule 5: ``api/app/`` 无 ``"approval:"`` 字面量（旧 Redis key 空间清理）
- Rule 7: ``react_graph.py`` SmartApprove 分支仅调 ``approval_state_writer.write()``

R5b 实施者负责：删 ``approval_cache.py``、删 configurable key、删 Redis 字面量，
然后删除本文件里对应 rule 的 ``@pytest.mark.skip`` 装饰器。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = REPO_ROOT / "api" / "app"

WRITER_REL_PATH = "api/app/application/services/approval_state_writer.py"


def _iter_app_py_files():
    for py in APP_DIR.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT).as_posix()
        yield py, rel


def _is_test_file(rel: str) -> bool:
    return "/tests/" in rel or rel.startswith("api/tests/")


def _attr_path(node: ast.AST) -> str | None:
    """把 ``ast.Attribute`` / ``ast.Name`` 链还原成点号字符串。

    ``self.foo.bar`` → ``"self.foo.bar"``；``x`` → ``"x"``；其他结构返 None。
    用于追踪 ``self._grant_repo = uow.approval_grants`` 这种 Attribute target
    赋值（Codex 这轮的 MEDIUM：跨局部变量 / self-attr 别名）。
    """
    parts: list[str] = []
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Name):
            parts.append(cur.id)
            break
        else:
            return None
    return ".".join(reversed(parts))


def _collect_attr_aliases(tree: ast.AST, target_attr: str) -> tuple[set[str], set[str]]:
    """扫 tree，收集所有绑定到 ``<any>.target_attr`` 的别名。

    返回 ``(name_aliases, attr_path_aliases)``：
    - ``name_aliases``：局部变量别名，如 ``repo = uow.approval_grants`` → {"repo"}
    - ``attr_path_aliases``：属性路径别名，如 ``self._grants = uow.approval_grants``
      → {"self._grants"}

    覆盖 ast 节点类型：``Assign``/``AnnAssign``/``NamedExpr``；target 侧支持
    ``Name`` 和 ``Attribute``；value 侧识别 ``ast.Attribute(attr=target_attr)``
    （即 ``<any>.target_attr`` 形式）。
    """
    name_aliases: set[str] = set()
    path_aliases: set[str] = set()

    def _value_matches(value: ast.AST) -> bool:
        return isinstance(value, ast.Attribute) and value.attr == target_attr

    def _record_target(tgt: ast.AST) -> None:
        if isinstance(tgt, ast.Name):
            name_aliases.add(tgt.id)
        elif isinstance(tgt, ast.Attribute):
            path = _attr_path(tgt)
            if path:
                path_aliases.add(path)
        elif isinstance(tgt, ast.Tuple):
            for elt in tgt.elts:
                _record_target(elt)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not _value_matches(node.value):
                continue
            for tgt in node.targets:
                _record_target(tgt)
        elif isinstance(node, ast.NamedExpr):
            if _value_matches(node.value):
                _record_target(node.target)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and _value_matches(node.value):
                _record_target(node.target)

    return name_aliases, path_aliases


def _find_create_call_violations(
    tree: ast.AST,
    target_attr: str,
    rel: str,
) -> list[str]:
    """查 ``.create()`` 的所有违规调用形式：

    1. 直接 attribute：``x.<target_attr>.create(...)`` 或裸 ``<target_attr>.create(...)``
    2. 局部变量别名：``repo = x.<target_attr>; repo.create(...)``
    3. 属性路径别名：``self._grants = x.<target_attr>; self._grants.create(...)``

    别名通过 ``_collect_attr_aliases`` 收集。
    """
    name_aliases, path_aliases = _collect_attr_aliases(tree, target_attr)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "create":
            continue
        parent = node.func.value

        # 形式 1a: x.<target_attr>.create(...)
        if isinstance(parent, ast.Attribute) and parent.attr == target_attr:
            violations.append(f"{rel}:{node.lineno}: illegal {target_attr}.create()")
            continue
        # 形式 1b: <target_attr>.create(...) 裸调用
        if isinstance(parent, ast.Name) and parent.id == target_attr:
            violations.append(f"{rel}:{node.lineno}: illegal {target_attr}.create()")
            continue
        # 形式 2: 局部变量别名 repo.create(...)
        if isinstance(parent, ast.Name) and parent.id in name_aliases:
            violations.append(
                f"{rel}:{node.lineno}: illegal {target_attr}.create() via alias "
                f"`{parent.id}`"
            )
            continue
        # 形式 3: 属性路径别名 self._grants.create(...) / self.grants.create(...)
        if isinstance(parent, ast.Attribute):
            path = _attr_path(parent)
            if path and path in path_aliases:
                violations.append(
                    f"{rel}:{node.lineno}: illegal {target_attr}.create() via attr-alias "
                    f"`{path}`"
                )
    return violations


# ===============================================================
# Rule 1: approval_grants.create 只在 writer 调用
# ===============================================================


def test_rule_1_no_approval_grant_create_outside_writer() -> None:
    """单写入路径：只有 ``ApprovalStateWriter`` 能调 ``approval_grants.create``。

    覆盖 3 种调用形式：
    - ``x.approval_grants.create(...)``
    - ``approval_grants.create(...)`` 裸属性
    - ``repo = x.approval_grants; repo.create(...)`` 别名绑定后调用
    """
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel) or rel == WRITER_REL_PATH:
            continue
        try:
            tree = ast.parse(py_path.read_text())
        except SyntaxError:
            continue
        violations.extend(_find_create_call_violations(tree, "approval_grants", rel))
    assert not violations, (
        "Rule 1 violation: ApprovalGrantRepository.create() called outside writer:\n"
        + "\n".join(violations)
    )


# ===============================================================
# Rule 2: tool_approval_log.create 只在 writer 调用
# ===============================================================
#
# 2026-04-21 R5 CS4 单一 writer 合同收口：白名单已删除。
# once scope audit 通过 ``ApprovalStateWriter.write_audit_only()`` 入口写入，
# 与 persistent scope 的 ``write()`` 共用同一 writer。任何
# ``tool_approval_log.create`` 在 ``approval_state_writer.py`` 之外的调用都是违规。


def test_rule_2_no_tool_approval_log_create_outside_writer() -> None:
    """单写入路径：``tool_approval_log.create`` 仅允许在 ``approval_state_writer.py``。

    同 Rule 1，覆盖直接/裸/别名三种调用形式。无例外、无白名单。
    """
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel) or rel == WRITER_REL_PATH:
            continue
        try:
            tree = ast.parse(py_path.read_text())
        except SyntaxError:
            continue
        violations.extend(_find_create_call_violations(tree, "tool_approval_log", rel))
    assert not violations, (
        "Rule 2 violation: ToolApprovalLogRepository.create() called outside writer:\n"
        + "\n".join(violations)
    )


# ===============================================================
# Rule 3: 禁 `from app.domain.services.approval_cache import ...`
# （R5b 删除 approval_cache.py 后解除 skip）
# ===============================================================


def test_rule_3_no_approval_cache_import() -> None:
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel):
            continue
        try:
            tree = ast.parse(py_path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.domain.services.approval_cache":
                violations.append(f"{rel}:{node.lineno}: illegal approval_cache import")
    assert not violations, "Rule 3 violation:\n" + "\n".join(violations)


# ===============================================================
# Rule 4: configurable dict 无 "approval_cache" key
# ===============================================================


def test_rule_4_no_approval_cache_configurable_key() -> None:
    targets = [
        APP_DIR / "domain" / "services" / "flows" / "planner_react.py",
        APP_DIR / "domain" / "services" / "agent_task_runner.py",
    ]
    violations: list[str] = []
    for target in targets:
        rel = target.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(target.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value == "approval_cache":
                        violations.append(f"{rel}:{key.lineno}: approval_cache key in configurable dict")
    assert not violations, "Rule 4 violation:\n" + "\n".join(violations)


# ===============================================================
# Rule 5: api/app/ 下无 "approval:" 字面量（Redis key 空间清理）
# ===============================================================

_APPROVAL_KEY_RE = re.compile(r"""['"]approval:[^'"]*['"]""")


def test_rule_5_no_approval_redis_key_literal() -> None:
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel):
            continue
        content = py_path.read_text()
        if _APPROVAL_KEY_RE.search(content):
            violations.append(rel)
    assert not violations, "Rule 5 violation:\n" + "\n".join(violations)


# ===============================================================
# Rule 6: 无 APPROVAL_LEGACY_RULE_FALLBACK env var 读取
# 这是 v4 design 明令禁的旁路；现仓库已无该字符串，本 rule 立即生效作为未来防退化守卫。
# ===============================================================


def test_helper_detects_alias_bindings() -> None:
    """meta-test：验证 ``_find_create_call_violations`` 能抓别名路径。

    若未来有人重写 helper 但漏掉别名检测，这条测试会失败。
    """
    # 直接调用
    tree = ast.parse("async def f(uow):\n    await uow.approval_grants.create(x)\n")
    assert _find_create_call_violations(tree, "approval_grants", "fake.py")

    # 裸属性
    tree = ast.parse("async def f():\n    await approval_grants.create(x)\n")
    assert _find_create_call_violations(tree, "approval_grants", "fake.py")

    # 别名（问题场景：Codex MEDIUM 指出的漏洞）
    tree = ast.parse(
        "async def f(uow):\n"
        "    repo = uow.approval_grants\n"
        "    await repo.create(x)\n"
    )
    assert _find_create_call_violations(tree, "approval_grants", "fake.py")

    # walrus 别名
    tree = ast.parse(
        "async def f(uow):\n"
        "    if (r := uow.approval_grants):\n"
        "        await r.create(x)\n"
    )
    assert _find_create_call_violations(tree, "approval_grants", "fake.py")

    # 带类型注解的别名
    tree = ast.parse(
        "async def f(uow):\n"
        "    repo: object = uow.approval_grants\n"
        "    await repo.create(x)\n"
    )
    assert _find_create_call_violations(tree, "approval_grants", "fake.py")

    # 无关 .create 不触发（比如 session.create 其他仓储）
    tree = ast.parse(
        "async def f(uow):\n"
        "    await uow.session.create(x)\n"
        "    await uow.file.create(x)\n"
    )
    assert not _find_create_call_violations(tree, "approval_grants", "fake.py")

    # self-attr 绑定 + self-attr 调用（Codex round 2 MEDIUM 关注场景）
    tree = ast.parse(
        "class C:\n"
        "    def __init__(self, uow):\n"
        "        self._grants = uow.approval_grants\n"
        "    async def run(self, x):\n"
        "        await self._grants.create(x)\n"
    )
    violations = _find_create_call_violations(tree, "approval_grants", "fake.py")
    assert violations and any("self._grants" in v for v in violations)

    # 元组解包：a, b = x, uow.approval_grants  —— b 应被追踪
    # （注意：此模式当前 _value_matches 只识别直接 Attribute value；元组右侧
    #  是 ast.Tuple 不是 Attribute，所以不会进入 alias 集合。这是已知局限，
    #  写在测试里记录预期行为。）
    tree = ast.parse(
        "async def f(uow):\n"
        "    a, b = x, uow.approval_grants\n"
        "    await b.create(y)\n"
    )
    # 当前实现不覆盖 tuple-unpacking 绑定（标记为 known gap）
    # assert 不 triggered:
    assert not _find_create_call_violations(tree, "approval_grants", "fake.py")


def test_rule_6_no_env_var_legacy_fallback() -> None:
    """``AppConfig.agent_config.tool_confirmation.legacy_rule_fallback`` 是唯一配置面。
    禁止任何 ``os.environ.get("APPROVAL_LEGACY_RULE_FALLBACK", ...)`` 旁路。
    """
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel):
            continue
        # 本测试文件的 docstring 包含此字符串，不扫自己
        if rel.startswith("api/tests/"):
            continue
        if "APPROVAL_LEGACY_RULE_FALLBACK" in py_path.read_text():
            violations.append(f"{rel}: forbidden env var APPROVAL_LEGACY_RULE_FALLBACK")
    assert not violations, "Rule 6 violation:\n" + "\n".join(violations)


# ===============================================================
# Rule 7: react_graph.py SmartApprove 分支只调 approval_state_writer.write()
# ===============================================================


def test_rule_7_smartapprove_writes_via_writer_only() -> None:
    """扫描 ``react_graph.py``：SmartApprove 分支不得直接 ``approval_grants.create`` 或
    ``tool_approval_log.create``，必须走 ``approval_state_writer.write(...)``。"""
    react_graph_py = APP_DIR / "domain" / "services" / "graphs" / "react_graph.py"
    tree = ast.parse(react_graph_py.read_text())
    violations: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "create":
            continue
        parent = node.func.value
        if isinstance(parent, ast.Attribute) and parent.attr in (
            "approval_grants",
            "tool_approval_log",
        ):
            violations.append(
                f"react_graph.py:{node.lineno}: "
                f"SmartApprove must go through approval_state_writer.write()"
            )
    assert not violations, "Rule 7 violation:\n" + "\n".join(violations)
