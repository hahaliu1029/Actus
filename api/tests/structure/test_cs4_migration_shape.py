"""R5 CS4 migration 形状检查（无 DB 依赖）。

覆盖 test plan §1/§5 里不需要真实 DB 的 3 条：
- ``test_r5_migration_only_ddl``：r5 migration 只含 DDL，**无 INSERT / bulk_insert / op.execute 写路径**
- ``test_r5_alembic_chain_head``：r5 挂到 memory chain 的当前 head（``m3_memory_system_notifications``）
- ``test_backfill_not_in_alembic_chain``：``alembic/versions/`` 下无 ``backfill`` 名字的 migration（脱离 alembic chain 验证）
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERSIONS_DIR = REPO_ROOT / "api" / "alembic" / "versions"
R5_MIGRATION = VERSIONS_DIR / "r5_add_tool_approval_grants.py"


def test_r5_migration_only_ddl() -> None:
    """r5 migration 只含 DDL：create_table / add_column / create_index /
    create_foreign_key / drop_* 等。禁 ``op.execute`` / ``op.bulk_insert`` /
    文本 ``INSERT`` SQL 字面量（这些是 data migration，设计明确脱离 alembic）。"""
    tree = ast.parse(R5_MIGRATION.read_text())

    # 禁用的 alembic op 函数名
    FORBIDDEN_OP_CALLS = {"execute", "bulk_insert"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # op.execute / op.bulk_insert
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
            and node.func.attr in FORBIDDEN_OP_CALLS
        ):
            violations.append(
                f"{R5_MIGRATION.name}:{node.lineno}: op.{node.func.attr}() is data migration"
            )

    # 文本 INSERT 字面量（sa.text("INSERT ...")）
    content = R5_MIGRATION.read_text()
    for match in re.finditer(r"""['"]\s*INSERT\s+INTO""", content, re.IGNORECASE):
        violations.append(
            f"{R5_MIGRATION.name}: INSERT literal at offset {match.start()} is data migration"
        )

    assert not violations, (
        "r5 migration must be DDL-only (backfill is a separate CLI, not alembic):\n"
        + "\n".join(violations)
    )


def test_r5_alembic_chain_head() -> None:
    """r5 migration 的 ``down_revision`` 锚定到 memory chain 当前 head。

    r5 挂在 ``m3_memory_system_notifications`` 后：migration 自动 upgrade 时
    r5 位于正确的 chain 位置，不会与未来的 memory migration 分叉。
    """
    tree = ast.parse(R5_MIGRATION.read_text())
    revision: str | None = None
    down_revision: object = object()  # sentinel
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "revision" and isinstance(node.value, ast.Constant):
            revision = node.value.value
        elif node.target.id == "down_revision" and isinstance(node.value, ast.Constant):
            down_revision = node.value.value

    assert revision == "r5_add_tool_approval_grants"
    assert down_revision == "m3_memory_system_notifications", (
        f"r5 must chain after m3_memory_system_notifications, got {down_revision!r}"
    )


def test_backfill_not_in_alembic_chain() -> None:
    """``alembic/versions/`` 下无 backfill 相关 migration（R5 backfill 脱离 chain）。"""
    offenders = [
        p.name
        for p in VERSIONS_DIR.glob("*.py")
        if "backfill" in p.name.lower() and p.name != "__init__.py"
    ]
    assert not offenders, (
        "Backfill must live in app/cli/, NOT in alembic/versions/ (design doc §Distribution Plan step 3):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# PE-4d2: guarded forward-only drop of the legacy tool_approval_rules table.
# These tests run WITHOUT a DB — they AST/text-inspect the migration file to
# lock its shape (down_revision, forward-only downgrade, count-and-abort guard,
# and the do-NOT-name-with-backfill constraint that keeps
# test_backfill_not_in_alembic_chain green).
# ---------------------------------------------------------------------------

PE4D2_MIGRATION = VERSIONS_DIR / "pe4d2_drop_tool_approval_rules.py"


def _pe4d2_module_constants() -> dict[str, object]:
    """Parse revision/down_revision module-level assignments from the PE-4d2
    migration without importing it (avoids alembic env side effects)."""
    tree = ast.parse(PE4D2_MIGRATION.read_text())
    consts: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Constant):
            consts[node.targets[0].id] = node.value.value
    return consts


def test_pe4d2_migration_exists() -> None:
    """PE-4d2 drop migration present at the expected path."""
    assert PE4D2_MIGRATION.exists(), (
        "PE-4d2 guarded drop-table migration expected at "
        "api/alembic/versions/pe4d2_drop_tool_approval_rules.py"
    )


def test_pe4d2_filename_has_no_backfill_substring() -> None:
    """The drop migration must NOT contain 'backfill' in its filename — otherwise
    test_backfill_not_in_alembic_chain (above) goes red."""
    assert "backfill" not in PE4D2_MIGRATION.name.lower()


def test_pe4d2_down_revision_is_current_head() -> None:
    """PE-4d2 chains directly after the current alembic head c2pr7_envelope_store.

    Re-verify the head with `uv run alembic heads` if this ever drifts; the head
    is the revision id, not the filename
    (c2pr7_add_coordinator_result_envelope_store.py defines revision
    'c2pr7_envelope_store')."""
    consts = _pe4d2_module_constants()
    assert consts.get("revision") == "pe4d2_drop_tool_approval_rules"
    assert "backfill" not in str(consts.get("revision", "")).lower()
    assert consts.get("down_revision") == "c2pr7_envelope_store", (
        f"PE-4d2 must chain after c2pr7_envelope_store, "
        f"got {consts.get('down_revision')!r}"
    )


def test_pe4d2_downgrade_is_forward_only() -> None:
    """downgrade() must raise (forward-only; restore from a DB snapshot)."""
    tree = ast.parse(PE4D2_MIGRATION.read_text())
    downgrade_fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "downgrade"
        ),
        None,
    )
    assert downgrade_fn is not None, "PE-4d2 must define downgrade()"
    raises = [n for n in ast.walk(downgrade_fn) if isinstance(n, ast.Raise)]
    assert raises, "downgrade() must raise (forward-only)"


def test_pe4d2_upgrade_guards_then_drops() -> None:
    """upgrade() must (a) count un-migrated rules and abort, AND (b) drop the
    table — and the count query must use the rule→grant anti-join key."""
    content = PE4D2_MIGRATION.read_text()
    tree = ast.parse(content)
    upgrade_fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "upgrade"
        ),
        None,
    )
    assert upgrade_fn is not None, "PE-4d2 must define upgrade()"

    # (a) count-and-abort guard present
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(upgrade_fn))
    assert has_raise, "upgrade() must raise (abort) when un-migrated rules exist"

    # (b) drops the table
    drops = [
        n
        for n in ast.walk(upgrade_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "op"
        and n.func.attr == "drop_table"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "tool_approval_rules"
    ]
    assert drops, "upgrade() must op.drop_table('tool_approval_rules')"

    # (c) count query uses the rule→grant NOT EXISTS anti-join key (the 5-key
    #     join that defines an un-migrated row for the IRREVERSIBLE drop guard).
    for predicate in (
        "g.user_id = r.user_id",
        "g.tool_name = r.tool_name",
        "g.primary_arg = r.command_pattern",
        "g.dir_arg = COALESCE(NULLIF(r.dir_pattern, ''), '')",
        "g.scope = 'always'",
    ):
        assert predicate in content, (
            f"PE-4d2 count guard missing backfill join predicate: {predicate!r}"
        )
    assert "tool_approval_grants" in content, (
        "count guard must reference tool_approval_grants (NOT EXISTS subquery)"
    )


# ---------------------------------------------------------------------------
# PE-4d2 (codex R1 P2 hardening): the substring shape test above proves the 5
# join predicates appear *somewhere in the file* — a comment or an extra wrong
# predicate could satisfy it, and it cannot catch op.drop_table being reordered
# ahead of the guard. For an IRREVERSIBLE drop, the safety contract deserves a
# real source-level + ordering check. These two tests:
#   1. extract the ACTUAL SQL string passed to text(...) inside upgrade()
#      (so comments/docstrings cannot satisfy it) and assert the count-guard
#      anti-join constrains EXACTLY the 5 grant columns — no extra/missing
#      predicate (a drift would break the abort→backfill→re-run remediation loop
#      the migration's abort message promised; this also re-covers the
#      key-identity assertion lost when the r5 backfill integration test was
#      deleted). PE-4d3 retired the backfill CLI, so the former cross-file half
#      of this contract (mirroring the CLI's _SELECT_LEGACY_SQL) is gone — the
#      migration-guard half remains the live safety contract.
#   2. AST-assert the guard raise precedes op.drop_table in upgrade()'s body.
# ---------------------------------------------------------------------------


def _first_text_call_sql(node: ast.AST) -> str | None:
    """Return the string Constant arg of the first ``text(...)`` / ``sa.text(...)``
    call found under ``node``. Adjacent string literals are folded into a single
    ``ast.Constant`` by the parser, so a multi-line concatenated SQL string is
    one Constant — comments are NOT part of that value."""
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        is_text = (isinstance(fn, ast.Name) and fn.id == "text") or (
            isinstance(fn, ast.Attribute) and fn.attr == "text"
        )
        if (
            is_text
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        ):
            return n.args[0].value
    return None


def test_pe4d2_guard_sql_constrains_exactly_five_grant_columns() -> None:
    """STRONGER than the substring shape test: pull the real SQL string from
    text(...) inside upgrade() (comments/docstrings excluded) and assert the
    count-guard anti-join constrains EXACTLY the 5 grant columns.

    PE-4d3 retired the backfill CLI, so the former cross-file half of this
    contract (mirroring the CLI's ``_SELECT_LEGACY_SQL``) is gone; the
    migration-guard half — proving the IRREVERSIBLE drop's safety predicate is
    exactly the 5-key anti-join — remains the live contract."""
    mig_tree = ast.parse(PE4D2_MIGRATION.read_text())
    upgrade_fn = next(
        (
            n
            for n in ast.walk(mig_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "upgrade"
        ),
        None,
    )
    assert upgrade_fn is not None, "PE-4d2 must define upgrade()"
    guard_sql = _first_text_call_sql(upgrade_fn)
    assert guard_sql is not None, (
        "upgrade() must build its count guard via text(...) (not found in the "
        "executed body — a predicate living only in a comment does not count)"
    )

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    guard_n = _norm(guard_sql)
    for predicate in (
        "g.user_id = r.user_id",
        "g.tool_name = r.tool_name",
        "g.primary_arg = r.command_pattern",
        "g.dir_arg = COALESCE(NULLIF(r.dir_pattern, ''), '')",
        "g.scope = 'always'",
    ):
        assert predicate in guard_n, (
            f"migration guard SQL (the executed text(...), not comments) is "
            f"missing join predicate: {predicate!r}"
        )
    assert "NOT EXISTS" in guard_n.upper(), (
        "guard must be a NOT EXISTS anti-join, not a plain count"
    )
    assert "tool_approval_grants" in guard_n, (
        "guard NOT EXISTS subquery must target tool_approval_grants"
    )

    # codex R2 P2: equivalence, not mere containment. Assert the guard
    # constrains EXACTLY the 5 grant columns — no EXTRA predicate (e.g.
    # `AND g.effect = 'approve'`) and none missing. An extra column would have
    # made the (now-retired) abort→backfill→re-run loop NON-CONVERGENT; the
    # exactly-5-key shape is still the safety predicate locked in for the
    # IRREVERSIBLE drop. (Containment alone — the loop above — cannot catch an
    # ADDED predicate, only a missing one.)
    def _grant_cols(sql: str) -> set[str]:
        # left-hand grant columns of each `g.<col> = ...` anti-join predicate;
        # `\s*` makes it whitespace-insensitive. All `g.` refs live inside the
        # NOT EXISTS subquery, so no region-isolation is needed.
        return set(re.findall(r"g\.(\w+)\s*=", sql))

    expected_cols = {"user_id", "tool_name", "primary_arg", "dir_arg", "scope"}
    assert _grant_cols(guard_n) == expected_cols, (
        f"migration guard constrains grant columns {_grant_cols(guard_n)!r}, "
        f"expected EXACTLY {expected_cols!r} — an extra/missing predicate "
        f"changes the IRREVERSIBLE drop's safety guard semantics."
    )


def test_pe4d2_guard_raise_precedes_drop_table() -> None:
    """The count-and-abort guard must raise BEFORE op.drop_table runs. If the
    drop were reordered ahead of the guard, the table (and its un-migrated rows)
    would be destroyed before the safety check — defeating the guard entirely.
    A substring/text test cannot catch statement reordering; this is an AST
    statement-order assertion over upgrade()'s top-level body."""
    mig_tree = ast.parse(PE4D2_MIGRATION.read_text())
    upgrade_fn = next(
        (
            n
            for n in ast.walk(mig_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "upgrade"
        ),
        None,
    )
    assert upgrade_fn is not None, "PE-4d2 must define upgrade()"

    # codex R2 P3: match the DIRECT top-level op.drop_table statement (no
    # ast.walk descent — a drop buried in a nested helper would not be the
    # executed top-level DDL), and require it be qualified `op.drop_table`.
    def _is_op_drop_table(call: ast.AST) -> bool:
        return (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "drop_table"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "op"
        )

    drop_idx = next(
        (
            i
            for i, stmt in enumerate(upgrade_fn.body)
            if isinstance(stmt, ast.Expr) and _is_op_drop_table(stmt.value)
        ),
        None,
    )
    # The guard `raise` legitimately lives inside the `if unmigrated:` block, so
    # the raise matcher DOES descend (ast.walk) within each top-level statement.
    raise_idx = next(
        (
            i
            for i, stmt in enumerate(upgrade_fn.body)
            if any(isinstance(n, ast.Raise) for n in ast.walk(stmt))
        ),
        None,
    )
    assert drop_idx is not None, (
        "upgrade() must call op.drop_table(...) as a direct top-level statement"
    )
    assert raise_idx is not None, "upgrade() must contain a guard raise"
    assert raise_idx < drop_idx, (
        f"guard raise (top-level stmt {raise_idx}) must PRECEDE op.drop_table "
        f"(stmt {drop_idx}) — the drop must never run before the un-migrated "
        f"check, or the safety guard is dead."
    )


def test_backfill_cli_exists_test_was_removed() -> None:
    """PE-4d3: the backfill CLI is deleted, so the CLI-exists guard test
    must be removed (it would fail asserting a deleted file exists)."""
    import tests.structure.test_cs4_migration_shape as mod

    assert not hasattr(mod, "test_backfill_cli_exists_at_expected_path"), (
        "backfill_approval_grants CLI was deleted in PE-4d3; remove "
        "test_backfill_cli_exists_at_expected_path"
    )
