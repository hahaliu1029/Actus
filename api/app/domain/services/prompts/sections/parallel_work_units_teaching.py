"""parallel_work_units_teaching section — flag-gated planner/updater teaching (WS0).

Injects the bilingual ``parallel_work_units`` teaching into the planner +
updater prompts ONLY when ``ACTUS_C2_COORDINATOR_ENABLED`` is truthy, so a
flag-ON real provider learns the schema (without it the flag flip is inert —
the planner never emits ``parallel_work_units``, F15). Flag-OFF →
``SectionOutput(text=None)`` so a flag-off planner never learns the schema and
cannot trip the executor ``assert_coordinator_enabled()`` hard-gate
(``main_graph.py:751-755``, load-bearing).

The teaching constants live HERE (not in ``bundles/{en,zh}.py``) to break the
construction-order trap: ``SectionRegistry.__post_init__`` eager-renders every
section against ``_FIXTURE_CTX`` at registry construction (= bundle import,
``section.py:258-285``), so the constant must be defined before the section
that references it. A lazy import inside ``_render`` does NOT help — the eager
fixture render fires at import. (R2 P1#2.)
"""
from __future__ import annotations

from app.domain.services.coordinator_feature_flag import is_coordinator_enabled
from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


# Moved verbatim from bundles/en.py (C2 PR-1 Task 1.10). Schema example keeps
# English keys so the LLM does not translate them (schema-drift guard).
PARALLEL_WORK_UNITS_TEACHING_EN = """
## Parallel Work Units (advanced — use sparingly)

When a step decomposes into independent sub-tasks (no cross-dependencies),
set `parallel_work_units` on the step instead of single ReAct task. Each
work_unit dispatches to its own restricted child agent.

Use when:
- Multiple independent file modifications (no shared mutable state)
- Multi-angle independent research
- Multi-file lint/format fixes

Do NOT use when:
- Sub-tasks share state / read each other's outputs
- Cross-step reasoning required
- Single short task (overhead not worth it)

Two-phase pattern:
- Phase 1 `exploration`: child reads, returns proposed_write_plan
- Phase 2 `write`: child performs declared writes within lease

Schema:
{
  "parallel_work_units": {
    "work_units": [
      {
        "objective": "rewrite api/utils/foo.py",
        "phase": "write",
        "allowed_tools": ["file_read", "file_write"],
        "proposed_paths": [{"path": "api/utils/foo.py", "op": "modify"}]
      }
    ]
  }
}

Path contract (REQUIRED): every `proposed_paths` entry MUST be a
directory-qualified, workspace-relative path (e.g. `api/utils/foo.py`,
`workspace/notes.md`) — NEVER a bare filename like `notes.md` and never an
absolute path that resolves to the workspace root. A bare path is rejected
before any child is dispatched. For a brand-new file with no natural package,
put it in a subdirectory such as `workspace/` (e.g. `workspace/result.md`).

Hard cap: 5 work_units per step.
"""


# Moved verbatim from bundles/zh.py. schema 示例保留英文 key，避免 LLM 翻译出现 schema drift。
PARALLEL_WORK_UNITS_TEACHING_ZH = """
## 并行工作单元 (Parallel Work Units，高级用法，节制使用)

当一个 step 可以拆解成互不依赖的多个子任务（彼此不共享可变状态、不读
对方输出）时，在该 step 上设置 `parallel_work_units`，而不是只写一个
单一的 ReAct 任务。每个 work_unit 会被派发到独立、受限的子 agent。

适合使用的场景：
- 多个互不依赖的文件修改（彼此无共享可变状态）
- 多角度独立调研
- 跨文件的 lint / 格式化修复

不适合使用的场景：
- 子任务之间需要共享状态 / 读取彼此的输出
- 跨 step 的推理依赖
- 单一短任务（编排开销不值）

两阶段模式：
- Phase 1 `exploration`：子 agent 只读取，返回 proposed_write_plan
- Phase 2 `write`：子 agent 在租约内执行已声明的写操作

Schema:
{
  "parallel_work_units": {
    "work_units": [
      {
        "objective": "rewrite api/utils/foo.py",
        "phase": "write",
        "allowed_tools": ["file_read", "file_write"],
        "proposed_paths": [{"path": "api/utils/foo.py", "op": "modify"}]
      }
    ]
  }
}

路径合同（必须遵守）：每个 `proposed_paths` 路径必须是带目录的 workspace-relative
路径（例如 `api/utils/foo.py`、`workspace/notes.md`），禁止使用裸文件名（如
`notes.md`），也禁止使用解析到 workspace 根目录的绝对路径。裸路径会在派发任何
子 agent 之前被直接拒绝。新建文件若没有天然所属目录，请放到 `workspace/` 等
子目录下（例如 `workspace/result.md`）。

硬上限：每个 step 最多 5 个 work_units。
"""


def _render(ctx: RenderContext) -> SectionOutput:
    """Flag-gated: emit the bilingual teaching only when the coordinator is on."""
    if not is_coordinator_enabled():
        return SectionOutput(text=None)
    text = (
        PARALLEL_WORK_UNITS_TEACHING_EN
        if ctx.lang == "en"
        else PARALLEL_WORK_UNITS_TEACHING_ZH
    ).strip()
    return SectionOutput(text=text, metadata={"coordinator_teaching": True})


parallel_work_units_teaching_section = Section(
    id="parallel_work_units_teaching",
    # priority=8 == CRITICAL_PRIORITY_MIN (assembler.py:51) / budget.critical_priority_min
    # (budget.py:33) → PROTECTED from budget-driven dropping, so flag-on
    # enablement can't silently vanish under prompt-budget pressure (R2 P2).
    priority=8,
    cacheable=False,  # runtime-flag-gated output
    dynamic=True,
    render=_render,
)
