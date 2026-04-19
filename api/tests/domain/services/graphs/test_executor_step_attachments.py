"""Verify executor_node populates ``current_step.attachments`` from the final
AI message's JSON envelope, and EXECUTION_PROMPT renders the
``prior_step_outputs`` block for the next step.

Context: a run trace showed 3 plan steps each redoing web searches because
each step's HumanMessage rendered ``附件: 无`` even after the previous step
produced files. Root cause: ``executor_node`` extracts ``summary`` from the
final AI message but never extracts ``attachments`` to write them back onto
the completed step, so downstream steps see no record of prior outputs.

Fix A lives in ``main_graph.py::executor_node`` (attachments extraction +
``step.model_copy(attachments=...)``) and ``main_graph.py::_format_prior_step_outputs``
(renders completed steps' attachments as the ``{prior_step_outputs}``
placeholder in ``EXECUTION_PROMPT``).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.domain.models.plan import ExecutionStatus, Plan, Step


MAIN_GRAPH_PATH = (
    Path(__file__).resolve().parents[4]
    / "app"
    / "domain"
    / "services"
    / "graphs"
    / "main_graph.py"
)


def _main_graph_source() -> str:
    return MAIN_GRAPH_PATH.read_text(encoding="utf-8")


def _find_function(
    tree: ast.Module, name: str
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    return None


def _executor_source() -> str:
    source = _main_graph_source()
    tree = ast.parse(source)
    executor = _find_function(tree, "executor_node")
    assert executor is not None
    return ast.unparse(executor)


# ---- Fix A part 1: executor writes step.attachments -------------------- #


def test_executor_extracts_attachments_from_ai_envelope() -> None:
    """executor_node must call ``unwrap_message_envelope`` (or equivalent)
    on the last AI content to extract attachments, and pass them into
    ``step.model_copy(update=...)``."""
    src = _executor_source()

    assert "unwrap_message_envelope" in src, (
        "executor_node should use unwrap_message_envelope to extract "
        "attachments from the final AI message"
    )
    # The model_copy call for completing the step must include an
    # attachments update key.
    # We look for the pattern inside the executor body.
    assert '"attachments"' in src or "'attachments'" in src, (
        "executor_node must write attachments when completing the step"
    )


def test_executor_passes_step_attachments_to_model_copy() -> None:
    """The completion ``step.model_copy(update={...})`` call must set
    ``attachments`` alongside ``status``/``success``/``result``."""
    src = _executor_source()

    tree = ast.parse(src)
    found_model_copy_with_attachments = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "model_copy"):
            continue
        # Must have an ``update={...}`` keyword whose dict includes
        # ``status`` and ``attachments`` keys (the completion call, not
        # arbitrary unrelated ``model_copy`` invocations).
        for kw in node.keywords:
            if kw.arg != "update" or not isinstance(kw.value, ast.Dict):
                continue
            keys = {
                k.value for k in kw.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if {"status", "attachments"}.issubset(keys):
                found_model_copy_with_attachments = True
                break

    assert found_model_copy_with_attachments, (
        "executor_node must complete the step via "
        "``step.model_copy(update={..., 'status': ..., 'attachments': ...})``"
    )


# ---- Fix A part 2: prior_step_outputs is rendered ---------------------- #


def test_execution_prompt_format_includes_prior_step_outputs() -> None:
    """Both EXECUTION_PROMPT.format() sites in executor_node must pass
    ``prior_step_outputs=``."""
    src = _main_graph_source()
    # Two call sites exist (new-history path and first-step path).
    count = src.count("prior_step_outputs=")
    assert count >= 2, (
        f"EXECUTION_PROMPT.format() must be called with prior_step_outputs= "
        f"at both executor render sites (found {count} occurrences)"
    )


def test_format_prior_step_outputs_helper_exists() -> None:
    """A pure helper ``_format_prior_step_outputs(plan)`` must exist so it
    can be unit-tested and the executor can call it without re-implementing
    the formatting logic."""
    src = _main_graph_source()
    tree = ast.parse(src)
    helper = _find_function(tree, "_format_prior_step_outputs")
    assert helper is not None, (
        "main_graph must expose _format_prior_step_outputs(plan) helper"
    )


class TestFormatPriorStepOutputs:
    """Behavior of the pure helper."""

    def setup_method(self) -> None:
        # Late import to let the helper ship in the same PR as these tests.
        from app.domain.services.graphs.main_graph import _format_prior_step_outputs

        self.fn = _format_prior_step_outputs

    def test_none_plan_returns_placeholder(self) -> None:
        assert self.fn(None).strip() in {"无", "None", ""}

    def test_empty_plan_returns_placeholder(self) -> None:
        plan = Plan()
        rendered = self.fn(plan)
        assert rendered.strip() in {"无", "None", ""}

    def test_plan_with_no_completed_steps_returns_placeholder(self) -> None:
        plan = Plan(steps=[Step(description="step1")])
        assert self.fn(plan).strip() in {"无", "None", ""}

    def test_completed_step_without_attachments_is_skipped(self) -> None:
        plan = Plan(steps=[
            Step(
                description="step1",
                status=ExecutionStatus.COMPLETED,
                attachments=[],
            ),
        ])
        rendered = self.fn(plan)
        assert rendered.strip() in {"无", "None", ""}

    def test_empty_placeholder_is_localized(self) -> None:
        """Empty-state placeholder must switch by language so the EN prompt
        does not contain stray Chinese characters (Codex P2)."""
        assert self.fn(None, "zh").strip() == "无"
        assert self.fn(None, "en").strip() == "None"
        # Unknown language must not crash; fall back to the ZH placeholder.
        assert self.fn(None, "fr").strip() == "无"

    def test_single_completed_step_with_attachment(self) -> None:
        plan = Plan(steps=[
            Step(
                description="搜索国内 AI 新闻",
                status=ExecutionStatus.COMPLETED,
                success=True,
                result="产出文件如附件",
                attachments=["/home/ubuntu/a.md"],
            ),
        ])
        rendered = self.fn(plan)
        assert "/home/ubuntu/a.md" in rendered
        assert "搜索国内 AI 新闻" in rendered

    def test_multiple_completed_steps_preserve_order(self) -> None:
        plan = Plan(steps=[
            Step(
                description="step A",
                status=ExecutionStatus.COMPLETED,
                attachments=["/home/ubuntu/a.md"],
            ),
            Step(
                description="step B",
                status=ExecutionStatus.COMPLETED,
                attachments=["/home/ubuntu/b.md", "/home/ubuntu/b2.md"],
            ),
            Step(description="step C (pending)"),
        ])
        rendered = self.fn(plan)
        idx_a = rendered.find("/home/ubuntu/a.md")
        idx_b = rendered.find("/home/ubuntu/b.md")
        idx_b2 = rendered.find("/home/ubuntu/b2.md")
        assert idx_a != -1 and idx_b != -1 and idx_b2 != -1
        assert idx_a < idx_b, "completed step order must be preserved"
        # pending step must not appear in the rendered output
        assert "step C" not in rendered


# ---- Prompt templates: placeholder exists in both ZH and EN ------------ #


class TestExecutionPromptPriorOutputsPlaceholder:
    """Both ZH and EN EXECUTION_PROMPT must accept a ``prior_step_outputs``
    format arg."""

    def test_zh_template_has_placeholder(self) -> None:
        from app.domain.services.prompts.react import EXECUTION_PROMPT
        assert "{prior_step_outputs}" in EXECUTION_PROMPT

    def test_en_template_has_placeholder(self) -> None:
        from app.domain.services.prompts.en.react import EXECUTION_PROMPT
        assert "{prior_step_outputs}" in EXECUTION_PROMPT

    @pytest.mark.parametrize("lang", ["zh", "en"])
    def test_template_renders_with_prior_outputs(self, lang: str) -> None:
        from app.domain.services.prompts import get_prompt_bundle

        bundle = get_prompt_bundle(lang)
        rendered = bundle.EXECUTION_PROMPT.format(
            step="task X",
            message="msg",
            attachments="无",
            prior_step_outputs="- step1: /home/ubuntu/a.md",
            language=lang,
        )
        assert "/home/ubuntu/a.md" in rendered


# ---- Behavioral regression: long AI envelope must still yield attachments ---- #
#
# This is the runtime core of the fix. In the original run trace
# (run-019da66b-...json) each step produced an AI JSON envelope of the
# form ``{"success": true, "attachments": [...], "result": "<long md>"}``
# — but earlier code called ``unwrap_message_envelope`` on a
# ``content[:500]`` summary, which cut off envelopes where ``result``
# ran before ``attachments``. The fix parses the FULL content. This
# regression pins that contract so a future "optimization" that truncates
# first can't silently re-break it.


def _build_long_envelope_ai_content(
    attachments_at_end: bool,
    attachments: list[str],
) -> str:
    """Build a JSON envelope whose serialized form exceeds 500 chars.

    ``attachments_at_end=True`` places the ``attachments`` key after a
    long ``result`` body — the exact failure mode observed in the run
    trace. ``attachments_at_end=False`` places it first; both shapes
    must work.
    """
    import json

    long_result = (
        "# 本周 AI 重要新闻总结\n\n"
        + ("- 条目：某新闻正文很长，足以把整个 envelope 推过 500 字节。" * 30)
    )
    if attachments_at_end:
        payload = {"success": True, "result": long_result, "attachments": attachments}
    else:
        payload = {"success": True, "attachments": attachments, "result": long_result}
    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) > 600, "fixture must exceed 500 chars to be meaningful"
    return serialized


class TestLongEnvelopeAttachmentRegression:
    """Pin the runtime contract that the executor relies on."""

    def test_unwrap_extracts_attachments_when_after_long_result(self) -> None:
        """Original run trace shape: result first, attachments last."""
        from app.domain.services.json_envelope import unwrap_message_envelope

        expected = ["/home/ubuntu/本周AI重要新闻总结.md"]
        content = _build_long_envelope_ai_content(
            attachments_at_end=True,
            attachments=expected,
        )
        assert len(content) > 500

        _, attachments = unwrap_message_envelope(content)
        assert attachments == expected, (
            "Attachments placed after a long ``result`` body must still be "
            "extracted from the FULL envelope (not a 500-char summary)."
        )

    def test_unwrap_extracts_attachments_when_before_long_result(self) -> None:
        """Key order should not matter — attachments first is also valid."""
        from app.domain.services.json_envelope import unwrap_message_envelope

        expected = ["/home/ubuntu/a.md", "/home/ubuntu/b.md"]
        content = _build_long_envelope_ai_content(
            attachments_at_end=False,
            attachments=expected,
        )
        _, attachments = unwrap_message_envelope(content)
        assert attachments == expected

    def test_truncated_envelope_loses_attachments(self) -> None:
        """Negative pin: if we naïvely called ``unwrap_message_envelope``
        on a 500-char-truncated summary for the attachments-at-end shape,
        we would get back ``[]`` (no attachments). This test documents
        the exact failure mode the fix avoids.
        """
        from app.domain.services.json_envelope import unwrap_message_envelope

        expected = ["/home/ubuntu/本周AI重要新闻总结.md"]
        full = _build_long_envelope_ai_content(
            attachments_at_end=True,
            attachments=expected,
        )
        truncated = full[:500]
        _, attachments_from_truncated = unwrap_message_envelope(truncated)
        assert attachments_from_truncated == [], (
            "A 500-char summary of an ``attachments``-at-end envelope "
            "cannot yield attachments; executor must parse full content."
        )

    def test_updater_node_skips_llm_when_no_pending_step(self) -> None:
        """updater_node must not call ``structured_llm.ainvoke`` when the
        plan has no pending steps left.

        Observed failure mode: after the final step completed, updater_node
        still hit the planner LLM to produce a ``PlanUpdateResponse`` that
        would never be used. If the outer LangGraph task got cancelled
        (SSE client disconnect, watchdog abort) while that call was in
        flight, the openai/httpx read raised ``asyncio.CancelledError`` —
        which is a ``BaseException`` in py3.8+ and therefore escaped
        updater_node's ``except Exception`` block, surfacing as a run
        failure even though the step already succeeded.

        Pin the guard so future refactors can't regress it silently.
        """
        src = _main_graph_source()
        tree = ast.parse(src)
        updater = _find_function(tree, "updater_node")
        assert updater is not None
        updater_src = ast.unparse(updater)

        # The function must reference ``get_next_step`` (the source of
        # truth for "is there a pending step") before invoking the
        # planner LLM for the update.
        assert "get_next_step" in updater_src, (
            "updater_node must consult plan.get_next_step() to skip the "
            "LLM call when no pending step remains"
        )
        # And the invocation ordering must be: get_next_step check
        # appears before structured_llm.ainvoke in source order.
        idx_guard = updater_src.find("get_next_step")
        idx_invoke = updater_src.find("structured_llm.ainvoke")
        # If structured_llm.ainvoke isn't present (someone removed it),
        # the test can't make a claim — but we still need the guard to
        # exist, so keep the first assertion above.
        if idx_invoke != -1:
            assert 0 <= idx_guard < idx_invoke, (
                "updater_node must check get_next_step BEFORE invoking "
                "structured_llm.ainvoke (found guard at "
                f"{idx_guard}, invoke at {idx_invoke})"
            )

    def test_executor_parses_full_content_not_summary(self) -> None:
        """AST-level pin: executor_node must call ``unwrap_message_envelope``
        on the *full* AI content variable (the same var assigned before
        the ``[:500]`` slice). Without this ordering, the fix silently
        regresses back to the truncated-parse behavior.
        """
        src = _executor_source()
        # We expect a pattern like:
        #   content_str = msg.content ...
        #   summary = content_str[:500]
        #   _, step_attachments = unwrap_message_envelope(content_str)
        #
        # Heuristic: the name passed to unwrap_message_envelope must be
        # the same variable used BEFORE the ``[:500]`` slice, not the
        # sliced result.
        # Find the unwrap call:
        import re as _re
        m = _re.search(
            r"unwrap_message_envelope\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
            src,
        )
        assert m is not None, "executor_node must call unwrap_message_envelope"
        arg = m.group(1)
        # The argument must NOT be a name that ends with ``summary`` /
        # ``truncated`` / ``preview`` (common naming for sliced vars).
        assert not arg.endswith("summary"), (
            f"unwrap_message_envelope was called with {arg!r} — looks like "
            f"a truncated summary. Must be the full content variable."
        )
        # And the argument must appear earlier in the source than the
        # ``[:500]`` slice so we know it holds the unsliced value.
        slice_idx = src.find("[:500]")
        arg_first_idx = src.find(arg)
        assert arg_first_idx != -1 and arg_first_idx < slice_idx, (
            f"Variable {arg!r} passed to unwrap_message_envelope must be "
            f"assigned BEFORE the [:500] truncation so it holds full content."
        )
