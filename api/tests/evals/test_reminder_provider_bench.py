"""B5 C12: Provider bench for ``<system-reminder>`` vs markdown-header rendering.

**What this is for**

C8 shipped the ``ReminderRegistry`` infrastructure and 3 inert stub
reminders, but the rollout decision was blocked on a hard open
question: does the Anthropic-specific ``<system-reminder>`` tag
actually improve attention compared to a plain markdown header
``## 重要提醒``?

The tag is known to be Anthropic-specific ephemeral syntax — Claude
recognizes it and may attenuate its attention on subsequent turns.
OpenAI/GPT models do NOT have a special handler for the tag and may
treat it as dead syntax (counted against the token budget but with
no behavioral effect, or worse: actively confused by the XML).

This bench gives us the data to decide rollout strategy:
- Anthropic tag > markdown (> 10% compliance delta): **provider-aware rendering**
- Anthropic ≈ markdown and OpenAI ≈ markdown (< 5% delta on both): **do not roll out**
- OpenAI tag < markdown: **reject tag on OpenAI path**

**What this file ships**

This file is the **bench scaffold** — data classes, dataset, runner,
judge, and a mocked pytest that exercises the whole pipeline with
fake LLMs. The real bench (actual API calls to GPT-4o and
Claude-3.5-sonnet) is gated behind the ``@pytest.mark.live_llm``
marker and skipped by default because:

1. It costs real money and requires two API keys
2. It takes 2-4 hours to run 15 scenarios × 4 variants × ~1-2 min/call
3. It's inherently non-deterministic (LLM sampling)

To run the real bench::

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export ANTHROPIC_BASE_URL=https://api.anthropic.com  # or openrouter
    pytest -m live_llm tests/evals/test_reminder_provider_bench.py

Results land in ``tests/evals/reminder_bench_results.json`` (gitignored).

**Judge design**

Simple substring match: each scenario declares an
``expected_compliance_keyword`` and the judge scores 1.0 if the
keyword appears in the LLM output (case-insensitive), else 0.0.

This is deliberately primitive — a proper LLM-as-judge would double
the API cost and introduce another source of variance. The bench
looks for coarse-grained signal (10%+ delta), and substring match
is good enough at that granularity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

import pytest


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class BenchScenario:
    """One test case in the reminder bench dataset.

    ``system_prompt_template`` must contain the literal substring
    ``"{reminder_slot}"`` — the runner substitutes it with the rendered
    reminder (tag or markdown variant).
    """

    id: str
    system_prompt_template: str
    reminder_body: str
    user_message: str
    expected_compliance_keyword: str
    description: str


@dataclass(frozen=True)
class BenchVariant:
    """A single (provider, reminder_format) combination."""

    provider: Literal["openai", "anthropic"]
    reminder_format: Literal["tag", "markdown"]

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.reminder_format}"


@dataclass(frozen=True)
class BenchRunResult:
    """One LLM call result for (scenario, variant)."""

    scenario_id: str
    variant: BenchVariant
    llm_output: str
    compliance_score: float  # 0.0 or 1.0 from substring judge
    error: str | None = None


@dataclass(frozen=True)
class BenchAggregate:
    """Aggregated results for one provider."""

    provider: str
    tag_compliance_rate: float
    markdown_compliance_rate: float
    delta: float  # tag_rate - markdown_rate (positive = tag wins)
    n_scenarios: int


# ============================================================================
# Dataset: 13 fixture scenarios
# ============================================================================


_SCENARIOS: tuple[BenchScenario, ...] = (
    BenchScenario(
        id="tool_result_priority",
        system_prompt_template=(
            "You are a task execution agent. You use tools to gather "
            "information and must trust their output.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "Tool analysis results take priority over the task description. "
            "If a tool says the image is a dashboard but the task says "
            "login page, trust the tool."
        ),
        user_message=(
            "The task description says 'analyze the login page', but "
            "image_analysis_tool returned: {type: 'dashboard', "
            "elements: ['chart', 'metrics']}. What is this image?"
        ),
        expected_compliance_keyword="dashboard",
        description=(
            "Core EXECUTION_PROMPT hardcoded reminder: tool results beat "
            "task description when they disagree. Tests whether the "
            "reminder format affects tool-result-trust compliance."
        ),
    ),
    BenchScenario(
        id="file_view_preference",
        system_prompt_template=(
            "You are a task execution agent with access to file_view and "
            "file_read tools.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "For images, PDFs, audio, and video files, use file_view "
            "instead of file_read. file_read on binary files returns garbage."
        ),
        user_message="I need to understand /tmp/report.pdf. Which tool should you call?",
        expected_compliance_keyword="file_view",
        description=(
            "C3 tools_guide_stable file_view hint. Tests whether the "
            "reminder nudges the model away from file_read on PDFs."
        ),
    ),
    BenchScenario(
        id="memory_search_trigger",
        system_prompt_template=(
            "You are a task execution agent with memory_search and "
            "memory_get tools.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "When the user mentions 'previously', 'last time', or 'we "
            "discussed before', use memory_search to find relevant "
            "historical context."
        ),
        user_message=(
            "Last time we discussed the API authentication approach. "
            "Can you remind me what we decided?"
        ),
        expected_compliance_keyword="memory_search",
        description=(
            "C3 tools_guide_stable memory hint. Tests whether the "
            "reminder triggers memory_search on temporal cue phrases."
        ),
    ),
    BenchScenario(
        id="skill_install_confirm_pause",
        system_prompt_template=(
            "You are a task execution agent that can generate and "
            "install skills.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "After calling generate_skill, you MUST wait for explicit "
            "user confirmation before calling install_skill. Show the "
            "blueprint and pause."
        ),
        user_message=(
            "You just ran generate_skill and it returned a valid "
            "blueprint. The user hasn't said anything yet. Next action?"
        ),
        expected_compliance_keyword="wait",
        description=(
            "C2 behavior_core skill creation rule. Tests the pause-gating "
            "signal that was a known C0b miss before B5."
        ),
    ),
    BenchScenario(
        id="mcp_tool_priority",
        system_prompt_template=(
            "You are a task execution agent. Available tools: "
            "mcp_notion_search, mcp_notion_create_page, browser_navigate, "
            "shell_execute.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "When a task involves a service with an MCP tool (like Notion), "
            "use the MCP tool — NOT the browser — because MCP operates via "
            "API and is more reliable."
        ),
        user_message="Create a new Notion page titled 'Meeting Notes'. Which tool do you call?",
        expected_compliance_keyword="mcp_notion_create_page",
        description=(
            "C2 behavior_core MCP priority rule. Tests whether the "
            "reminder steers the model to MCP over browser for Notion."
        ),
    ),
    BenchScenario(
        id="tool_call_one_per_iter",
        system_prompt_template=(
            "You are a task execution agent in a ReAct loop.\n\n{reminder_slot}"
        ),
        reminder_body="Choose only one tool call per iteration.",
        user_message=(
            "Analyze /tmp/data.csv and write a summary to /tmp/summary.md. "
            "What tools do you invoke on your next step?"
        ),
        expected_compliance_keyword="one",
        description=(
            "C2 identity section 5-step loop rule. Tests whether the "
            "reminder prevents parallel tool invocation in a single step."
        ),
    ),
    BenchScenario(
        id="message_ask_user_takeover",
        system_prompt_template=(
            "You are a task execution agent. Tools: shell_execute, "
            "browser_navigate, message_ask_user.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "When a shell or browser tool fails, you MUST call "
            "message_ask_user with suggest_user_takeover='shell' or "
            "'browser' to request user intervention."
        ),
        user_message=(
            "shell_execute just returned [TOOL_ERROR] for your last "
            "command. You have no shell alternatives. Next action?"
        ),
        expected_compliance_keyword="suggest_user_takeover",
        description=(
            "C2 behavior_core takeover rule. Tests whether the reminder "
            "surfaces the takeover parameter when tools fail."
        ),
    ),
    BenchScenario(
        id="output_json_format",
        system_prompt_template=(
            "You are a task execution agent.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "Your return value must be valid JSON matching the "
            "interface: {success: boolean, attachments: string[], "
            "result: string}."
        ),
        user_message="The task 'write a poem about autumn' is complete. What's your response format?",
        expected_compliance_keyword="success",
        description=(
            "C2 output_format section. Tests whether the reminder "
            "enforces JSON schema compliance."
        ),
    ),
    BenchScenario(
        id="no_image_hallucination",
        system_prompt_template=(
            "You are a task planner agent creating a plan from user "
            "messages.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "You CANNOT see image content — only file names. Do not "
            "describe, guess, or assert anything about images. Use "
            "generic phrasing: 'analyze the user's uploaded image'."
        ),
        user_message=(
            "The user uploaded ui_mockup.png and wrote: 'make this look "
            "like the attached screen'. Write the first step of your plan."
        ),
        expected_compliance_keyword="analyze",
        description=(
            "C6 planner image hallucination prohibition. Tests whether "
            "the reminder keeps the planner from inventing image content."
        ),
    ),
    BenchScenario(
        id="language_preservation",
        system_prompt_template=(
            "You are a task execution agent.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "Respond in the same language the user used in their message."
        ),
        user_message="请帮我查一下今天北京的天气",
        expected_compliance_keyword="天气",
        description=(
            "C2 identity language rule. Tests whether the reminder "
            "preserves the user's language even with an English-heavy system prompt."
        ),
    ),
    BenchScenario(
        id="available_tools_authority",
        system_prompt_template=(
            "You are a task execution agent. The Available Tool Summary "
            "lists exactly: shell_execute, file_read.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "The Available Tool Summary is the authoritative source for "
            "callable tools. Never call a tool that is not in the list."
        ),
        user_message="I want you to call browser_navigate to open google.com. What do you do?",
        expected_compliance_keyword="not available",
        description=(
            "C3 tools_guide_dynamic authority rule. Tests whether the "
            "reminder prevents hallucinated tool calls."
        ),
    ),
    BenchScenario(
        id="no_todo_lists",
        system_prompt_template=(
            "You are a task execution agent.\n\n{reminder_slot}"
        ),
        reminder_body=(
            "Deliver the final result directly, not a todo list, advice, "
            "or plan. Execute the task with tools."
        ),
        user_message="Summarize the concept of monads in functional programming.",
        expected_compliance_keyword="monad",
        description=(
            "C2 behavior_core direct-delivery rule. Tests whether the "
            "reminder prevents the model from producing a todo list instead "
            "of doing the work."
        ),
    ),
    BenchScenario(
        id="soft_hint_retry",
        system_prompt_template=(
            "You are a task execution agent with message_ask_user.\n\n"
            "{reminder_slot}"
        ),
        reminder_body=(
            "When message_ask_user returns SOFT_HINT, the system suggests "
            "trying tools first. If you believe user intervention is truly "
            "needed, call message_ask_user again — SOFT_HINT is not a hard block."
        ),
        user_message=(
            "You called message_ask_user and got SOFT_HINT back. You "
            "really do need the user to decide between two options. "
            "What do you do?"
        ),
        expected_compliance_keyword="again",
        description=(
            "C2 behavior_core SOFT_HINT rule. Tests whether the reminder "
            "communicates the retry semantic correctly."
        ),
    ),
)


def get_scenarios() -> list[BenchScenario]:
    """Return a copy of the bench dataset. Tests use this instead of
    mutating the module-level tuple."""
    return list(_SCENARIOS)


# ============================================================================
# Runner / Aggregate / Judge
# ============================================================================


def render_reminder_for_variant(body: str, variant: BenchVariant) -> str:
    """Apply the variant's reminder format to the reminder body.

    Uses the same envelope as ``render_reminder_block`` in the
    ``reminders`` package, but inlined here so the bench is independent
    of that module's signature (the bench should still compile if the
    reminder package is refactored).
    """
    if variant.reminder_format == "tag":
        return f"<system-reminder>{body}</system-reminder>"
    return f"## 重要提醒\n\n{body}"


def build_system_prompt(scenario: BenchScenario, variant: BenchVariant) -> str:
    """Substitute the rendered reminder into the scenario template."""
    rendered = render_reminder_for_variant(scenario.reminder_body, variant)
    return scenario.system_prompt_template.format(reminder_slot=rendered)


def substring_judge(scenario: BenchScenario, llm_output: str) -> float:
    """Return 1.0 if the compliance keyword appears (case-insensitive) in
    the output, else 0.0. This is intentionally primitive — see the
    module docstring for rationale."""
    if not llm_output:
        return 0.0
    return 1.0 if scenario.expected_compliance_keyword.lower() in llm_output.lower() else 0.0


def _default_variants() -> list[BenchVariant]:
    """The 4 variant combinations exercised by the live bench."""
    return [
        BenchVariant(provider="openai", reminder_format="tag"),
        BenchVariant(provider="openai", reminder_format="markdown"),
        BenchVariant(provider="anthropic", reminder_format="tag"),
        BenchVariant(provider="anthropic", reminder_format="markdown"),
    ]


async def run_bench(
    scenarios: list[BenchScenario],
    call_llm: Callable[[BenchScenario, BenchVariant, str], Awaitable[str]],
    judge: Callable[[BenchScenario, str], float] = substring_judge,
    variants: list[BenchVariant] | None = None,
) -> list[BenchRunResult]:
    """Run every (scenario, variant) combination and return raw results.

    - ``call_llm`` is the only provider-specific hook: given a
      (scenario, variant, system_prompt), return the LLM's response
      as a string. Real implementations build ``ActusChatModel``
      with the right provider; tests pass a fake.
    - Errors from ``call_llm`` or ``judge`` are captured in the result's
      ``error`` field; the bench ALWAYS finishes the full grid.
    """
    variants = variants or _default_variants()
    results: list[BenchRunResult] = []
    for scenario in scenarios:
        for variant in variants:
            system_prompt = build_system_prompt(scenario, variant)
            try:
                output = await call_llm(scenario, variant, system_prompt)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                results.append(
                    BenchRunResult(
                        scenario_id=scenario.id,
                        variant=variant,
                        llm_output="",
                        compliance_score=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            try:
                score = judge(scenario, output)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    BenchRunResult(
                        scenario_id=scenario.id,
                        variant=variant,
                        llm_output=output,
                        compliance_score=0.0,
                        error=f"judge error: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            results.append(
                BenchRunResult(
                    scenario_id=scenario.id,
                    variant=variant,
                    llm_output=output,
                    compliance_score=score,
                    error=None,
                )
            )
    return results


def aggregate_results(results: list[BenchRunResult]) -> dict[str, BenchAggregate]:
    """Group by provider and compute tag vs markdown compliance deltas.

    Scenarios where the LLM call errored are counted as compliance=0
    (the raw result's score is already 0.0). This is a conservative
    choice — a raise is treated as the LLM failing to follow the
    reminder, not as a missing data point.
    """
    providers: dict[str, dict[str, list[float]]] = {}
    for r in results:
        provider = r.variant.provider
        fmt = r.variant.reminder_format
        providers.setdefault(provider, {"tag": [], "markdown": []})
        providers[provider][fmt].append(r.compliance_score)

    out: dict[str, BenchAggregate] = {}
    for provider, by_format in providers.items():
        tag_scores = by_format.get("tag", [])
        md_scores = by_format.get("markdown", [])
        tag_rate = sum(tag_scores) / len(tag_scores) if tag_scores else 0.0
        md_rate = sum(md_scores) / len(md_scores) if md_scores else 0.0
        out[provider] = BenchAggregate(
            provider=provider,
            tag_compliance_rate=tag_rate,
            markdown_compliance_rate=md_rate,
            delta=tag_rate - md_rate,
            n_scenarios=max(len(tag_scores), len(md_scores)),
        )
    return out


# ============================================================================
# Tests (default): schema + mocked end-to-end
# ============================================================================


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class TestDatasetSchema:
    def test_all_scenarios_have_required_fields(self) -> None:
        scenarios = get_scenarios()
        assert len(scenarios) >= 13, (
            f"Expected at least 13 scenarios, got {len(scenarios)}"
        )
        for s in scenarios:
            assert s.id, f"scenario missing id: {s}"
            assert s.system_prompt_template.strip(), f"{s.id}: empty template"
            assert "{reminder_slot}" in s.system_prompt_template, (
                f"{s.id}: template missing {{reminder_slot}} placeholder"
            )
            assert s.reminder_body.strip(), f"{s.id}: empty reminder_body"
            assert s.user_message.strip(), f"{s.id}: empty user_message"
            assert s.expected_compliance_keyword.strip(), (
                f"{s.id}: empty expected_compliance_keyword"
            )
            assert s.description.strip(), f"{s.id}: empty description"

    def test_scenario_ids_are_unique(self) -> None:
        scenarios = get_scenarios()
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids)), (
            f"Duplicate scenario ids: {sorted(set(ids))}"
        )

    def test_default_variants_cover_4_combinations(self) -> None:
        variants = _default_variants()
        labels = {v.label for v in variants}
        assert labels == {
            "openai:tag",
            "openai:markdown",
            "anthropic:tag",
            "anthropic:markdown",
        }


class TestRendering:
    def test_tag_variant_wraps_body(self) -> None:
        v = BenchVariant(provider="anthropic", reminder_format="tag")
        out = render_reminder_for_variant("be careful", v)
        assert out == "<system-reminder>be careful</system-reminder>"

    def test_markdown_variant_uses_header(self) -> None:
        v = BenchVariant(provider="openai", reminder_format="markdown")
        out = render_reminder_for_variant("be careful", v)
        assert out == "## 重要提醒\n\nbe careful"

    def test_build_system_prompt_substitutes_reminder(self) -> None:
        scenario = BenchScenario(
            id="test",
            system_prompt_template="You are an agent.\n\n{reminder_slot}",
            reminder_body="remember X",
            user_message="hi",
            expected_compliance_keyword="ok",
            description="test",
        )
        v = BenchVariant(provider="anthropic", reminder_format="tag")
        prompt = build_system_prompt(scenario, v)
        assert "You are an agent." in prompt
        assert "<system-reminder>remember X</system-reminder>" in prompt


class TestSubstringJudge:
    def test_keyword_present_scores_one(self) -> None:
        scenario = BenchScenario(
            id="test",
            system_prompt_template="{reminder_slot}",
            reminder_body="",
            user_message="",
            expected_compliance_keyword="dashboard",
            description="",
        )
        assert substring_judge(scenario, "The image shows a dashboard.") == 1.0

    def test_keyword_absent_scores_zero(self) -> None:
        scenario = BenchScenario(
            id="test",
            system_prompt_template="{reminder_slot}",
            reminder_body="",
            user_message="",
            expected_compliance_keyword="dashboard",
            description="",
        )
        assert substring_judge(scenario, "The image shows a chart.") == 0.0

    def test_case_insensitive(self) -> None:
        scenario = BenchScenario(
            id="test",
            system_prompt_template="{reminder_slot}",
            reminder_body="",
            user_message="",
            expected_compliance_keyword="DASHBOARD",
            description="",
        )
        assert substring_judge(scenario, "A nice dashboard here") == 1.0

    def test_empty_output_scores_zero(self) -> None:
        scenario = BenchScenario(
            id="test",
            system_prompt_template="{reminder_slot}",
            reminder_body="",
            user_message="",
            expected_compliance_keyword="x",
            description="",
        )
        assert substring_judge(scenario, "") == 0.0


class TestRunnerMocked:
    async def test_full_pipeline_with_fake_llm(self) -> None:
        """End-to-end smoke test with a fake LLM that returns canned
        outputs. Verifies the runner produces one result per
        (scenario, variant) cell and that aggregation computes sane
        compliance rates."""
        scenarios = get_scenarios()
        variants = _default_variants()

        # Fake LLM: returns the expected keyword for 80% of (scenario,
        # variant) combinations, and a miss for the rest. We make the
        # miss pattern depend on (provider, reminder_format) so the
        # aggregate shows a measurable delta — simulating a real bench
        # where one variant is slightly better than the other.
        async def fake_llm(
            scenario: BenchScenario,
            variant: BenchVariant,
            system_prompt: str,
        ) -> str:
            # Anthropic + tag gets 100% compliance, everything else
            # gets 50%. This shape reproduces a realistic "tag wins
            # on Anthropic, markdown wins elsewhere" signal.
            scenario_idx = [s.id for s in scenarios].index(scenario.id)
            if variant.provider == "anthropic" and variant.reminder_format == "tag":
                return f"Response with keyword: {scenario.expected_compliance_keyword}"
            if scenario_idx % 2 == 0:
                return f"Response with keyword: {scenario.expected_compliance_keyword}"
            return "Response without the keyword."

        results = await run_bench(scenarios, fake_llm)

        # Every (scenario, variant) cell produced a result
        expected_cells = len(scenarios) * len(variants)
        assert len(results) == expected_cells

        # No errors in the fake path
        assert all(r.error is None for r in results)

        # Aggregation sanity
        aggregates = aggregate_results(results)
        assert "openai" in aggregates
        assert "anthropic" in aggregates
        assert aggregates["anthropic"].tag_compliance_rate == 1.0
        assert 0.0 < aggregates["openai"].tag_compliance_rate < 1.0

    async def test_runner_captures_call_llm_errors(self) -> None:
        """If ``call_llm`` raises, the result carries the error and the
        bench keeps going."""
        scenarios = [get_scenarios()[0]]

        call_count = {"n": 0}

        async def raising_llm(
            scenario: BenchScenario,
            variant: BenchVariant,
            system_prompt: str,
        ) -> str:
            call_count["n"] += 1
            if variant.reminder_format == "tag":
                raise RuntimeError("simulated API failure")
            return f"Response with keyword: {scenario.expected_compliance_keyword}"

        results = await run_bench(scenarios, raising_llm)
        assert len(results) == 4  # 1 scenario × 4 variants
        # Tag variants errored
        errored = [r for r in results if r.error is not None]
        assert len(errored) == 2
        assert all(r.compliance_score == 0.0 for r in errored)
        # Markdown variants succeeded
        succeeded = [r for r in results if r.error is None]
        assert len(succeeded) == 2
        assert all(r.compliance_score == 1.0 for r in succeeded)

    async def test_runner_captures_judge_errors(self) -> None:
        """If the judge raises, the result carries the error and the
        bench keeps going."""
        scenarios = [get_scenarios()[0]]

        async def ok_llm(*args: Any, **kwargs: Any) -> str:
            return "some output"

        def raising_judge(scenario: BenchScenario, output: str) -> float:
            raise ValueError("simulated judge failure")

        results = await run_bench(scenarios, ok_llm, judge=raising_judge)
        assert len(results) == 4
        assert all(r.error is not None for r in results)
        assert all("judge error" in (r.error or "") for r in results)


class TestAggregate:
    def test_aggregate_handles_empty(self) -> None:
        out = aggregate_results([])
        assert out == {}

    def test_aggregate_single_provider_perfect_tag(self) -> None:
        results = [
            BenchRunResult(
                scenario_id="s1",
                variant=BenchVariant(provider="anthropic", reminder_format="tag"),
                llm_output="",
                compliance_score=1.0,
            ),
            BenchRunResult(
                scenario_id="s1",
                variant=BenchVariant(provider="anthropic", reminder_format="markdown"),
                llm_output="",
                compliance_score=0.0,
            ),
        ]
        out = aggregate_results(results)
        assert "anthropic" in out
        assert out["anthropic"].tag_compliance_rate == 1.0
        assert out["anthropic"].markdown_compliance_rate == 0.0
        assert out["anthropic"].delta == 1.0

    def test_aggregate_groups_by_provider(self) -> None:
        results = []
        for provider in ("openai", "anthropic"):
            results.append(
                BenchRunResult(
                    scenario_id="s1",
                    variant=BenchVariant(provider=provider, reminder_format="tag"),  # type: ignore[arg-type]
                    llm_output="",
                    compliance_score=0.8,
                )
            )
            results.append(
                BenchRunResult(
                    scenario_id="s1",
                    variant=BenchVariant(provider=provider, reminder_format="markdown"),  # type: ignore[arg-type]
                    llm_output="",
                    compliance_score=0.6,
                )
            )
        out = aggregate_results(results)
        assert set(out.keys()) == {"openai", "anthropic"}
        for agg in out.values():
            assert abs(agg.delta - 0.2) < 1e-9


# ============================================================================
# Live bench runner (skipped by default)
# ============================================================================


@pytest.mark.live_llm
async def test_live_reminder_provider_bench(tmp_path) -> None:
    """Real-API bench runner. Skipped by default — run with
    ``pytest -m live_llm tests/evals/test_reminder_provider_bench.py``.

    Requires env vars:
    - ``OPENAI_API_KEY`` (required)
    - ``ANTHROPIC_API_KEY`` (required)
    - ``ANTHROPIC_BASE_URL`` (optional, defaults to OpenRouter's
      Anthropic proxy — set to ``https://api.anthropic.com`` for direct)

    Writes ``reminder_bench_results.json`` next to this file. The
    file is gitignored to keep real API responses out of the repo.
    """
    import json
    from pathlib import Path

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set — skipping live bench")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live bench")

    from langchain_core.messages import HumanMessage, SystemMessage

    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

    anthropic_base = os.environ.get(
        "ANTHROPIC_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    def _build_model(variant: BenchVariant) -> ActusChatModel:
        if variant.provider == "openai":
            return ActusChatModel(
                api_key=openai_key,
                model_name="gpt-5.4",
                base_url="https://api.openai.com/v1",
                temperature=0.0,
                max_tokens=512,
                provider_name="openai",
            )
        return ActusChatModel(
            api_key=anthropic_key,
            model_name="anthropic/claude-5.5-sonnet",
            base_url=anthropic_base,
            temperature=0.0,
            max_tokens=512,
            provider_name="anthropic",
        )

    async def call_llm(
        scenario: BenchScenario,
        variant: BenchVariant,
        system_prompt: str,
    ) -> str:
        llm = _build_model(variant)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=scenario.user_message),
        ]
        result = await llm._agenerate(messages)
        return result.generations[0].message.content or ""

    scenarios = get_scenarios()
    results = await run_bench(scenarios, call_llm)
    aggregates = aggregate_results(results)

    # Write results to disk (gitignored)
    results_path = Path(__file__).parent / "reminder_bench_results.json"
    payload = {
        "scenarios_count": len(scenarios),
        "raw_results": [
            {
                "scenario_id": r.scenario_id,
                "variant": r.variant.label,
                "compliance_score": r.compliance_score,
                "error": r.error,
                "output_preview": (r.llm_output[:200] if r.llm_output else ""),
            }
            for r in results
        ],
        "aggregates": {
            provider: {
                "tag_compliance_rate": agg.tag_compliance_rate,
                "markdown_compliance_rate": agg.markdown_compliance_rate,
                "delta": agg.delta,
                "n_scenarios": agg.n_scenarios,
            }
            for provider, agg in aggregates.items()
        },
    }
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Basic sanity assertion — the bench ran to completion
    assert len(results) == len(scenarios) * 4
    # Report aggregate to stdout (pytest -s shows it)
    for provider, agg in aggregates.items():
        print(
            f"[bench] {provider}: "
            f"tag={agg.tag_compliance_rate:.2f} "
            f"md={agg.markdown_compliance_rate:.2f} "
            f"delta={agg.delta:+.2f}"
        )
