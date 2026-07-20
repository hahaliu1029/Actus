"""C4 PR-3 — A2A 适配器 + 防御式解析 helper 测试（spec §5 + §7 PR-3）。"""
from __future__ import annotations

from app.application.services.a2a_subagent_worker import (
    MAX_A2A_ARTIFACTS,
    MAX_A2A_ERROR_CHARS,
    MAX_A2A_PARTS,
    MAX_A2A_SUMMARY_CHARS,
    MAX_PREVIEW_CHARS,
    _redact_error,
    _sanitize_summary,
    bounded_preview,
    collect_artifact_text,
    iter_text_parts,
    parts_to_summary,
    safe_get,
    safe_list,
)
from app.application.services.a2a_subagent_worker import extract_summary

import pytest

from app.application.services.a2a_subagent_worker import A2aSubagentWorker
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerSpec,
    WorkerTerminalOutcome,
)
from app.domain.models.tool_result import ToolResult


class TestSafeHelpers:
    def test_safe_get_nested(self) -> None:
        assert safe_get({"a": {"b": {"c": 1}}}, "a", "b", "c") == 1

    def test_safe_get_non_mapping_midpath(self) -> None:
        assert safe_get({"a": 5}, "a", "b") is None
        assert safe_get(None, "a") is None
        assert safe_get("str", "a") is None

    def test_safe_get_missing_key(self) -> None:
        assert safe_get({"a": 1}, "z") is None

    def test_safe_list(self) -> None:
        assert safe_list([1, 2]) == [1, 2]
        assert safe_list("nope") == []
        assert safe_list(None) == []
        assert safe_list({"a": 1}) == []


class TestPartsExtraction:
    def test_iter_text_parts_collects_text(self) -> None:
        parts = [{"text": "a"}, {"text": "b"}, {"kind": "no-text"}, "scalar"]
        assert iter_text_parts(parts) == ["a", "b"]

    def test_iter_text_parts_non_list(self) -> None:
        assert iter_text_parts(None) == []
        assert iter_text_parts({"text": "x"}) == []

    def test_iter_text_parts_caps_at_max_parts(self) -> None:
        parts = [{"text": "x"}] * (MAX_A2A_PARTS + 10)
        assert len(iter_text_parts(parts)) == MAX_A2A_PARTS

    def test_parts_to_summary_joins_and_caps(self) -> None:
        assert parts_to_summary(["a", "b"]) == "a\nb"
        big = ["x" * MAX_A2A_SUMMARY_CHARS, "y" * 100]
        assert len(parts_to_summary(big)) == MAX_A2A_SUMMARY_CHARS

    def test_collect_artifact_text_double_bounded(self) -> None:
        arts = [{"parts": [{"text": "a"}]}, {"parts": [{"text": "b"}]}]
        assert collect_artifact_text(arts) == ["a", "b"]

    def test_collect_artifact_text_artifact_cap(self) -> None:
        arts = [{"parts": [{"text": "x"}]}] * (MAX_A2A_ARTIFACTS + 50)
        assert len(collect_artifact_text(arts)) <= MAX_A2A_ARTIFACTS

    def test_collect_artifact_text_global_part_cap(self) -> None:
        arts = [{"parts": [{"text": "x"}] * (MAX_A2A_PARTS + 100)}]
        assert len(collect_artifact_text(arts)) <= MAX_A2A_PARTS

    def test_collect_artifact_text_zero_part_artifacts_bounded(self) -> None:
        # 巨大 artifacts、每个 0 part：外层 islice 保证有界（不挂）。
        arts = [{"parts": []}] * 100_000
        assert collect_artifact_text(arts) == []

    def test_collect_artifact_text_malformed(self) -> None:
        assert collect_artifact_text(None) == []
        assert collect_artifact_text([{"no_parts": 1}, "scalar", {"parts": "notlist"}]) == []

    def test_iter_text_parts_single_giant_part_truncated_in_place(self) -> None:
        # R3#P2：即时截断——单个超大 part 只持有 MAX_A2A_SUMMARY_CHARS，不持完整远端串
        out = iter_text_parts([{"text": "x" * (MAX_A2A_SUMMARY_CHARS + 100)}])
        assert len(out) == 1
        assert len(out[0]) == MAX_A2A_SUMMARY_CHARS

    def test_collect_artifact_text_single_giant_part_truncated(self) -> None:
        out = collect_artifact_text([{"parts": [{"text": "x" * (MAX_A2A_SUMMARY_CHARS + 100)}]}])
        assert sum(len(s) for s in out) <= MAX_A2A_SUMMARY_CHARS


class TestBoundedPreview:
    def test_simple(self) -> None:
        out = bounded_preview({"a": 1, "b": "x"})
        assert isinstance(out, str)
        assert "a" in out

    def test_caps_length(self) -> None:
        assert len(bounded_preview({"k": "v" * 100_000})) <= MAX_PREVIEW_CHARS

    def test_cycle_safe(self) -> None:
        d: dict = {"self": None}
        d["self"] = d
        assert "<cycle>" in bounded_preview(d)

    def test_deep_nesting_bounded(self) -> None:
        node: dict = {}
        cur = node
        for _ in range(50):
            child: dict = {}
            cur["child"] = child
            cur = child
        assert "<...>" in bounded_preview(node)

    def test_huge_int_not_materialized(self) -> None:
        # R6#P2-b：巨型 int 用 bit_length 判断 → "<int>"，不 str() 物化百万字符
        out = bounded_preview({"junk": 10 ** 1_000_000})
        assert "<int>" in out
        assert len(out) <= MAX_PREVIEW_CHARS

    def test_normal_scalars_preserved(self) -> None:
        out = bounded_preview({"n": 42, "f": 3.5, "b": True, "z": None})
        assert "42" in out and "3.5" in out and "true" in out and "null" in out

    def test_huge_int_key_not_materialized(self) -> None:
        # R7#P2：巨型 int KEY 也走有界化（Py3.12 str(bigint) 会抛 ValueError）
        out = bounded_preview({10 ** 1_000_000: "v"})
        assert "<int>" in out
        assert len(out) <= MAX_PREVIEW_CHARS

    def test_tuple_with_huge_int_key_bounded(self) -> None:
        out = bounded_preview({(10 ** 1_000_000,): "v"})
        assert "<int>" in out
        assert len(out) <= MAX_PREVIEW_CHARS

    def test_frozenset_with_huge_int_key_bounded(self) -> None:
        # R8：frozenset 是合法 hashable key；str(frozenset({巨型int})) 会抛 ValueError
        out = bounded_preview({frozenset({10 ** 1_000_000}): "v"})
        assert "<int>" in out
        assert len(out) <= MAX_PREVIEW_CHARS

    def test_frozenset_huge_int_value_bounded(self) -> None:
        out = bounded_preview({"k": frozenset({10 ** 1_000_000})})
        assert "<int>" in out
        assert len(out) <= MAX_PREVIEW_CHARS

    def test_arbitrary_object_with_raising_str_is_total(self) -> None:
        # R8：__str__ 抛错的对象也不破坏 bounded_preview 的"不抛"契约
        class _Boom:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        out = bounded_preview({"k": _Boom()})
        assert "<unrepr>" in out


class TestSanitizers:
    def test_sanitize_summary_preserves_newline_tab(self) -> None:
        assert _sanitize_summary("a\nb\tc") == "a\nb\tc"

    def test_sanitize_summary_strips_control(self) -> None:
        assert _sanitize_summary("a\x00\x07\rb") == "ab"

    def test_sanitize_summary_caps(self) -> None:
        assert len(_sanitize_summary("z" * (MAX_A2A_SUMMARY_CHARS + 50))) == MAX_A2A_SUMMARY_CHARS

    def test_redact_error_strips_url(self) -> None:
        out = _redact_error("failed calling https://secret.host/agent/x?token=abc now")
        assert "https://" not in out
        assert "secret.host" not in out

    def test_redact_error_strips_secret_kv(self) -> None:
        assert "Bearer_xyz" not in _redact_error("authorization=Bearer_xyz failed")

    def test_redact_error_collapses_whitespace_and_caps(self) -> None:
        assert _redact_error("a\n\n\tb") == "a b"

    def test_redact_error_bearer_with_space(self) -> None:
        # R3#P2：带空格的 Bearer <token> 也必须清除（generic kv `\S+` 漏空格后的 token）
        assert "SEKRET" not in _redact_error("authorization=Bearer SEKRET failed")
        assert "SEKRET" not in _redact_error("Authorization: Bearer SEKRET")

    def test_redact_error_does_not_over_redact_plain_english(self) -> None:
        # R4#P2：auth-上下文限定，不误伤普通英文 "bearer ..."
        assert "news" in _redact_error("the bearer of bad news")

    def test_redact_error_huge_input_bounded(self) -> None:
        # R6#P1-a：超大 error 先前缀截断，不全串扫描/物化
        assert len(_redact_error("x\n" * 5_000_000)) <= MAX_A2A_ERROR_CHARS

    def test_redact_error_non_http_and_protocol_relative_urls(self) -> None:
        # R6#P1-b：任意 scheme:// + 协议相对 // 都要剥
        assert "internal" not in _redact_error("failed ftp://internal/x")
        assert "internal" not in _redact_error("see //internal/y now")

    def test_redact_error_bearer_whitespace_separated(self) -> None:
        # R6#P1-b：tab/空白分隔的 Authorization Bearer 也要清除
        assert "SEKRET" not in _redact_error("Authorization\tBearer SEKRET")
        assert "SEKRET" not in _redact_error("authorization Bearer SEKRET")

    def test_redact_error_strips_ansi_control(self) -> None:
        # R6#P2-a：ESC/C0 控制字符不得存活（防 ANSI 注入），可读内容保留
        out = _redact_error("\x1b[2J\x1b[31mFAILED\x1b[0m")
        assert "\x1b" not in out
        assert "FAILED" in out


class TestExtractSummary:
    """spec §5.2：(a)–(e) 取首个成串非空；全空 (f) 按 state_was_parsed 二分。"""

    def test_a_status_message_parts(self) -> None:
        data = {"result": {"status": {"message": {"parts": [{"text": "hello"}]}}}}
        assert extract_summary(data, state_was_parsed=True) == ("hello", False)

    def test_b_artifacts(self) -> None:
        data = {"result": {"artifacts": [{"parts": [{"text": "art"}]}]}}
        assert extract_summary(data, state_was_parsed=False) == ("art", False)

    def test_c_message_like_parts(self) -> None:
        data = {"result": {"parts": [{"text": "msg"}]}}
        assert extract_summary(data, state_was_parsed=False) == ("msg", False)

    def test_d_payload_scalar(self) -> None:
        data = {"result": "plain answer"}
        assert extract_summary(data, state_was_parsed=False) == ("plain answer", False)

    def test_d_payload_text_key(self) -> None:
        data = {"result": {"text": "t"}}
        assert extract_summary(data, state_was_parsed=False) == ("t", False)

    def test_e_top_level_reply(self) -> None:
        # 非 conformant {"reply": ...}（引 test_r2_wrapper_integration.py:142 同形）
        data = {"reply": "task done"}
        assert extract_summary(data, state_was_parsed=False) == ("task done", False)

    def test_f_state_parsed_no_text_empty_no_unparsed(self) -> None:
        # conformant 状态（input-required）但无文本：summary="", unparsed=False
        summary, unparsed = extract_summary(
            {"result": {"status": {"state": "input-required"}}}, state_was_parsed=True
        )
        assert summary == ""
        assert unparsed is False

    def test_f_state_unparsed_no_text_preview_unparsed(self) -> None:
        # 非 conformant 且无可抽文本：preview + unparsed=True
        summary, unparsed = extract_summary(
            {"weird": {"deeply": "nested"}}, state_was_parsed=False
        )
        assert unparsed is True
        assert summary  # 非空 preview

    def test_priority_a_over_e(self) -> None:
        data = {
            "result": {"status": {"message": {"parts": [{"text": "win"}]}}},
            "reply": "lose",
        }
        assert extract_summary(data, state_was_parsed=True) == ("win", False)

    def test_e_top_level_priority_text_over_reply_over_output(self) -> None:
        # (e) 顺序：text > reply > output（spec §5.2 + §7 PR-3）
        assert extract_summary(
            {"text": "T", "reply": "R", "output": "O"}, state_was_parsed=False
        ) == ("T", False)
        assert extract_summary(
            {"reply": "R", "output": "O"}, state_was_parsed=False
        ) == ("R", False)
        assert extract_summary({"output": "O"}, state_was_parsed=False) == ("O", False)

    def test_never_raises_on_garbage(self) -> None:
        # total-robust（spec §5.2）：奇异输入也只返回 (str, bool)，绝不抛
        for bad in [None, 42, "str", [], {"result": object()}]:
            out = extract_summary(bad, state_was_parsed=False)
            assert isinstance(out, tuple) and len(out) == 2
            assert isinstance(out[0], str) and isinstance(out[1], bool)


class _FakeA2ATool:
    """duck-typed A2ATool：call_remote_agent 返回预置 ToolResult 或抛异常。"""

    def __init__(self, *, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    async def call_remote_agent(self, id: str, query: str):
        self.calls.append((id, query))
        if self._raises is not None:
            raise self._raises
        return self._result


def _remote_spec(**kw) -> WorkerSpec:
    base = dict(
        worker_runtime_type=WorkerRuntimeType.REMOTE,
        parent_session_id="p",
        objective="do it",
        remote_target="agent-1",
    )
    base.update(kw)
    return WorkerSpec(**base)


def _worker(result=None, *, raises=None) -> A2aSubagentWorker:
    return A2aSubagentWorker(_FakeA2ATool(result=result, raises=raises))


class TestRunGuard:
    @pytest.mark.anyio
    async def test_non_remote_spec_rejected(self) -> None:
        worker = _worker(ToolResult(success=True, data={"result": {}}))
        spec = WorkerSpec(
            worker_runtime_type=WorkerRuntimeType.LOCAL,
            parent_session_id="p",
            child_session_id="c",
            objective="x",
        )
        with pytest.raises(ValueError, match="REMOTE WorkerSpec"):
            await worker.run(spec)


class TestRunTransport:
    @pytest.mark.anyio
    async def test_transport_failure_clean_error_no_url(self) -> None:
        result = ToolResult(
            success=False,
            message="调用远程Agent[agent-1:https://host/x]出错: boom",
        )
        r = await _worker(result).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.FAILED
        assert r.error_summary == "a2a transport error"
        assert "https://" not in (r.error_summary or "")

    @pytest.mark.anyio
    async def test_timeout(self) -> None:
        r = await _worker(raises=TimeoutError()).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.TIMED_OUT

    @pytest.mark.anyio
    async def test_unexpected_exception_failed(self) -> None:
        r = await _worker(raises=RuntimeError("x")).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.FAILED
        assert r.error_summary == "a2a transport error"

    @pytest.mark.anyio
    async def test_data_none_unknown(self) -> None:
        r = await _worker(ToolResult(success=True, data=None)).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.UNKNOWN
        assert r.error_summary == "empty A2A payload"

    @pytest.mark.anyio
    async def test_data_non_dict_unknown(self) -> None:
        r = await _worker(ToolResult(success=True, data="oops")).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.UNKNOWN

    @pytest.mark.anyio
    async def test_data_empty_dict_unknown(self) -> None:
        r = await _worker(ToolResult(success=True, data={})).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.UNKNOWN

    @pytest.mark.anyio
    async def test_jsonrpc_error_in_200_failed_redacted(self) -> None:
        result = ToolResult(
            success=True, data={"error": {"message": "boom at https://h/x"}}
        )
        r = await _worker(result).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.FAILED
        assert "https://" not in (r.error_summary or "")
        assert "boom" in (r.error_summary or "")


class TestRunStateClassification:
    async def _run_state(self, state: str, **extra):
        status = {"state": state}
        status.update(extra)
        data = {"result": {"status": status}}
        return await _worker(ToolResult(success=True, data=data)).run(_remote_spec())

    @pytest.mark.anyio
    async def test_completed_success(self) -> None:
        r = await self._run_state("completed", message={"parts": [{"text": "done"}]})
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert r.summary == "done"

    @pytest.mark.anyio
    async def test_failed(self) -> None:
        r = await self._run_state("failed")
        assert r.terminal_outcome == WorkerTerminalOutcome.FAILED

    @pytest.mark.anyio
    async def test_canceled(self) -> None:
        r = await self._run_state("canceled")
        assert r.terminal_outcome == WorkerTerminalOutcome.CANCELLED

    @pytest.mark.anyio
    async def test_input_required_waiting(self) -> None:
        r = await self._run_state("input-required")
        assert r.lifecycle_state == WorkerLifecycleState.WAITING_INPUT
        assert r.terminal_outcome is None

    @pytest.mark.anyio
    async def test_working_unknown(self) -> None:
        r = await self._run_state("working")
        assert r.terminal_outcome == WorkerTerminalOutcome.UNKNOWN

    @pytest.mark.anyio
    async def test_unknown_state_unknown(self) -> None:
        r = await self._run_state("banana")
        assert r.terminal_outcome == WorkerTerminalOutcome.UNKNOWN


class TestRunNonConformant:
    @pytest.mark.anyio
    async def test_reply_shape_success(self) -> None:
        # {"reply":"task done"} → SUCCESS + summary，无 error（引 test_r2_wrapper_integration.py:142）
        r = await _worker(ToolResult(success=True, data={"reply": "task done"})).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert r.summary == "task done"
        assert r.error_summary is None

    @pytest.mark.anyio
    async def test_result_plain_string_success(self) -> None:
        r = await _worker(ToolResult(success=True, data={"result": "plain"})).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert r.summary == "plain"

    @pytest.mark.anyio
    async def test_unparseable_dict_success_with_preview(self) -> None:
        # 非空 dict、无 state、无可抽文本 → 默认 SUCCESS + preview + unparsed error
        r = await _worker(ToolResult(success=True, data={"weird": {"x": 1}})).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert r.error_summary == "unparsed A2A payload"
        assert r.summary  # 非空 preview


class TestRunRobustness:
    @pytest.mark.anyio
    async def test_malformed_shapes_do_not_crash(self) -> None:
        malformed = [
            {"result": {"status": "not-a-dict"}},
            {"result": {"status": {"message": {"parts": "not-a-list"}}}},
            {"result": {"artifacts": "not-a-list"}},
            {"result": {"parts": [{"text": 123}]}},          # text 非 str
            {"result": 42},                                   # payload 标量
            {"result": {"status": {"state": ["nested"]}}},    # state 非 str
        ]
        for data in malformed:
            r = await _worker(ToolResult(success=True, data=data)).run(_remote_spec())
            assert isinstance(r, SubagentRunResult)
            assert r.lifecycle_state in {
                WorkerLifecycleState.TERMINAL,
                WorkerLifecycleState.WAITING_INPUT,
            }

    @pytest.mark.anyio
    async def test_giant_text_capped(self) -> None:
        data = {"result": {"status": {"message": {"parts": [{"text": "x" * 1_000_000}]}}}}
        r = await _worker(ToolResult(success=True, data=data)).run(_remote_spec())
        assert len(r.summary) <= 16_384


class TestRunIdentityFields:
    @pytest.mark.anyio
    async def test_identity_cost_duration(self) -> None:
        spec = _remote_spec(child_session_id="cs")
        data = {"result": {"status": {"state": "completed", "message": {"parts": [{"text": "ok"}]}}}}
        r = await _worker(ToolResult(success=True, data=data)).run(spec)
        assert r.worker_runtime_type == WorkerRuntimeType.REMOTE
        assert r.parent_session_id == "p"
        assert r.child_session_id == "cs"
        assert r.source_ref == "a2a:agent-1"
        assert r.cost_summary is None
        assert r.cost_authoritative is False
        assert r.duration_source == "observed_local"
        assert r.duration_seconds is not None and r.duration_seconds >= 0

    @pytest.mark.anyio
    async def test_never_pending_or_running(self) -> None:
        for data in [
            {"result": {"status": {"state": "completed"}}},
            {"result": {"status": {"state": "input-required"}}},
            {"reply": "x"},
            {},
        ]:
            r = await _worker(ToolResult(success=True, data=data)).run(_remote_spec())
            assert r.lifecycle_state not in {
                WorkerLifecycleState.PENDING,
                WorkerLifecycleState.RUNNING,
            }


class TestRunSpecCoverage:
    """spec §7 PR-3 补全 case（R2#P2-a）：adapter 级 Message-like / 深嵌套 fallback /
    giant error / giant fallback / REMOTE 缺 remote_target guard。"""

    @pytest.mark.anyio
    async def test_message_like_parts_success(self) -> None:
        # adapter 级 Message-like（result.parts 直挂，无 status）→ SUCCESS via (c)
        data = {"result": {"parts": [{"text": "m"}]}}
        r = await _worker(ToolResult(success=True, data=data)).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert r.summary == "m"

    @pytest.mark.anyio
    async def test_deep_nesting_fallback_no_crash(self) -> None:
        node: dict = {}
        cur = node
        for _ in range(60):
            child: dict = {}
            cur["child"] = child
            cur = child
        r = await _worker(ToolResult(success=True, data={"result": node})).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert isinstance(r.summary, str)

    @pytest.mark.anyio
    async def test_giant_error_message_capped(self) -> None:
        data = {"error": {"message": "x" * 1_000_000}}
        r = await _worker(ToolResult(success=True, data=data)).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.FAILED
        assert len(r.error_summary or "") <= 2_048

    @pytest.mark.anyio
    async def test_giant_fallback_preview_capped(self) -> None:
        data = {"junk": "y" * 1_000_000}
        r = await _worker(ToolResult(success=True, data=data)).run(_remote_spec())
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert len(r.summary) <= 4_096

    @pytest.mark.anyio
    async def test_remote_spec_missing_target_rejected(self) -> None:
        # model_construct 绕过 validator，构造 REMOTE + remote_target=None，验 run() guard 第二支
        spec = WorkerSpec.model_construct(
            worker_runtime_type=WorkerRuntimeType.REMOTE,
            parent_session_id="p",
            child_session_id=None,
            objective="x",
            permission_scope_descriptor=None,
            remote_target=None,
            expected_result_schema=None,
        )
        with pytest.raises(ValueError, match="REMOTE WorkerSpec"):
            await _worker(ToolResult(success=True, data={})).run(spec)
