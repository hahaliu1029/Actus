"""B1-2 ToolCallStreamCollector — spec §5.2 failure-mode table driven."""
from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from app.domain.services.executor import CompletedToolCall, ToolCallStreamCollector


def _chunk(tccs: list[dict] | None = None, content: str = "", usage=None) -> AIMessageChunk:
    kwargs: dict = {"content": content}
    if tccs is not None:
        kwargs["tool_call_chunks"] = [
            {"type": "tool_call_chunk", **t} for t in tccs
        ]
    if usage is not None:
        kwargs["usage_metadata"] = usage
    return AIMessageChunk(**kwargs)


def _tcc(index: int, *, id: str | None = None, name: str | None = None,
         args: str | None = None) -> dict:
    return {"index": index, "id": id, "name": name, "args": args}


class TestIncrementalCompletion:
    def test_single_call_completes_at_stream_end(self):
        c = ToolCallStreamCollector()
        assert c.ingest(_chunk([_tcc(0, id="a", name="file_write", args='{"pa')])) == []
        assert c.ingest(_chunk([_tcc(0, args='th": "/x"}')])) == []
        final, tail = c.finalize()
        assert [t.tool_call_id for t in tail] == ["a"]
        assert tail[0].args == {"path": "/x"}
        assert final.tool_calls and final.tool_calls[0]["id"] == "a"

    def test_index_switch_completes_prior_call(self):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="f", args='{"x": 1}')]))
        done = c.ingest(_chunk([_tcc(1, id="b", name="g", args='{}')]))
        assert [t.tool_call_id for t in done] == ["a"]
        _, tail = c.finalize()
        assert [t.tool_call_id for t in tail] == ["b"]

    def test_same_index_new_id_completes_prior_call(self):
        """Ollama 复用 index=0，仅靠 id 区分（spec §0.5 结论 → §5.2 判定）。"""
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="f", args='{"x": 1}')]))
        done = c.ingest(_chunk([_tcc(0, id="b", name="g", args='{"y": 2}')]))
        assert [t.tool_call_id for t in done] == ["a"]
        _, tail = c.finalize()
        assert [t.tool_call_id for t in tail] == ["b"]


class TestFailureModeTable:
    def test_empty_or_none_args_is_empty_dict(self):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="noarg", args="")]))
        _, tail = c.finalize()
        assert tail and tail[0].args == {}

        # §5.2 第一行的 None 分支（R2#4 修：与 "" 分开断言）
        c2 = ToolCallStreamCollector()
        c2.ingest(_chunk([_tcc(0, id="b", name="noarg", args=None)]))
        _, tail2 = c2.finalize()
        assert tail2 and tail2[0].args == {}

    def test_malformed_tail_args_no_incremental_but_authoritative_keeps_langchain_semantics(self):
        """R14#4 分歧 fixture：parse_partial_json 可救回（'{"a": 1'→{"a": 1}）但
        strict json.loads 拒绝 → 零增量发射；权威合并消息仍按 LangChain 自身
        语义处置（本测试杀死「collector 偷读 .tool_calls」的规避实现）。"""
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="f", args='{"a": 1')]))  # 无闭括号
        final, tail = c.finalize()
        assert tail == [], "strict json.loads 失败 → 不发增量 CALLING"
        # 权威产物：与 chunk-sum 基线逐字段一致（LangChain parse_partial_json
        # 可能把它救进 .tool_calls——那是执行侧既有语义，collector 不干预）
        baseline = _chunk([_tcc(0, id="a", name="f", args='{"a": 1')])
        assert final.tool_calls == baseline.tool_calls
        assert final.invalid_tool_calls == baseline.invalid_tool_calls

    def test_completed_without_id_or_name_not_emitted(self):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id=None, name="f", args="{}")]))
        _, tail = c.finalize()
        assert tail == []

        c2 = ToolCallStreamCollector()
        c2.ingest(_chunk([_tcc(0, id="a", name=None, args="{}")]))
        _, tail2 = c2.finalize()
        assert tail2 == []

    def test_duplicate_final_id_second_not_emitted(self):
        """R2#4 修：显式捕获 ingest 返回——精确证明「后者不发」而非计数上界。"""
        c = ToolCallStreamCollector()
        first = c.ingest(_chunk([_tcc(0, id="dup", name="f", args="{}")]))
        assert first == []                       # 尚在累积
        switched = c.ingest(_chunk([_tcc(1, id="dup", name="f", args="{}")]))
        assert [t.tool_call_id for t in switched] == ["dup"], \
            "第一个 dup 在 index 切换点完成发射"
        _, tail = c.finalize()
        assert tail == [], "重复 final id：第二个绝不发增量 CALLING（§5.2）"

    def test_usage_only_chunk_joins_sum_skips_dispatch(self):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="f", args="{}")]))
        done = c.ingest(_chunk(
            None, usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        ))
        assert done == []
        final, tail = c.finalize()
        assert final.usage_metadata is not None
        assert [t.tool_call_id for t in tail] == ["a"]

    def test_late_same_index_after_close_degrades_advisory(self):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="f", args="{}")]))
        done = c.ingest(_chunk([_tcc(1, id="b", name="g", args="{}")]))
        assert [t.tool_call_id for t in done] == ["a"]
        # index 0 在关闭后迟到 → 降级：此后零增量发射（已发的不撤回）
        c.ingest(_chunk([_tcc(0, args='{"late": true}')]))
        _, tail = c.finalize()
        assert tail == [], "降级后不再发增量（b 的尾部完成被抑制）"
        assert c.degraded is True


class TestFinalizeNormalization:
    def test_out_of_order_index_normalized_ascending(self):
        """乱序 index 1→0：merge_lists first-seen append 不排序——finalize 必须
        按数值 index 稳定升序，使执行输入与 ainvoke 基线数组序对齐（spec R3#3）。
        （R4#1 修：b 在第二次 ingest 的 index 切换点完成发射，tail 只含 a。）"""
        c = ToolCallStreamCollector()
        first = c.ingest(_chunk([_tcc(1, id="b", name="g", args="{}")]))
        assert first == []
        switched = c.ingest(_chunk([_tcc(0, id="a", name="f", args="{}")]))
        assert [t.tool_call_id for t in switched] == ["b"], "index 切换关闭 b"
        final, tail = c.finalize()
        assert [t.tool_call_id for t in tail] == ["a"], "a 在流结束点完成"
        assert [t["id"] for t in final.tool_calls] == ["a", "b"]
        assert [tc.get("index") for tc in final.tool_call_chunks] == [0, 1]

    def test_same_index_multi_call_stable_first_seen_order(self):
        """同 index 多逻辑调用（Ollama）：稳定排序保持 first-seen 到达序（R7#8）。

        Step 4 自审注收紧：LangChain `merge_lists` 对同 index 双 id chunk-sum 的
        实测行为 = 保留两个逻辑调用（不合并、不抛错），`.tool_calls` ids == ['a','b']、
        `invalid_tool_calls` == []，`tool_call_chunks` 保持 first-seen 到达序
        (a→b, 皆 index 0)。advisory 层硬断言：a 于同 index 新 id 检测点完成、
        b 于尾部完成 → tail == ['b']。"""
        c = ToolCallStreamCollector()
        first = c.ingest(_chunk([_tcc(0, id="a", name="f", args="{}")]))
        assert first == []
        # 同 index 新非空 id → a 于此点完成发射
        done = c.ingest(_chunk([_tcc(0, id="b", name="g", args="{}")]))
        assert [t.tool_call_id for t in done] == ["a"]
        final, tail = c.finalize()
        # 权威产物（实测收紧）：两个逻辑调用皆在，first-seen 到达序 a<b
        assert [t["id"] for t in final.tool_calls] == ["a", "b"]
        assert final.invalid_tool_calls == []
        assert [c.get("id") for c in final.tool_call_chunks] == ["a", "b"]
        assert [c.get("index") for c in final.tool_call_chunks] == [0, 0]
        # advisory 硬断言：b 于流结束点完成
        assert [t.tool_call_id for t in tail] == ["b"]
