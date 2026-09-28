"""B1-2 ToolCallStreamCollector — spec §5.2 failure-mode table driven."""
from __future__ import annotations

import pytest

from app.application.errors.exceptions import ServerRequestsError
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
    @pytest.mark.parametrize("raw", ["", None, '{"a": 1', '[]', 'null', '{"x":NaN}', '{"x":Infinity}'])
    def test_invalid_raw_arguments_cannot_execute(self, raw):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="a", name="f", args=raw)]))
        with pytest.raises(ServerRequestsError, match="arguments"):
            c.finalize()

    @pytest.mark.parametrize("call_id,name", [(None, "f"), ("a", None)])
    def test_missing_identity_cannot_execute(self, call_id, name):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id=call_id, name=name, args="{}")]))
        with pytest.raises(ServerRequestsError, match="identity"):
            c.finalize()

    def test_duplicate_final_id_rejects_batch(self):
        c = ToolCallStreamCollector()
        c.ingest(_chunk([_tcc(0, id="dup", name="f", args="{}")]))
        switched = c.ingest(_chunk([_tcc(1, id="dup", name="f", args="{}")]))
        assert [t.tool_call_id for t in switched] == ["dup"]
        with pytest.raises(ServerRequestsError, match="identity"):
            c.finalize()

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
        with pytest.raises(ServerRequestsError, match="arguments"):
            c.finalize()
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
