"""B1-2 tool-call streaming collector — dual products (spec §5.2).

1. AUTHORITATIVE merged message: pure chunk-sum (``acc = acc + chunk``,
   LangChain semantics). Execution input is ALWAYS taken from this product —
   byte-equivalent to the ainvoke baseline, field by field.
2. ADVISORY incremental completion stream: built from RAW ``tool_call_chunks``
   (never mid-stream ``.tool_calls`` — parse_partial_json is best-effort).
   Completion = index switch ∨ new non-empty id at same index ∨ stream end;
   strict ``json.loads`` on close. Any anomaly degrades THIS layer only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessageChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletedToolCall:
    tool_call_id: str
    name: str
    args: dict
    index: int


@dataclass
class _PartialCall:
    index: int
    id: str | None = None
    name: str | None = None
    args_parts: list = field(default_factory=list)


class ToolCallStreamCollector:
    def __init__(self) -> None:
        self._acc: AIMessageChunk | None = None
        self._current: _PartialCall | None = None
        self._closed_indexes: set[int] = set()
        self._emitted_ids: set[str] = set()
        self.degraded: bool = False

    # ---- product 1: authoritative chunk-sum ----

    def _sum(self, chunk: AIMessageChunk) -> None:
        self._acc = chunk if self._acc is None else self._acc + chunk

    # ---- product 2: advisory incremental completions ----

    def ingest(self, chunk: AIMessageChunk) -> list[CompletedToolCall]:
        self._sum(chunk)
        if self.degraded:
            return []
        newly: list[CompletedToolCall] = []
        for tcc in (chunk.tool_call_chunks or []):
            idx = tcc.get("index")
            if idx is None:
                # index-less provider chunk — advisory 无法定位槽位 → 降级
                self._degrade("tool_call_chunk without index")
                return newly
            if self._current is not None and idx == self._current.index:
                new_id = tcc.get("id")
                if new_id and self._current.id and new_id != self._current.id:
                    # 同 index 新非空 id（Ollama 复用模式）→ 前者完成
                    self._close_current(newly)
                    self._open(idx, tcc)
                    continue
                self._absorb(tcc)
                continue
            if idx in self._closed_indexes:
                # finalize 后同 index 迟到 chunk → 记 error、整批降级（§5.2）
                self._degrade(f"late chunk for closed index {idx}")
                return newly
            if self._current is not None:
                self._close_current(newly)   # index 切换 → 前者完成
            self._open(idx, tcc)
        return newly

    def finalize(self) -> tuple[AIMessageChunk, list[CompletedToolCall]]:
        tail: list[CompletedToolCall] = []
        if not self.degraded and self._current is not None:
            self._close_current(tail)        # 流结束 → 收尾完成
        final = self._normalized_final()
        return final, tail

    # ---- internals ----

    def _open(self, idx: int, tcc: dict) -> None:
        self._current = _PartialCall(index=idx)
        self._absorb(tcc)

    def _absorb(self, tcc: dict) -> None:
        cur = self._current
        assert cur is not None
        if tcc.get("id"):
            cur.id = tcc["id"]
        if tcc.get("name"):
            cur.name = tcc["name"]
        if tcc.get("args"):
            cur.args_parts.append(tcc["args"])

    def _close_current(self, sink: list[CompletedToolCall]) -> None:
        cur = self._current
        self._current = None
        if cur is None:
            return
        self._closed_indexes.add(cur.index)
        if not cur.id or not cur.name:
            return                            # invalid — 不发射（§5.2）
        if cur.id in self._emitted_ids:
            return                            # 重复 final id — 后者不发（§5.2）
        raw = "".join(cur.args_parts)
        if raw.strip() == "":
            args: dict = {}                   # 无参工具（§5.2）
        else:
            try:
                args = json.loads(raw)
            except (ValueError, TypeError):
                return                        # strict 失败 → 零增量；执行输入不受影响
            if not isinstance(args, dict):
                return
        self._emitted_ids.add(cur.id)
        sink.append(CompletedToolCall(
            tool_call_id=cur.id, name=cur.name, args=args, index=cur.index,
        ))

    def _degrade(self, reason: str) -> None:
        if not self.degraded:
            logger.warning("ToolCallStreamCollector degraded (advisory off): %s", reason)
        self.degraded = True
        self._current = None

    def _normalized_final(self) -> AIMessageChunk:
        """finalize 规范化（R3#3/R7#8）：tool_calls / tool_call_chunks 按数值
        index 稳定升序（同 index 保持 first-seen 到达序）。merge_lists 是
        first-seen append 不排序——乱序 index 到达时不规范化会与 ainvoke
        基线数组序不一致。"""
        final = self._acc if self._acc is not None else AIMessageChunk(content="")
        chunks = list(final.tool_call_chunks or [])
        if not chunks:
            return final
        order = sorted(
            range(len(chunks)),
            key=lambda i: (
                chunks[i].get("index") if chunks[i].get("index") is not None else i,
                i,
            ),
        )
        sorted_chunks = [chunks[i] for i in order]
        # tool_calls 与 chunks 经 id 对齐重排；缺 id 的保持相对原序殿后
        by_id = {tc.get("id"): tc for tc in (final.tool_calls or []) if tc.get("id")}
        sorted_calls = [
            by_id[c["id"]] for c in sorted_chunks
            if c.get("id") and c["id"] in by_id
        ]
        leftovers = [
            tc for tc in (final.tool_calls or [])
            if not tc.get("id") or tc.get("id") not in {c.get("id") for c in sorted_chunks}
        ]
        return final.model_copy(update={
            "tool_calls": sorted_calls + leftovers,
            "tool_call_chunks": sorted_chunks,
        })
