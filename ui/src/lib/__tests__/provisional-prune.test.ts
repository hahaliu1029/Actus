import { describe, expect, it } from "vitest";

import {
  applyProvisionalSignal,
  createProvisionalState,
  shouldDropReplayedCalling,
} from "@/lib/event-normalize";
import type { SessionEventRecord } from "@/lib/event-normalize";

function toolEvt(id: string, status: string, seq: number): SessionEventRecord {
  return { event: "tool", data: { tool_call_id: id, status, seq, name: "file", function: "file_write", args: {} } };
}
function msgEvt(seq: number): SessionEventRecord {
  return { event: "message", data: { role: "assistant", message: "final", seq } };
}
function doneEvt(seq: number): SessionEventRecord {
  return { event: "done", data: { seq } };
}

function run(events: SessionEventRecord[]) {
  let state = createProvisionalState();
  let list: SessionEventRecord[] = [];
  for (const e of events) {
    const out = applyProvisionalSignal(state, list, e);
    state = out.state;
    list = out.events;
  }
  return { state, list };
}
const callingIds = (list: SessionEventRecord[]) =>
  list.filter((e) => e.event === "tool" && e.data.status === "calling")
      .map((e) => e.data.tool_call_id);

describe("B1-2 provisional prune (spec §5.3)", () => {
  it("scenario i: sibling survives when batchmate completes (upgrade never prunes)", () => {
    const { list } = run([
      toolEvt("a", "calling", 10),
      toolEvt("b", "calling", 11),
      toolEvt("a", "called", 12),   // a 完成 —— b 仍 provisional，绝不误杀
    ]);
    expect(callingIds(list)).toEqual(["b"]);
  });

  it("scenario ii: id-swap retry after a closed turn prunes stale orphans", () => {
    const { list } = run([
      toolEvt("a", "calling", 10),
      toolEvt("a", "called", 11),    // turn 边界（close 信号）
      toolEvt("b", "calling", 20),   // 上一轮孤儿？无 —— b 注册
      toolEvt("c", "calling", 30),   // 同轮 sibling（无边界介入）→ 不清除 b
    ]);
    expect(callingIds(list)).toEqual(["b", "c"]);

    const withOrphan = run([
      toolEvt("a", "calling", 10),
      toolEvt("a", "called", 11),    // close
      toolEvt("b", "calling", 20),   // b 成孤儿（其后无升级）
      msgEvt(25),                    // close（假设纯文本插入）→ b 被清除
      toolEvt("b2", "calling", 30),  // 新轮
    ]);
    expect(callingIds(withOrphan.list)).toEqual(["b2"]);
  });

  it("scenario iii: pure-text retry — MessageEvent prunes all provisional", () => {
    const { list } = run([
      toolEvt("a", "calling", 10),
      toolEvt("b", "calling", 11),
      msgEvt(20),
    ]);
    expect(callingIds(list)).toEqual([]);
  });

  it("scenario iv: blank retry — DoneEvent prune-all backstop (R17#1)", () => {
    const { list } = run([
      toolEvt("a", "calling", 10),   // 失败 attempt 的孤儿（其间无任何 close 信号）
      toolEvt("a2", "calling", 20),  // 重试同样失败 → 空白完成
      doneEvt(30),
    ]);
    expect(callingIds(list)).toEqual([]);
  });

  it("done prunes ONLY provisional — completed history untouched", () => {
    const { list } = run([
      toolEvt("a", "calling", 10),
      toolEvt("a", "called", 11),
      toolEvt("b", "calling", 20),
      doneEvt(30),
    ]);
    expect(list.some((e) => e.data.tool_call_id === "a")).toBe(true);
    expect(callingIds(list)).toEqual([]);
  });

  it("scenario v: replayed calling below watermark cannot resurrect (R11#3)", () => {
    const { state } = run([
      toolEvt("a", "calling", 10),
      msgEvt(20),                    // 清除 a，水位线=20
    ]);
    expect(
      shouldDropReplayedCalling(state, toolEvt("a", "calling", 10), /*hasCard=*/ false)
    ).toBe(true);
    expect(
      shouldDropReplayedCalling(state, toolEvt("z", "calling", 25), false)
    ).toBe(false);
  });

  it("scenario v-b: replayed old calling never downgrades an upgraded card (R1#5)", () => {
    const { list } = run([
      toolEvt("a", "calling", 10),
      toolEvt("a", "called", 11),
      toolEvt("a", "calling", 10),   // recovery merge 重放旧事件
    ]);
    const card = list.find((e) => e.data.tool_call_id === "a");
    expect(card?.data.status).toBe("called");
  });

  it("scenario v-c: upgrade boundary advances watermark — stale different-id calling dropped (R10#1)", () => {
    const { list, state } = run([
      toolEvt("a", "calling", 10),
      toolEvt("a", "called", 11),      // 升级边界：watermark → 11
      toolEvt("z", "calling", 9),      // 重放的陈旧异 id CALLING（seq < 11，无同 id 卡）
    ]);
    expect(state.watermark).toBe(11);
    expect(callingIds(list)).toEqual([]);  // z 被防重放丢弃，不得被当作新轮首卡
  });

  it("scenario vii: fold refold of an OPEN turn's own calling preserves the card (R13#1)", () => {
    // live：a calling(10) → state {w:10, tc:false}。refetch fold 从空列表重建，
    // merged 流含同一 a calling(10)——它是开放轮次自己的卡，绝不能被防重放误杀。
    const live = run([toolEvt("a", "calling", 10)]);
    expect(live.state.watermark).toBe(10);
    expect(live.state.turnClosed).toBe(false);
    // 模拟 applyProvisionalReplay：从 live.state + 空列表 fold merged 流
    const state = live.state;
    const list: SessionEventRecord[] = [];
    const out = applyProvisionalSignal(state, list, toolEvt("a", "calling", 10));
    expect(callingIds(out.events)).toEqual(["a"]);
    // 对照：已关闭边界上的无卡 CALLING 仍然丢弃
    expect(
      shouldDropReplayedCalling(
        { watermark: 10, turnClosed: true }, toolEvt("x", "calling", 10), false
      )
    ).toBe(true);
  });

  it("scenario vi: error is a close signal — marks boundary and advances watermark without pruning (R10#1)", () => {
    const errEvt = { event: "error", data: { message: "boom", seq: 15 } } as const;
    const { list, state } = run([
      toolEvt("a", "calling", 10),
      errEvt as unknown as SessionEventRecord,
    ]);
    expect(state.turnClosed).toBe(true);
    expect(state.watermark).toBe(15);
    expect(callingIds(list)).toEqual(["a"]);  // error 不清除 provisional（Done 兜底负责）
    // 但边界之后的新轮 CALLING 会触发清除：
    const after = run([
      toolEvt("a", "calling", 10),
      errEvt as unknown as SessionEventRecord,
      toolEvt("a2", "calling", 20),    // 重试新轮
    ]);
    expect(callingIds(after.list)).toEqual(["a2"]);
  });
});
