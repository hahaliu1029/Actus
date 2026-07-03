import { describe, expect, it } from "vitest";

import { getToolDisplayCopy } from "@/lib/session-ui";

function toolEvent(status: string) {
  return {
    name: "file",
    function: "file_write",
    args: { path: "/x" },
    status,
    tool_call_id: "t1",
  } as Record<string, unknown>;
}

describe("B1-1b running status rendering (spec §4.2)", () => {
  it("calling copy is unchanged (flags-off regression, INV-B1-0)", () => {
    const copy = getToolDisplayCopy(toolEvent("calling"));
    expect(JSON.stringify(copy)).toContain("正在");
    expect(JSON.stringify(copy)).not.toContain("已完成");
  });

  it("running renders as in-progress (tolerant non-called path)", () => {
    const copy = getToolDisplayCopy(toolEvent("running"));
    expect(JSON.stringify(copy)).toContain("正在");
    expect(JSON.stringify(copy)).not.toContain("已完成");
  });

  it("called copy is unchanged", () => {
    const copy = getToolDisplayCopy(toolEvent("called"));
    expect(JSON.stringify(copy)).toContain("已完成");
  });

  it("running and calling produce identical copy shape (card upsert-safe)", () => {
    // 按 tool_call_id 幂等升级：calling → running 覆盖同一张卡，
    // copy 形状一致意味着升级不会闪变卡片结构。
    const calling = getToolDisplayCopy(toolEvent("calling"));
    const running = getToolDisplayCopy(toolEvent("running"));
    expect(Object.keys(running)).toEqual(Object.keys(calling));
  });
});
