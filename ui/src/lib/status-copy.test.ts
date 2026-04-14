import { describe, expect, it } from "vitest";

import { getSessionStatusMeta, getStepStatusMeta } from "./status-copy";

describe("status-copy", () => {
  it("maps known session statuses", () => {
    expect(getSessionStatusMeta("running").text).toBe("执行中");
    expect(getSessionStatusMeta("completed").text).toBe("已完成");
    expect(getSessionStatusMeta("waiting").text).toBe("等待中");
    expect(getSessionStatusMeta("takeover_pending").text).toBe("待接管");
    expect(getSessionStatusMeta("takeover").text).toBe("接管中");
  });

  it("returns finishing meta", () => {
    const meta = getSessionStatusMeta("finishing");
    expect(meta.text).toBe("整理中");
    expect(meta.tone).toBe("info");
    expect(meta.spinning).toBeUndefined();
  });

  it("maps known step statuses", () => {
    expect(getStepStatusMeta("started").text).toBe("执行中");
    expect(getStepStatusMeta("failed").text).toBe("失败");
  });

  it("falls back for unknown statuses", () => {
    expect(getSessionStatusMeta("whatever").text).toBe("未知状态");
    expect(getSessionStatusMeta(undefined).text).toBe("待执行");
    expect(getStepStatusMeta("xx").text).toBe("未知状态");
    expect(getStepStatusMeta(null).text).toBe("待执行");
  });
});
