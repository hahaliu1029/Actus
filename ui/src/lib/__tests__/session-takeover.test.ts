import { describe, expect, it, vi } from "vitest";
import { startTakeoverWithReopen } from "../session-takeover";
import type { StartTakeoverResponse } from "@/lib/api/types";

const RESULT: StartTakeoverResponse = {
  status: "takeover_pending",
  request_status: "starting",
  scope: "shell",
};

describe("startTakeoverWithReopen", () => {
  it("completed → reopen THEN start (in order)", async () => {
    const calls: string[] = [];
    const reopenTakeover = vi.fn(async () => {
      calls.push("reopen");
    });
    const startTakeover = vi.fn(async () => {
      calls.push("start");
      return RESULT;
    });
    await startTakeoverWithReopen("s1", "shell", "completed", {
      reopenTakeover,
      startTakeover,
    });
    expect(calls).toEqual(["reopen", "start"]);
    expect(startTakeover).toHaveBeenCalledWith("s1", { scope: "shell" });
  });

  it("non-completed → start only, no reopen", async () => {
    const reopenTakeover = vi.fn(async () => {});
    const startTakeover = vi.fn(async () => RESULT);
    await startTakeoverWithReopen("s1", "browser", "waiting", {
      reopenTakeover,
      startTakeover,
    });
    expect(reopenTakeover).not.toHaveBeenCalled();
    expect(startTakeover).toHaveBeenCalledWith("s1", { scope: "browser" });
  });
});
