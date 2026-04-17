import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { parseToolEventEnvelope } from "../session-ui";

describe("envelope_version tolerant reader", () => {
  const baseWire = {
    tool_call_id: "c1",
    name: "shell",
    function: "shell_execute",
    args: { command: "ls" },
    status: "called" as const,
    activity_description: "",
  };

  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it("envelope_version=1 parses cleanly without warning", () => {
    const env = parseToolEventEnvelope({ envelope_version: 1, ...baseWire });
    expect(env).not.toBeNull();
    expect(env?.envelope_version).toBe(1);
    expect(env?.name).toBe("shell");
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("envelope_version=2 (future) normalizes to v1 subset with warning", () => {
    const env = parseToolEventEnvelope({
      envelope_version: 2,
      unknown_future_field: "x",
      ...baseWire,
    });
    expect(env).not.toBeNull();
    expect(env?.envelope_version).toBe(1); // normalized back to v1
    expect(env?.name).toBe("shell"); // existing subset preserved
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it("envelope_version=undefined falls back to v1 without warning", () => {
    const env = parseToolEventEnvelope({ ...baseWire });
    expect(env).not.toBeNull();
    expect(env?.envelope_version).toBe(1);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("envelope_version=null falls back to v1 without warning", () => {
    const env = parseToolEventEnvelope({ envelope_version: null, ...baseWire });
    expect(env).not.toBeNull();
    expect(env?.envelope_version).toBe(1);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("non-object input returns null", () => {
    expect(parseToolEventEnvelope(null)).toBeNull();
    expect(parseToolEventEnvelope(undefined)).toBeNull();
    expect(parseToolEventEnvelope("not an object")).toBeNull();
    expect(parseToolEventEnvelope(42)).toBeNull();
  });

  it("array input returns null (Gate 3.5 P2: arrays are typeof object but not valid envelope)", () => {
    expect(parseToolEventEnvelope([])).toBeNull();
    expect(parseToolEventEnvelope([baseWire])).toBeNull();
    expect(parseToolEventEnvelope([1, 2, 3])).toBeNull();
  });
});
