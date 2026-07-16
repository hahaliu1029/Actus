import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// SPM Task 30 (contract C): drive VNCViewer's RFB event listeners and assert the
// `onStatus` callback now receives status *enum* values instead of Chinese strings.
// MockRFB lives in vi.hoisted so the (hoisted) vi.mock factory can reference it.
const { MockRFB } = vi.hoisted(() => {
  type RfbEvent = { detail?: { clean?: boolean } };
  class MockRFB {
    static instances: MockRFB[] = [];
    listeners: Record<string, Array<(event?: RfbEvent) => void>> = {};
    viewOnly = false;
    scaleViewport = false;
    resizeSession = false;
    background = "";

    constructor() {
      MockRFB.instances.push(this);
    }

    addEventListener(type: string, cb: (event?: RfbEvent) => void) {
      (this.listeners[type] ??= []).push(cb);
    }

    disconnect() {}

    emit(type: string, event?: RfbEvent) {
      (this.listeners[type] ?? []).forEach((cb) => cb(event));
    }
  }
  return { MockRFB };
});

vi.mock("@novnc/novnc/lib/rfb", () => ({ default: MockRFB }));

import { VNCViewer } from "./vnc-viewer";

const ENUM = new Set(["connecting", "connected", "disconnected", "error"]);

describe("VNCViewer onStatus enum (SPM Task 30)", () => {
  beforeEach(() => {
    MockRFB.instances = [];
  });

  it("connect 事件回调传枚举 connected", () => {
    const onStatus = vi.fn();
    render(<VNCViewer url="wss://x" onStatus={onStatus} />);
    MockRFB.instances[0].emit("connect");
    expect(onStatus).toHaveBeenCalledWith("connected");
  });

  it("disconnect(clean) 回调传枚举 disconnected", () => {
    const onStatus = vi.fn();
    render(<VNCViewer url="wss://x" onStatus={onStatus} />);
    MockRFB.instances[0].emit("disconnect", { detail: { clean: true } });
    expect(onStatus).toHaveBeenCalledWith("disconnected");
  });

  it("disconnect(非 clean) 回调传枚举 error", () => {
    const onStatus = vi.fn();
    render(<VNCViewer url="wss://x" onStatus={onStatus} />);
    MockRFB.instances[0].emit("disconnect", { detail: { clean: false } });
    expect(onStatus).toHaveBeenCalledWith("error");
  });

  it("onStatus 收到的实参恒 ∈ 枚举集（无中文串）", () => {
    const onStatus = vi.fn();
    render(<VNCViewer url="wss://x" onStatus={onStatus} />);
    const rfb = MockRFB.instances[0];
    rfb.emit("connect");
    rfb.emit("disconnect", { detail: { clean: true } });
    rfb.emit("disconnect", { detail: { clean: false } });
    expect(onStatus).toHaveBeenCalled();
    for (const call of onStatus.mock.calls) {
      expect(ENUM.has(call[0] as string)).toBe(true);
    }
  });
});
