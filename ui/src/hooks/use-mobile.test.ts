import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useIsMobile } from "./use-mobile";

describe("useIsMobile", () => {
  const listeners = new Set<() => void>();

  beforeEach(() => {
    listeners.clear();
    vi.stubGlobal("innerWidth", 820);
    vi.stubGlobal("matchMedia", vi.fn(() => ({
      addEventListener: (_event: string, listener: () => void) => listeners.add(listener),
      removeEventListener: (_event: string, listener: () => void) => listeners.delete(listener),
    })));
  });

  afterEach(() => vi.unstubAllGlobals());

  it("820px 仍使用桌面侧栏，同时允许工作区采用较宽的抽屉断点", () => {
    const sidebar = renderHook(() => useIsMobile());
    const workbench = renderHook(() => useIsMobile(1200));
    expect(sidebar.result.current).toBe(false);
    expect(workbench.result.current).toBe(true);
  });

  it("跨过自定义断点后更新结果，卸载会清理监听器", () => {
    const { result, unmount } = renderHook(() => useIsMobile(1200));
    expect(result.current).toBe(true);
    act(() => {
      vi.stubGlobal("innerWidth", 1200);
      for (const listener of listeners) listener();
    });
    expect(result.current).toBe(false);
    unmount();
    expect(listeners.size).toBe(0);
  });
});
