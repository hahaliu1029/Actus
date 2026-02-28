import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockGetTakeover,
  terminalInstances,
  fitAddonInstances,
  resizeObserverInstances,
  MockTerminal,
  MockFitAddon,
  MockResizeObserver,
} = vi.hoisted(() => {
  type TerminalDisposable = { dispose: () => void };
  type ResizePayload = { cols: number; rows: number };

  const takeover = vi.fn();
  const terminals: Array<{
    cols: number;
    rows: number;
    open: ReturnType<typeof vi.fn>;
    focus: ReturnType<typeof vi.fn>;
    write: ReturnType<typeof vi.fn>;
    writeln: ReturnType<typeof vi.fn>;
    loadAddon: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
    onData: ReturnType<typeof vi.fn>;
    onResize: ReturnType<typeof vi.fn>;
    emitData: (value: string) => void;
    emitResize: (cols: number, rows: number) => void;
  }> = [];
  const addons: Array<{ fit: ReturnType<typeof vi.fn> }> = [];
  const observers: Array<{ observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }> = [];

  class TerminalImpl {
    cols = 120;
    rows = 32;
    open = vi.fn();
    focus = vi.fn();
    write = vi.fn();
    writeln = vi.fn();
    loadAddon = vi.fn();
    dispose = vi.fn();

    private onDataHandler: ((value: string) => void) | null = null;
    private onResizeHandler: ((value: ResizePayload) => void) | null = null;

    constructor() {
      terminals.push(this as unknown as (typeof terminals)[number]);
    }

    onData = vi.fn((handler: (value: string) => void): TerminalDisposable => {
      this.onDataHandler = handler;
      return {
        dispose: vi.fn(),
      };
    });

    onResize = vi.fn((handler: (value: ResizePayload) => void): TerminalDisposable => {
      this.onResizeHandler = handler;
      return {
        dispose: vi.fn(),
      };
    });

    emitData(value: string) {
      this.onDataHandler?.(value);
    }

    emitResize(cols: number, rows: number) {
      this.cols = cols;
      this.rows = rows;
      this.onResizeHandler?.({ cols, rows });
    }
  }

  class FitAddonImpl {
    fit = vi.fn();

    constructor() {
      addons.push(this as unknown as (typeof addons)[number]);
    }
  }

  class ResizeObserverImpl {
    observe = vi.fn();
    disconnect = vi.fn();

    constructor(_callback: ResizeObserverCallback) {
      observers.push(this as unknown as (typeof observers)[number]);
    }
  }

  return {
    mockGetTakeover: takeover,
    terminalInstances: terminals,
    fitAddonInstances: addons,
    resizeObserverInstances: observers,
    MockTerminal: TerminalImpl,
    MockFitAddon: FitAddonImpl,
    MockResizeObserver: ResizeObserverImpl,
  };
});

vi.mock("@/lib/store/auth-store", () => ({
  useAuthStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      accessToken: "token-1",
    }),
}));

vi.mock("@/lib/api/session", () => ({
  sessionApi: {
    getTakeover: mockGetTakeover,
  },
}));

vi.mock("@xterm/xterm", () => ({
  Terminal: MockTerminal,
}));

vi.mock("@xterm/addon-fit", () => ({
  FitAddon: MockFitAddon,
}));

import { WorkbenchInteractiveTerminal } from "./workbench-interactive-terminal";

type WSHandler<T = unknown> = ((event: T) => void) | null;

class MockWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  binaryType = "blob";
  onopen: WSHandler<Event> = null;
  onmessage: WSHandler<MessageEvent> = null;
  onerror: WSHandler<Event> = null;
  onclose: WSHandler<CloseEvent> = null;
  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
  });

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  emitOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  emitMessage(data: unknown) {
    this.onmessage?.({ data } as MessageEvent);
  }

  emitClose() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent("close"));
  }
}

describe("WorkbenchInteractiveTerminal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    MockWebSocket.instances = [];
    terminalInstances.length = 0;
    fitAddonInstances.length = 0;
    resizeObserverInstances.length = 0;
    mockGetTakeover.mockResolvedValue({
      status: "takeover",
      takeover_id: "tk-1",
    });
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
    vi.stubGlobal("ResizeObserver", MockResizeObserver as unknown as typeof ResizeObserver);
  });

  it("连接后可通过 xterm 输入发送字节，并支持 Ctrl+C", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchInteractiveTerminal
        sessionId="sid-1"
        takeoverId="tk-1"
        active={true}
        panelVisible={true}
      />
    );

    await act(async () => {});

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeDefined();
    await act(async () => {
      ws?.emitOpen();
    });

    const terminal = terminalInstances.at(-1);
    expect(terminal).toBeDefined();
    await act(async () => {
      terminal?.emitData("ls\n");
    });

    const commandPayload = ws?.send.mock.calls.find((call) =>
      ArrayBuffer.isView(call[0])
    )?.[0] as Uint8Array;
    expect(commandPayload).toBeDefined();
    expect(new TextDecoder().decode(commandPayload)).toBe("ls\n");

    await user.click(screen.getByRole("button", { name: "Ctrl+C" }));
    const binaryCalls = ws?.send.mock.calls.filter((call) => ArrayBuffer.isView(call[0]));
    const ctrlCPayload = binaryCalls?.[1]?.[0] as Uint8Array;
    expect(Array.from(ctrlCPayload)).toEqual([0x03]);
  });

  it("xterm 尺寸变化时发送 resize 文本帧", async () => {
    render(
      <WorkbenchInteractiveTerminal
        sessionId="sid-resize"
        takeoverId="tk-1"
        active={true}
        panelVisible={true}
      />
    );

    const ws = MockWebSocket.instances[0];
    const terminal = terminalInstances.at(-1);

    await act(async () => {
      ws?.emitOpen();
      terminal?.emitResize(132, 40);
    });

    const resizeCalls = ws?.send.mock.calls
      .filter(
        (call) =>
          typeof call[0] === "string" &&
          JSON.parse(call[0] as string).type === "resize"
      )
      .map((call) => JSON.parse(call[0] as string));
    const resizeCall = resizeCalls.at(-1);
    expect(resizeCall).toBeDefined();
    expect(resizeCall).toEqual({
      type: "resize",
      cols: 132,
      rows: 40,
    });
  });

  it("连接异常关闭后按 1s 退避重连", async () => {
    vi.useFakeTimers();
    render(
      <WorkbenchInteractiveTerminal
        sessionId="sid-2"
        takeoverId="tk-1"
        active={true}
        panelVisible={true}
      />
    );

    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws?.emitOpen();
      ws?.emitClose();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(mockGetTakeover).toHaveBeenCalledWith("sid-2");
    expect(MockWebSocket.instances.length).toBe(2);
  });

  it("收到 lease_expired 状态后触发接管失效回调", () => {
    const onTakeoverInvalidated = vi.fn();
    render(
      <WorkbenchInteractiveTerminal
        sessionId="sid-3"
        takeoverId="tk-1"
        active={true}
        panelVisible={true}
        onTakeoverInvalidated={onTakeoverInvalidated}
      />
    );

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws?.emitOpen();
      ws?.emitMessage(JSON.stringify({ type: "status", state: "lease_expired" }));
    });

    expect(onTakeoverInvalidated).toHaveBeenCalledWith("接管租约已失效");
  });
});
