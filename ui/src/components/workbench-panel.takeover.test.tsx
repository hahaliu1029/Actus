import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockStartTakeover,
  mockEndTakeover,
  mockRejectTakeover,
  mockRenewTakeover,
  mockReopenTakeover,
  mockFetchSessionById,
  mockSetMessage,
  getSandboxStoreOverrides,
  setSandboxStoreOverrides,
} = vi.hoisted(() => {
  // SPM Task 22: a mutable slice the mocked useSessionStore merges in, so each
  // sandbox test can inject `currentSession` + `sandboxBadge` without perturbing
  // the takeover tests (which leave it `{}` → currentSession stays null).
  let sandboxOverrides: Record<string, unknown> = {};
  return {
    mockStartTakeover: vi.fn(),
    mockEndTakeover: vi.fn(),
    mockRejectTakeover: vi.fn(),
    mockRenewTakeover: vi.fn(),
    mockReopenTakeover: vi.fn(),
    mockFetchSessionById: vi.fn(),
    mockSetMessage: vi.fn(),
    getSandboxStoreOverrides: () => sandboxOverrides,
    setSandboxStoreOverrides: (next: Record<string, unknown>) => {
      sandboxOverrides = next;
    },
  };
});

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
  }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock("@/hooks/use-mobile", () => ({
  useIsMobile: () => false,
}));

vi.mock("@/hooks/use-shell-preview", () => ({
  useShellPreview: () => ({
    consoleRecords: [],
    output: "",
    loading: false,
    error: null,
  }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      fetchSessionById: mockFetchSessionById,
      currentSession: null,
      sandboxBadge: "none",
      ...getSandboxStoreOverrides(),
    }),
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setMessage: mockSetMessage,
    }),
}));

vi.mock("@/lib/api/session", () => ({
  sessionApi: {
    startTakeover: mockStartTakeover,
    endTakeover: mockEndTakeover,
    rejectTakeover: mockRejectTakeover,
    renewTakeover: mockRenewTakeover,
    reopenTakeover: mockReopenTakeover,
  },
}));

vi.mock("./workbench-browser-preview", () => ({
  WorkbenchBrowserPreview: () => <div data-testid="browser-preview" />,
}));

vi.mock("./workbench-terminal-preview", () => ({
  WorkbenchTerminalPreview: () => <div data-testid="terminal-preview" />,
}));

vi.mock("./workbench-interactive-terminal", () => ({
  WorkbenchInteractiveTerminal: () => <div data-testid="interactive-terminal" />,
}));

vi.mock("./workbench-timeline", () => ({
  WorkbenchTimeline: () => <div data-testid="workbench-timeline" />,
}));

vi.mock("./vnc-viewer", () => ({
  VNCViewer: ({ url }: { url: string }) => <div data-testid="vnc-viewer" data-url={url} />,
}));

vi.mock("@/lib/store/auth-store", () => ({
  useAuthStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ accessToken: "test-token" }),
}));

vi.mock("@/lib/vnc/url", () => ({
  buildVNCProxyUrl: (sessionId: string, token: string) =>
    `wss://test/api/sessions/${sessionId}/vnc?token=${token}`,
}));

import { WorkbenchPanel } from "./workbench-panel";

describe("WorkbenchPanel takeover controls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    setSandboxStoreOverrides({});
    mockStartTakeover.mockResolvedValue({
      status: "running",
      request_status: "starting",
      scope: "shell",
    });
    mockEndTakeover.mockResolvedValue({
      status: "running",
      handoff_mode: "continue",
    });
    mockRejectTakeover.mockResolvedValue({
      status: "running",
      reason: "continue",
    });
    mockRenewTakeover.mockResolvedValue({
      status: "takeover",
      request_status: "renewed",
      takeover_id: "tk_renew_1",
    });
    mockReopenTakeover.mockResolvedValue({
      status: "takeover_pending",
      request_status: "reopened",
      reason: null,
      remaining_seconds: 240,
    });
    mockFetchSessionById.mockResolvedValue(undefined);

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => "visible",
    });
  });

  it("running 状态显示主动接管入口并可选择浏览器接管", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchPanel
        sessionId="sid-running"
        status="running"
        takeoverId={null}
        takeoverScope={null}
        takeoverExpiresAt={null}
        snapshots={[]}
        running={true}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    await user.click(screen.getByRole("button", { name: "主动接管" }));
    await user.click(await screen.findByRole("menuitem", { name: "接管浏览器" }));

    expect(mockStartTakeover).toHaveBeenCalledWith("sid-running", { scope: "browser" });
    expect(mockFetchSessionById).toHaveBeenCalledWith("sid-running", { silent: true });
  });

  it("takeover 状态显示结束接管按钮", () => {
    render(
      <WorkbenchPanel
        sessionId="sid-takeover"
        status="takeover"
        takeoverId="tk_takeover_1"
        takeoverScope="shell"
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    expect(screen.getByRole("button", { name: "结束接管" })).toBeInTheDocument();
  });

  it("takeover_pending 状态显示拒绝接管入口", () => {
    render(
      <WorkbenchPanel
        sessionId="sid-pending"
        status="takeover_pending"
        takeoverId="tk_pending_1"
        takeoverScope="shell"
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    expect(screen.getByRole("button", { name: "拒绝接管（继续执行）" })).toBeInTheDocument();
  });

  it("takeover(shell) 且在终端模式时展示交互终端组件", () => {
    render(
      <WorkbenchPanel
        sessionId="sid-interactive-shell"
        status="takeover"
        takeoverId="tk_shell_1"
        takeoverScope="shell"
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    expect(screen.getByTestId("interactive-terminal")).toBeInTheDocument();
    expect(screen.queryByTestId("terminal-preview")).not.toBeInTheDocument();
  });

  it("takeover(browser) 时展示内嵌 VNC 查看器", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchPanel
        sessionId="sid-interactive-browser"
        status="takeover"
        takeoverId="tk_browser_1"
        takeoverScope="browser"
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    await user.click(screen.getByRole("button", { name: "浏览器" }));
    expect(screen.getByTestId("vnc-viewer")).toBeInTheDocument();
    expect(screen.queryByTestId("interactive-terminal")).not.toBeInTheDocument();
  });

  it("takeover(browser) 时切换到终端标签显示静态终端预览", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchPanel
        sessionId="sid-browser-shell-switch"
        status="takeover"
        takeoverId="tk_browser_2"
        takeoverScope="browser"
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    await user.click(screen.getByRole("button", { name: "终端" }));
    expect(screen.getByTestId("terminal-preview")).toBeInTheDocument();
    // VNC viewer is still in the DOM but hidden via CSS (display:none)
    const vncViewer = screen.queryByTestId("vnc-viewer");
    if (vncViewer) {
      expect(vncViewer.closest(".hidden")).toBeTruthy();
    }
  });

  it("takeover 状态应按策略续期，并在页面隐藏时暂停、恢复时立即续期", async () => {
    vi.useFakeTimers();

    render(
      <WorkbenchPanel
        sessionId="sid-renew"
        status="takeover"
        takeoverId="tk_renew_1"
        takeoverScope="shell"
        takeoverExpiresAt={Math.floor(Date.now() / 1000) + 900}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    // 首次续期用较短初始延迟（min(renewIntervalMs, 5_000) = 5_000）
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(mockRenewTakeover).toHaveBeenCalledWith("sid-renew", {
      takeover_id: "tk_renew_1",
    });
    expect(mockRenewTakeover).toHaveBeenCalledTimes(1);

    // 后续续期按 renewIntervalMs = 300_000
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300_000);
    });
    expect(mockRenewTakeover).toHaveBeenCalledTimes(2);

    await act(async () => {
      Object.defineProperty(document, "visibilityState", {
        configurable: true,
        get: () => "hidden",
      });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    await vi.advanceTimersByTimeAsync(300_000);
    expect(mockRenewTakeover).toHaveBeenCalledTimes(2);

    await act(async () => {
      Object.defineProperty(document, "visibilityState", {
        configurable: true,
        get: () => "visible",
      });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    // 恢复可见后立即触发续期（无需等待延迟）
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(mockRenewTakeover).toHaveBeenCalledTimes(3);
  });

  it("续期响应包含 expires_at 时应按最新 TTL 调整下一次续期间隔", async () => {
    vi.useFakeTimers();
    mockRenewTakeover.mockResolvedValue({
      status: "takeover",
      request_status: "renewed",
      takeover_id: "tk_renew_1",
      expires_at: Math.floor(Date.now() / 1000) + 120,
    });

    render(
      <WorkbenchPanel
        sessionId="sid-renew-expire-at"
        status="takeover"
        takeoverId="tk_renew_1"
        takeoverScope="shell"
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    // 首次续期用初始短延迟（min(renewIntervalMs, 5_000) = 5_000）
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(mockRenewTakeover).toHaveBeenCalledTimes(1);

    // 续期响应 expires_at=+120s => TTL=120_000 => 新 interval=max(10_000, min(300_000, 40_000))=40_000
    await act(async () => {
      await vi.advanceTimersByTimeAsync(41_000);
    });
    expect(mockRenewTakeover).toHaveBeenCalledTimes(2);
  });

  it("首次延迟应与剩余 TTL 绑定，TTL<5s 时几乎立即续期", async () => {
    vi.useFakeTimers();

    render(
      <WorkbenchPanel
        sessionId="sid-short-ttl"
        status="takeover"
        takeoverId="tk_short_1"
        takeoverScope="shell"
        takeoverExpiresAt={Math.floor(Date.now() / 1000) + 3}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    // TTL=3000ms, safetyMargin=2000ms → initialDelay = min(5000, max(0, 3000-2000)) = 1000ms
    expect(mockRenewTakeover).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(mockRenewTakeover).toHaveBeenCalledTimes(1);
  });

  it("completed 状态下按钮可见", () => {
    render(
      <WorkbenchPanel
        sessionId="sid-completed"
        status="completed"
        takeoverId={null}
        takeoverScope={null}
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    expect(screen.getByRole("button", { name: "主动接管" })).toBeInTheDocument();
  });

  it("completed 状态先调用 reopen 再调用 startTakeover", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchPanel
        sessionId="sid-completed-reopen"
        status="completed"
        takeoverId={null}
        takeoverScope={null}
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    await user.click(screen.getByRole("button", { name: "主动接管" }));
    await user.click(await screen.findByRole("menuitem", { name: "接管终端" }));

    expect(mockReopenTakeover).toHaveBeenCalledWith("sid-completed-reopen");
    expect(mockStartTakeover).toHaveBeenCalledTimes(1);
    expect(mockStartTakeover).toHaveBeenCalledWith("sid-completed-reopen", { scope: "shell" });
  });

  it("reopen 成功但 start 失败时会刷新会话并给出正确提示", async () => {
    const user = userEvent.setup();
    mockStartTakeover.mockRejectedValueOnce(new Error("当前状态不支持启动接管"));

    render(
      <WorkbenchPanel
        sessionId="sid-reopen-start-fail"
        status="completed"
        takeoverId={null}
        takeoverScope={null}
        takeoverExpiresAt={null}
        snapshots={[]}
        running={false}
        visible={true}
        onPreviewImage={() => {}}
      />
    );

    await user.click(screen.getByRole("button", { name: "主动接管" }));
    await user.click(await screen.findByRole("menuitem", { name: "接管终端" }));

    expect(mockReopenTakeover).toHaveBeenCalledWith("sid-reopen-start-fail");
    expect(mockFetchSessionById).toHaveBeenCalledWith("sid-reopen-start-fail", { silent: true });
    expect(mockSetMessage).toHaveBeenCalledWith({
      type: "error",
      text: "当前状态不支持启动接管",
    });
  });
});

describe("WorkbenchPanel sandbox provisioning affordance (SPM Task 22)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setSandboxStoreOverrides({});
    // `t()` reads document.documentElement.lang; pin zh for deterministic copy.
    document.documentElement.lang = "zh";
  });

  function renderPanel(over: {
    sandboxMode?: "always" | "on_demand" | "off";
    events?: Array<{ event: string; data: Record<string, unknown> }>;
    sandboxBadge?: "provisioning" | "failed" | "none";
  }) {
    setSandboxStoreOverrides({
      sandboxBadge: over.sandboxBadge ?? "none",
      currentSession: {
        session_id: "sid-sandbox",
        title: null,
        status: "running",
        sandbox_mode: over.sandboxMode,
        events: over.events ?? [],
      },
    });
    return render(
      <WorkbenchPanel
        sessionId="sid-sandbox"
        status="running"
        takeoverId={null}
        takeoverScope={null}
        takeoverExpiresAt={null}
        snapshots={[]}
        running={true}
        visible={true}
        onPreviewImage={() => {}}
      />
    );
  }

  it("always mode renders NOTHING new (byte-zero: no sandbox badge even mid-provisioning)", () => {
    renderPanel({ sandboxMode: "always", sandboxBadge: "provisioning" });
    expect(screen.queryByTestId("sandbox-badge")).toBeNull();
  });

  it("off mode renders no sandbox badge", () => {
    renderPanel({ sandboxMode: "off", sandboxBadge: "provisioning" });
    expect(screen.queryByTestId("sandbox-badge")).toBeNull();
  });

  it("absent sandbox_mode renders no sandbox badge", () => {
    renderPanel({ sandboxMode: undefined, sandboxBadge: "provisioning" });
    expect(screen.queryByTestId("sandbox-badge")).toBeNull();
  });

  it("on_demand + zero sandbox events → notStarted empty-state copy", () => {
    renderPanel({ sandboxMode: "on_demand", events: [], sandboxBadge: "none" });
    const badge = screen.getByTestId("sandbox-badge");
    expect(badge.getAttribute("data-badge")).toBe("notStarted");
    expect(badge.textContent).toContain("沙箱未启动——首次需要时自动创建");
  });

  it("on_demand + provisioning badge → provisioning copy", () => {
    renderPanel({
      sandboxMode: "on_demand",
      events: [
        { event: "sandbox_state_changed", data: { new_state: "creating" } },
      ],
      sandboxBadge: "provisioning",
    });
    const badge = screen.getByTestId("sandbox-badge");
    expect(badge.getAttribute("data-badge")).toBe("provisioning");
    expect(badge.textContent).toContain("沙箱准备中…");
  });

  it("on_demand + failed badge → provisionFailed copy", () => {
    renderPanel({
      sandboxMode: "on_demand",
      events: [
        {
          event: "sandbox_state_changed",
          data: { new_state: "unbound", reason: "provision_failed" },
        },
      ],
      sandboxBadge: "failed",
    });
    const badge = screen.getByTestId("sandbox-badge");
    expect(badge.getAttribute("data-badge")).toBe("failed");
    expect(badge.textContent).toContain("沙箱启动失败，将在下次需要时重试");
  });

  it("on_demand + active sandbox (badge none, events present) → no badge", () => {
    renderPanel({
      sandboxMode: "on_demand",
      events: [
        { event: "sandbox_state_changed", data: { new_state: "active" } },
      ],
      sandboxBadge: "none",
    });
    expect(screen.queryByTestId("sandbox-badge")).toBeNull();
  });

  it("badge is scoped to the viewed session (mismatched currentSession → no badge)", () => {
    setSandboxStoreOverrides({
      sandboxBadge: "provisioning",
      currentSession: {
        session_id: "some-other-session",
        title: null,
        status: "running",
        sandbox_mode: "on_demand",
        events: [],
      },
    });
    render(
      <WorkbenchPanel
        sessionId="sid-sandbox"
        status="running"
        takeoverId={null}
        takeoverScope={null}
        takeoverExpiresAt={null}
        snapshots={[]}
        running={true}
        visible={true}
        onPreviewImage={() => {}}
      />
    );
    expect(screen.queryByTestId("sandbox-badge")).toBeNull();
  });
});
