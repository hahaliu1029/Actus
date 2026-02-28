import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockStartTakeover,
  mockEndTakeover,
  mockRejectTakeover,
  mockRenewTakeover,
  mockFetchSessionById,
  mockSetMessage,
} = vi.hoisted(() => ({
  mockStartTakeover: vi.fn(),
  mockEndTakeover: vi.fn(),
  mockRejectTakeover: vi.fn(),
  mockRenewTakeover: vi.fn(),
  mockFetchSessionById: vi.fn(),
  mockSetMessage: vi.fn(),
}));

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

import { WorkbenchPanel } from "./workbench-panel";

describe("WorkbenchPanel takeover controls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
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

  it("takeover(browser) 时仍展示浏览器预览，不展示交互终端", async () => {
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
    expect(screen.getByTestId("browser-preview")).toBeInTheDocument();
    expect(screen.queryByTestId("interactive-terminal")).not.toBeInTheDocument();
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
});
