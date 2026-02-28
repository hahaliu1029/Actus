import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockReplace,
  mockStopSession,
  mockDeleteSession,
  mockFetchSessionById,
  mockSetMessage,
  mockEndTakeover,
  mockLogout,
  mockUseIsMobile,
  mockSessionState,
} = vi.hoisted(() => {
  const stopSession = vi.fn();
  const deleteSession = vi.fn();
  const fetchSessionById = vi.fn();
  return {
    mockReplace: vi.fn(),
    mockStopSession: stopSession,
    mockDeleteSession: deleteSession,
    mockFetchSessionById: fetchSessionById,
    mockSetMessage: vi.fn(),
    mockEndTakeover: vi.fn(),
    mockLogout: vi.fn(),
    mockUseIsMobile: vi.fn(),
    mockSessionState: {
      currentSession: { title: "任务标题", status: "running" },
      stopSession,
      deleteSession,
      fetchSessionById,
    },
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockReplace,
  }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector(mockSessionState),
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setMessage: mockSetMessage,
    }),
}));

vi.mock("@/lib/api/session", () => ({
  sessionApi: {
    endTakeover: mockEndTakeover,
  },
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({
    user: { nickname: "Tester" },
    logout: mockLogout,
  }),
}));

vi.mock("@/hooks/use-mobile", () => ({
  useIsMobile: () => mockUseIsMobile(),
}));

vi.mock("@/components/manus-settings", () => ({
  ManusSettings: () => <button type="button">设置</button>,
}));

import { SessionHeader } from "./session-header";

describe("SessionHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseIsMobile.mockReturnValue(false);
    mockSessionState.currentSession = { title: "任务标题", status: "running" };
    mockStopSession.mockResolvedValue(undefined);
    mockDeleteSession.mockResolvedValue(undefined);
    mockFetchSessionById.mockResolvedValue(undefined);
    mockEndTakeover.mockResolvedValue({
      status: "running",
      handoff_mode: "continue",
    });
  });

  it("桌面端显示返回主页、设置、退出登录、停止、删除", () => {
    render(<SessionHeader sessionId="sid-1" />);

    expect(screen.getByRole("link", { name: "返回主页" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "设置" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "退出登录" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "停止" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除" })).toBeInTheDocument();
  });

  it("移动端通过更多菜单触发退出登录和停止操作", async () => {
    const user = userEvent.setup();
    mockUseIsMobile.mockReturnValue(true);
    render(<SessionHeader sessionId="sid-2" />);

    // 点击更多操作按钮打开菜单
    const moreButton = screen.getByRole("button", { name: "更多操作" });
    await user.click(moreButton);

    // 等待菜单打开并点击退出登录
    const logoutMenuItem = await screen.findByRole("menuitem", { name: "退出登录" });
    await user.click(logoutMenuItem);

    // 再次打开菜单点击停止
    await user.click(moreButton);
    const stopMenuItem = await screen.findByRole("menuitem", { name: "停止" });
    await user.click(stopMenuItem);

    expect(mockLogout).toHaveBeenCalledTimes(1);
    expect(mockStopSession).toHaveBeenCalledWith("sid-2");
  });

  it("顶部不再显示主动接管入口", () => {
    render(<SessionHeader sessionId="sid-3" />);
    expect(screen.queryByRole("button", { name: "主动接管" })).not.toBeInTheDocument();
  });
});
