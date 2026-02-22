import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockReplace = vi.fn();
const mockStopSession = vi.fn();
const mockDeleteSession = vi.fn();
const mockLogout = vi.fn();
const mockUseIsMobile = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockReplace,
  }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      currentSession: { title: "任务标题" },
      stopSession: mockStopSession,
      deleteSession: mockDeleteSession,
    }),
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
    mockUseIsMobile.mockReturnValue(false);
    mockStopSession.mockResolvedValue(undefined);
    mockDeleteSession.mockResolvedValue(undefined);
    vi.clearAllMocks();
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
});
