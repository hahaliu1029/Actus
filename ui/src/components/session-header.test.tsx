import { fireEvent, render, screen } from "@testing-library/react";
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
  });

  it("桌面端显示返回主页、设置、退出登录、停止、删除", () => {
    render(<SessionHeader sessionId="sid-1" />);

    expect(screen.getByRole("link", { name: "返回主页" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "设置" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "退出登录" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "停止" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除" })).toBeInTheDocument();
  });

  it("移动端通过更多菜单触发退出登录和停止操作", () => {
    mockUseIsMobile.mockReturnValue(true);
    render(<SessionHeader sessionId="sid-2" />);

    fireEvent.pointerDown(screen.getByRole("button", { name: "更多操作" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "退出登录" }));
    fireEvent.pointerDown(screen.getByRole("button", { name: "更多操作" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "停止" }));

    expect(mockLogout).toHaveBeenCalledTimes(1);
    expect(mockStopSession).toHaveBeenCalledWith("sid-2");
  });
});
