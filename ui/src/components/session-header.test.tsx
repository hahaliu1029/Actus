import { render as renderComponent, screen } from "@testing-library/react";
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
      currentSession: { title: "任务标题", status: "running" } as {
        title: string;
        status: string;
        session_id?: string;
      },
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

vi.mock("@/components/session-cost-summary", () => ({
  SessionCostSummary: ({ sessionId }: { sessionId: string }) => (
    <span data-testid="session-cost-summary">费用：{sessionId}</span>
  ),
}));

import { Sidebar, SidebarProvider } from "@/components/ui/sidebar";
import { SessionHeader } from "./session-header";

const render = (element: React.ReactNode) =>
  renderComponent(element, { wrapper: SidebarProvider });

describe("SessionHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseIsMobile.mockReturnValue(false);
    mockSessionState.currentSession = { session_id: "sid-1", title: "任务标题", status: "running" };
    mockStopSession.mockResolvedValue(undefined);
    mockDeleteSession.mockResolvedValue(undefined);
    mockFetchSessionById.mockResolvedValue(undefined);
    mockEndTakeover.mockResolvedValue({
      status: "running",
      handoff_mode: "continue",
    });
  });

  it("桌面端聚焦标题、状态和费用，低频操作收进更多菜单", async () => {
    const user = userEvent.setup();
    render(<SessionHeader sessionId="sid-1" />);

    expect(screen.getByRole("heading", { name: "任务标题" })).toBeInTheDocument();
    expect(screen.getByText("执行中")).toBeInTheDocument();
    expect(screen.getByTestId("session-cost-summary")).toHaveTextContent("sid-1");
    expect(screen.getByRole("button", { name: "设置" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "停止" })).toBeInTheDocument();
    expect(screen.queryByText("会话 ID：sid-1")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "删除" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "更多操作" }));
    expect(screen.getByText("会话 ID：sid-1")).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "返回主页" })).toHaveAttribute("href", "/");
    expect(screen.getByRole("menuitem", { name: "退出登录" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "删除" })).toBeInTheDocument();
  });

  it("移动端通过更多菜单退出登录，停止仍可直接操作", async () => {
    const user = userEvent.setup();
    mockUseIsMobile.mockReturnValue(true);
    mockSessionState.currentSession = { session_id: "sid-2", title: "任务标题", status: "running" };
    render(<SessionHeader sessionId="sid-2" />);

    // 点击更多操作按钮打开菜单
    const moreButton = screen.getByRole("button", { name: "更多操作" });
    await user.click(moreButton);

    // 等待菜单打开并点击退出登录
    const logoutMenuItem = await screen.findByRole("menuitem", { name: "退出登录" });
    await user.click(logoutMenuItem);

    await user.click(screen.getByRole("button", { name: "停止" }));

    expect(mockLogout).toHaveBeenCalledTimes(1);
    expect(mockStopSession).toHaveBeenCalledWith("sid-2");
  });

  it("移动端可从会话头打开会话侧栏", async () => {
    const user = userEvent.setup();
    mockUseIsMobile.mockReturnValue(true);
    render(
      <>
        <Sidebar>会话历史内容</Sidebar>
        <SessionHeader sessionId="sid-1" />
      </>
    );
    expect(screen.queryByText("会话历史内容")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "打开会话侧栏" }));
    expect(await screen.findByText("会话历史内容")).toBeVisible();
  });

  it("从菜单删除仍需确认，并删除当前路由会话", async () => {
    const user = userEvent.setup();
    render(<SessionHeader sessionId="sid-1" />);
    await user.click(screen.getByRole("button", { name: "更多操作" }));
    await user.click(screen.getByRole("menuitem", { name: "删除" }));
    expect(mockDeleteSession).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "确认" }));
    expect(mockDeleteSession).toHaveBeenCalledWith("sid-1");
    expect(mockReplace).toHaveBeenCalledWith("/");
  });

  it("顶部不再显示主动接管入口", () => {
    render(<SessionHeader sessionId="sid-3" />);
    expect(screen.queryByRole("button", { name: "主动接管" })).not.toBeInTheDocument();
  });

  // ---- A4-0 follow-up (a): session_id guard regression tests ----

  it("路由切到 B 但 currentSession 仍是 A(takeover) 时，不渲染结束接管", () => {
    mockSessionState.currentSession = {
      session_id: "A",
      status: "takeover",
      title: "任务标题",
    };
    render(<SessionHeader sessionId="B" />);
    expect(
      screen.queryByRole("button", { name: "结束接管" })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "任务标题" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "正在加载会话" })).toBeInTheDocument();
    expect(screen.queryByTestId("session-cost-summary")).not.toBeInTheDocument();
    expect(mockEndTakeover).not.toHaveBeenCalled();
  });

  it("currentSession 与路由一致且为 takeover 时，渲染结束接管并对该 sessionId 调用 endTakeover", async () => {
    const user = userEvent.setup();
    mockSessionState.currentSession = {
      session_id: "B",
      status: "takeover",
      title: "任务标题",
    };
    render(<SessionHeader sessionId="B" />);
    const btn = screen.getByRole("button", { name: "结束接管" });
    await user.click(btn);
    expect(mockEndTakeover).toHaveBeenCalledWith("B", {
      handoff_mode: "continue",
    });
  });

  it("currentSession 与路由一致但非 takeover 时，结束接管隐藏、停止和删除入口仍可用", async () => {
    const user = userEvent.setup();
    mockSessionState.currentSession = {
      session_id: "B",
      status: "running",
      title: "任务标题",
    };
    render(<SessionHeader sessionId="B" />);
    expect(
      screen.queryByRole("button", { name: "结束接管" })
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "停止" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "更多操作" }));
    expect(screen.getByRole("menuitem", { name: "删除" })).toBeInTheDocument();
  });

  it("从不匹配(A)切到匹配(B,takeover)后，结束接管由隐藏变为显示", () => {
    mockSessionState.currentSession = {
      session_id: "A",
      status: "takeover",
      title: "任务标题",
    };
    const { rerender } = render(<SessionHeader sessionId="B" />);
    expect(
      screen.queryByRole("button", { name: "结束接管" })
    ).not.toBeInTheDocument();

    mockSessionState.currentSession = {
      session_id: "B",
      status: "takeover",
      title: "任务标题",
    };
    rerender(<SessionHeader sessionId="B" />);
    expect(
      screen.getByRole("button", { name: "结束接管" })
    ).toBeInTheDocument();
  });

  it("移动端：currentSession 不匹配路由时，不显示结束接管", () => {
    mockUseIsMobile.mockReturnValue(true);
    mockSessionState.currentSession = {
      session_id: "A",
      status: "takeover",
      title: "任务标题",
    };
    render(<SessionHeader sessionId="B" />);
    expect(
      screen.queryByRole("button", { name: "结束接管" })
    ).not.toBeInTheDocument();
  });

  it("移动端：currentSession 匹配路由且 takeover 时，可直接结束接管", async () => {
    const user = userEvent.setup();
    mockUseIsMobile.mockReturnValue(true);
    mockSessionState.currentSession = {
      session_id: "B",
      status: "takeover",
      title: "任务标题",
    };
    render(<SessionHeader sessionId="B" />);
    await user.click(screen.getByRole("button", { name: "结束接管" }));
    expect(mockEndTakeover).toHaveBeenCalledWith("B", { handoff_mode: "continue" });
  });

  it("路由切换时旧会话不可触发当前路由的停止操作", async () => {
    const user = userEvent.setup();
    mockSessionState.currentSession = {
      session_id: "A",
      status: "takeover",
      title: "任务标题",
    };
    render(<SessionHeader sessionId="B" />);
    const stopButton = screen.getByRole("button", { name: "停止" });
    expect(stopButton).toBeDisabled();
    await user.click(stopButton);
    expect(mockStopSession).not.toHaveBeenCalled();
  });

  it("已完成会话不再允许停止", async () => {
    const user = userEvent.setup();
    mockSessionState.currentSession = { session_id: "sid-1", title: "任务标题", status: "completed" };
    render(<SessionHeader sessionId="sid-1" />);
    const stopButton = screen.getByRole("button", { name: "停止" });
    expect(stopButton).toBeDisabled();
    await user.click(stopButton);
    expect(mockStopSession).not.toHaveBeenCalled();
  });

  it("停止失败显示错误并恢复可重试状态", async () => {
    const user = userEvent.setup();
    mockStopSession.mockRejectedValueOnce(new Error("连接中断，请重试"));
    render(<SessionHeader sessionId="sid-1" />);
    await user.click(screen.getByRole("button", { name: "停止" }));
    expect(mockSetMessage).toHaveBeenCalledWith({ type: "error", text: "连接中断，请重试" });
    expect(screen.getByRole("button", { name: "停止" })).toBeEnabled();
  });
});
